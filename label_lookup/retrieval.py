"""Lexical, CLIP and learned ranking over the same package records."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models

from .corpus import Record, load
from .images import load_image
from .models import CLIP, REVISIONS, Clip


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25:
    def __init__(self, documents: list[str]) -> None:
        self.documents = [Counter(tokens(document)) for document in documents]
        self.lengths = np.array([sum(document.values()) for document in self.documents])
        self.average_length = max(float(self.lengths.mean()), 1.0)
        self.frequency = Counter(term for document in self.documents for term in document)

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        for term in set(tokens(query)):
            containing = self.frequency[term]
            if containing == 0:
                continue
            inverse_frequency = math.log1p((len(scores) - containing + 0.5) / (containing + 0.5))
            frequency = np.array([document[term] for document in self.documents])
            normalizer = 1.2 * (0.25 + 0.75 * self.lengths / self.average_length)
            scores += inverse_frequency * (frequency * 2.2 / (frequency + normalizer))
        return scores


def reciprocal_rank_fusion(lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ordered in lists:
        for position, item in enumerate(ordered, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return scores


@dataclass
class Features:
    lexical: np.ndarray
    dense: np.ndarray
    visual: np.ndarray
    has_image: bool

    def matrix(self) -> np.ndarray:
        maximum = max(float(self.lexical.max()), 1.0)
        return np.column_stack((
            self.lexical / maximum, self.dense, self.visual, np.full_like(self.visual, self.has_image),
        )).astype(np.float32)


class Index:
    def __init__(self, work: Path) -> None:
        self.work = work
        self.records, self.splits = load(work)
        self.positions = {record.id: position for position, record in enumerate(self.records)}
        self.lexical = BM25([record.text for record in self.records])
        digest = hashlib.sha256(REVISIONS[CLIP].encode())
        for record in self.records:
            for value in (record.id, record.family, record.text, record.image_sha256):
                digest.update(value.encode() + b"\0")
        self.fingerprint = digest.hexdigest()
        self.clip = Clip(work)
        self.text_vectors, self.image_vectors = self._embeddings()
        self.database = QdrantClient(path=str(work / "index"))
        self.collection = f"labels_{self.fingerprint[:16]}"
        if not self.database.collection_exists(self.collection):
            definition = {
                name: models.VectorParams(size=self.text_vectors.shape[1], distance=models.Distance.COSINE)
                for name in ("text", "image")
            }
            self.database.create_collection(self.collection, vectors_config=definition)
        if self.database.count(self.collection, exact=True).count != len(self.records):
            self.database.upsert(self.collection, wait=True, points=[
                models.PointStruct(
                    id=record.id, vector={"text": text.tolist(), "image": image.tolist()},
                    payload={"family": record.family},
                )
                for record, text, image in zip(self.records, self.text_vectors, self.image_vectors, strict=True)
            ])

    def close(self) -> None:
        self.database.close()

    def _embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        cache = self.work / f"embeddings_{self.fingerprint[:16]}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                return saved["text"], saved["image"]
        images, texts = [], []
        for start in range(0, len(self.records), 8):
            batch = self.records[start:start + 8]
            images.append(self.clip.image([load_image(self.work / record.image) for record in batch]))
            texts.append(self.clip.text([record.text for record in batch]))
        text, image = np.concatenate(texts), np.concatenate(images)
        with tempfile.NamedTemporaryFile(dir=self.work, delete=False) as temporary:
            staging = Path(temporary.name)
            np.savez_compressed(temporary, text=text, image=image)
        staging.replace(cache)
        return text, image

    def query_vector(self, vector: np.ndarray, kind: str, family: str | None = None) -> np.ndarray:
        criteria = None
        if family is not None:
            criteria = models.Filter(must=[
                models.FieldCondition(key="family", match=models.MatchValue(value=family))
            ])
        result = self.database.query_points(
            self.collection, query=vector.tolist(), using=kind, query_filter=criteria,
            limit=len(self.records), with_payload=False,
        )
        values = np.full(len(self.records), -1.0, dtype=np.float32)
        for point in result.points:
            values[self.positions[str(point.id)]] = point.score
        return values

    def features(self, query: str, image: Image.Image | None = None) -> Features:
        text = self.clip.text([query])[0]
        dense = self.query_vector(text, "text")
        visual = np.zeros(len(self.records), dtype=np.float32)
        if image is not None:
            visual = self.query_vector(self.clip.image([image])[0], "image")
        return Features(self.lexical.score(query), dense, visual, image is not None)

    def order(self, scores: np.ndarray) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.records[i].id))

    def fused(self, features: Features) -> list[int]:
        ranked = [self.order(features.dense)]
        lexical = [i for i in self.order(features.lexical) if features.lexical[i] > 0]
        if lexical:
            ranked.append(lexical)
        if features.has_image:
            ranked.append(self.order(features.visual))
        scores = reciprocal_rank_fusion(ranked)
        return sorted(scores, key=lambda i: (-scores[i], self.records[i].id))
