"""Train and reload a small CPU PyTorch ranker."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .benchmark_data import answerable, query_image
from .retrieval import Index

EPOCHS = 80
SEED = 19


def network(kind: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(4, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 1))
    raise ValueError(f"Unknown ranker {kind}")


def _examples(index: Index, split: str) -> torch.Tensor:
    families = set(index.splits[split])
    allowed = np.array([record.family in families for record in index.records])
    pairs = []
    for record in index.records:
        if record.family not in families:
            continue
        for case in answerable(record):
            matrix = index.features(case.question, query_image(case, index.work)).matrix()
            position = index.positions[record.id]
            others = allowed.copy()
            others[position] = False
            negatives = matrix[others]
            positives = np.broadcast_to(matrix[position], negatives.shape)
            pairs.append(np.stack((positives, negatives), axis=1))
    return torch.from_numpy(np.concatenate(pairs))


def train(index: Index) -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    train_pairs = _examples(index, "train")
    validation_pairs = _examples(index, "validation")
    summary = {}
    for kind in ("linear", "mlp"):
        ranker = network(kind)
        optimizer = torch.optim.AdamW(ranker.parameters(), lr=0.035)
        summary[kind] = _fit(index, kind, ranker, optimizer, train_pairs, validation_pairs)
    return summary


def _fit(index, kind, ranker, optimizer, train_pairs, validation_pairs):
    best = float("inf")
    chosen = 0
    weights = {}
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = _loss(ranker, train_pairs)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation = float(_loss(ranker, validation_pairs))
        if validation < best:
            best = validation
            chosen = epoch
            weights = {key: value.detach().clone() for key, value in ranker.state_dict().items()}
    if not weights:
        raise ValueError("Training produced no finite validation loss.")
    torch.save({
        "weights": weights, "fingerprint": index.fingerprint, "kind": kind, "epoch": chosen,
    }, index.work / f"ranker_{kind}.pt")
    return {"epoch": chosen, "validation_loss": best, "training_pairs": len(train_pairs)}


def _loss(ranker: nn.Module, pairs: torch.Tensor) -> torch.Tensor:
    predictions = ranker(pairs)
    return nn.functional.softplus(predictions[:, 1] - predictions[:, 0]).mean()


def load_ranker(index: Index, kind: str) -> nn.Module:
    path: Path = index.work / f"ranker_{kind}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["fingerprint"] != index.fingerprint or checkpoint["kind"] != kind:
        raise ValueError("Ranker checkpoint belongs to a different corpus or architecture. Run train again.")
    model = network(kind)
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


def score(model: nn.Module, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(features)).squeeze(-1).numpy()
