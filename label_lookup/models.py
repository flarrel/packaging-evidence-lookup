"""Pinned CLIP and FLAN-T5 CPU inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .downloads import download

CLIP = "Xenova/clip-vit-base-patch32"
FLAN = "Xenova/flan-t5-small"
REVISIONS = {
    CLIP: "6b7c523703ae26e39be4b03774966f8571c7f523",
    FLAN: "311454e83bc784267fd7eef5940ee854144abbec",
}
WEIGHTS = {
    CLIP: {
        "text": (64_504_507, "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"),
        "vision": (89_117_001, "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"),
    },
    FLAN: {
        "encoder": (35_759_127, "968ce399b8dcdfd4c36833db51e9f78f327f7272ee9e2777fc2b3c8693208e90"),
        "decoder": (58_943_318, "7f090cbd1281a9c896a070b424ceb18df45da6e78d16895a14c72387708db216"),
    },
}


def model_path(work: Path, name: str) -> Path:
    return work / "models" / name.split("/")[-1] / REVISIONS[name]


def fetch(work: Path) -> None:
    tasks = []
    for model in (CLIP, FLAN):
        revision = REVISIONS[model]
        root = f"https://huggingface.co/{model}/resolve/{revision}"
        files = ["config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
        if model == CLIP:
            files.append("preprocessor_config.json")
        for filename in files:
            tasks.append((f"{root}/{filename}", model_path(work, model) / filename, 3_000_000, ()))
        for kind, (size, sha256) in WEIGHTS[model].items():
            filename = f"{kind}_model_quantized.onnx"
            tasks.append((
                f"{root}/onnx/{filename}", model_path(work, model) / filename, size,
                [("sha256", sha256)],
            ))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda task: download(*task), tasks))


def _session(root: Path, kind: str):
    import onnxruntime as ort

    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(root / f"{kind}_model_quantized.onnx"), options, providers=["CPUExecutionProvider"]
    )


def _run(session, **arrays):
    feed = {item.name: arrays[item.name] for item in session.get_inputs()}
    return dict(zip((item.name for item in session.get_outputs()), session.run(None, feed), strict=True))


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-9)


class Clip:
    def __init__(self, work: Path) -> None:
        from transformers import AutoTokenizer, CLIPImageProcessor

        root = model_path(work, CLIP)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.processor = CLIPImageProcessor.from_pretrained(root, local_files_only=True)
        self.text_session = _session(root, "text")
        self.vision_session = _session(root, "vision")

    def text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, max_length=77, truncation=True, return_tensors="np")
        return _normalize(_run(self.text_session, **inputs)["text_embeds"])

    def image(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="np")
        return _normalize(_run(self.vision_session, **inputs)["image_embeds"])


class Flan:
    def __init__(self, work: Path) -> None:
        from transformers import AutoTokenizer

        root = model_path(work, FLAN)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.encoder = _session(root, "encoder")
        self.decoder = _session(root, "decoder")

    def answer(self, question: str, passage: str) -> str:
        prompt = f"question: {question} context: {passage}"
        inputs = self.tokenizer(prompt, max_length=256, truncation=True, return_tensors="np")
        hidden = _run(self.encoder, **inputs)["last_hidden_state"]
        tokens = [0]
        for _ in range(36):
            output = _run(
                self.decoder,
                input_ids=np.array([tokens], dtype=np.int64),
                encoder_hidden_states=hidden,
                encoder_attention_mask=inputs["attention_mask"],
            )
            token = int(np.argmax(output["logits"][0, -1]))
            if token == self.tokenizer.eos_token_id:
                break
            tokens.append(token)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
