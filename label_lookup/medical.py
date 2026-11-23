"""Reproducible preprocessing for the public PneumoniaMNIST benchmark."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .downloads import download
from .images import unit_pixels

URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
CHECKSUM = "28209eda62fecd6e6a2d98b1501bb15f"
SPLIT_SIZES = {"train": 4708, "val": 524, "test": 624}


def run(work: Path) -> dict:
    source = download(URL, work / "data/pneumoniamnist.npz", 5_000_000, [("md5", CHECKSUM)])
    output = work / "medical"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": "PneumoniaMNIST", "source": URL, "md5": CHECKSUM, "splits": {}}
    processed = {}
    with np.load(source, allow_pickle=False) as archive:
        for split, size in SPLIT_SIZES.items():
            images = archive[f"{split}_images"]
            labels = archive[f"{split}_labels"]
            if images.dtype != np.uint8 or images.shape != (size, 28, 28):
                raise ValueError(f"Unexpected {split} image dimensions or dtype")
            if labels.shape != (size, 1) or not np.isin(labels, [0, 1]).all():
                raise ValueError(f"Invalid {split} labels")
            pixels = np.stack([unit_pixels(Image.fromarray(image)) for image in images])
            processed[f"{split}_images"] = pixels[:, None]
            processed[f"{split}_labels"] = labels[:, 0]
            counts = np.bincount(labels[:, 0], minlength=2)
            summary["splits"][split] = {
                "shape": list(processed[f"{split}_images"].shape),
                "labels": {"normal": int(counts[0]), "pneumonia": int(counts[1])},
                "pixel_range": [float(pixels.min()), float(pixels.max())],
            }
        train_images = archive["train_images"]
        train_labels = archive["train_labels"][:, 0]
        grid = Image.new("L", (4 * 56, 2 * 56))
        for label in range(2):
            for column, index in enumerate(np.flatnonzero(train_labels == label)[:4]):
                tile = Image.fromarray(train_images[index]).resize((56, 56), Image.Resampling.NEAREST)
                grid.paste(tile, (column * 56, label * 56))
        grid.save(output / "samples.png")
    np.savez_compressed(output / "processed.npz", **processed)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
