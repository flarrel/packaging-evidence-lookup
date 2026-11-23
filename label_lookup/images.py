"""Shared CPU image preprocessing for package art and medical benchmarks."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(source: Path | bytes) -> Image.Image:
    stream = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(stream) as image:
        if image.width * image.height > 12_000_000:
            raise ValueError("Image exceeds the 12-megapixel limit.")
        return ImageOps.exif_transpose(image).convert("RGB")


def unit_pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0
