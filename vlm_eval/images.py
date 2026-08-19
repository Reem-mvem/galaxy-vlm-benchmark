"""Image loading and preprocessing helpers."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def read_image_bytes(img_path: str) -> bytes:
    with open(img_path, "rb") as f:
        return f.read()


def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def percentile_normalized_jpeg(img_path: str, size: int = 424, quality: int = 92) -> bytes:
    """Percentile-normalize each channel (1st–99th), resize, return JPEG bytes.

    Used for GZ Rings, whose raw cutouts have a very wide dynamic range that
    hides faint ring features unless stretched.
    """
    img = np.array(Image.open(img_path).convert("RGB")).astype(float)
    for c in range(3):
        lo, hi = np.percentile(img[:, :, c], [1, 99])
        img[:, :, c] = np.clip((img[:, :, c] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    pil = Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
