from __future__ import annotations

from io import BytesIO
from typing import Tuple

from PIL import Image


def bbox_in_valid_column(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> bool:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return False
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    header_frac = 0.10
    left_frac = 0.16
    right_frac = 0.82

    if cy < img_h * header_frac:
        return False
    if cx < img_w * left_frac:
        return False
    if cx > img_w * right_frac:
        return False
    return True


def looks_like_blue_link(png_bytes: bytes, bbox: Tuple[int, int, int, int]) -> bool:
    ratio = _blue_ratio(png_bytes, bbox)
    return ratio >= _blue_ratio_threshold(bbox)


def blue_ratio(png_bytes: bytes, bbox: Tuple[int, int, int, int]) -> float:
    return _blue_ratio(png_bytes, bbox)


def _blue_ratio_threshold(bbox: Tuple[int, int, int, int]) -> float:
    _, y1, _, y2 = bbox
    h = max(0, y2 - y1)
    if h <= 40:
        return 0.006
    return 0.008


def _blue_ratio(png_bytes: bytes, bbox: Tuple[int, int, int, int]) -> float:
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return 0.0
    region = img.crop((x1, y1, x2, y2))
    pixels = list(region.getdata())
    if not pixels:
        return 0.0
    blueish = 0
    for r, g, b in pixels:
        if (b > g + 30) and (b > r + 40) and (b > 90):
            blueish += 1
    return blueish / len(pixels)


def _clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2
