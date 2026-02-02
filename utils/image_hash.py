from __future__ import annotations

from PIL import Image


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    img = image.convert("L").resize((hash_size, hash_size))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ["1" if p >= avg else "0" for p in pixels]
    hex_string = ""
    for i in range(0, len(bits), 4):
        nibble = bits[i : i + 4]
        hex_string += f"{int(''.join(nibble), 2):x}"
    return hex_string
