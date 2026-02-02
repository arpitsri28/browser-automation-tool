from __future__ import annotations

from typing import Optional

from agent.state import ReleaseInfo
from tools.vision import VisionClient


class ReleaseExtractor:
    def __init__(self, vision: VisionClient) -> None:
        self._vision = vision

    def extract(self, png_bytes: bytes, repo: str) -> ReleaseInfo:
        data = self._vision.get_release_extract(png_bytes, repo)
        return ReleaseInfo(version=data.version, tag=data.tag, author=data.author)
