from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Stage(str, Enum):
    HOME = "HOME"
    SEARCH_RESULTS = "SEARCH_RESULTS"
    REPO = "REPO"
    RELEASES = "RELEASES"
    EXTRACTED = "EXTRACTED"
    DONE = "DONE"


class ReleaseInfo(BaseModel):
    version: Optional[str] = None
    tag: Optional[str] = None
    author: Optional[str] = None


class AgentState(BaseModel):
    target_repo: str = "openclaw/openclaw"
    start_url: str = "https://github.com"
    prompt: Optional[str] = None
    vlm_model: Optional[str] = None
    router_model: Optional[str] = None
    prompt_model: Optional[str] = None

    current_url: Optional[str] = None
    page_title: Optional[str] = None
    last_png_bytes: Optional[bytes] = None
    last_png_bytes_raw: Optional[bytes] = None
    last_screenshot_path: Optional[str] = None

    step_count: int = 0
    retry_count: int = 0
    max_steps: int = 25
    max_retries_per_stage: int = 3

    last_urls: List[str] = Field(default_factory=list)
    last_screenshot_hashes: List[str] = Field(default_factory=list)

    stage: Stage = Stage.HOME
    last_action: Optional[Dict[str, Any]] = None
    last_bbox: Optional[tuple[int, int, int, int]] = None
    refine_level: int = 0
    pending_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_release: Optional[ReleaseInfo] = None

    run_dir: Optional[str] = None
    next_node: Optional[str] = None

    def push_url(self, url: str, max_keep: int = 6) -> None:
        self.last_urls.append(url)
        if len(self.last_urls) > max_keep:
            self.last_urls = self.last_urls[-max_keep:]

    def push_hash(self, h: str, max_keep: int = 6) -> None:
        self.last_screenshot_hashes.append(h)
        if len(self.last_screenshot_hashes) > max_keep:
            self.last_screenshot_hashes = self.last_screenshot_hashes[-max_keep:]
