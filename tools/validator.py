from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image
from io import BytesIO

from agent.state import AgentState, Stage
from utils.image_hash import average_hash


@dataclass
class ValidationResult:
    new_stage: Optional[Stage] = None
    recovery_action: Optional[dict] = None
    should_extract: bool = False
    should_stop: bool = False
    reason: Optional[str] = None


class Validator:
    def __init__(self, repeat_threshold: int = 3) -> None:
        self._repeat_threshold = repeat_threshold

    def assess(self, state: AgentState, png_bytes: bytes) -> Tuple[ValidationResult, str]:
        img = Image.open(BytesIO(png_bytes))
        h = average_hash(img)
        state.push_hash(h)
        if state.current_url:
            state.push_url(state.current_url)

        repeated_hashes = state.last_screenshot_hashes.count(h)
        repeated_urls = state.current_url and state.last_urls.count(state.current_url) or 0
        stuck = repeated_hashes >= self._repeat_threshold and repeated_urls >= self._repeat_threshold

        result = ValidationResult()
        if state.step_count >= state.max_steps:
            result.should_stop = True
            result.reason = "max_steps"
            return result, h

        if stuck:
            recovery_step = state.retry_count % 3
            if recovery_step == 0:
                result.recovery_action = {
                    "type": "scroll",
                    "reason": "stuck_recovery_scroll_down",
                    "scroll": {"direction": "down", "amount": 500},
                }
            elif recovery_step == 1:
                result.recovery_action = {
                    "type": "scroll",
                    "reason": "stuck_recovery_scroll_up",
                    "scroll": {"direction": "up", "amount": 400},
                }
            else:
                result.recovery_action = {
                    "type": "back",
                    "reason": "stuck_recovery_back",
                }
            result.reason = "stuck"
            return result, h

        # Heuristic stage progression based on URL
        if state.current_url:
            url = state.current_url
            if "github.com/search" in url and state.stage == Stage.HOME:
                result.new_stage = Stage.SEARCH_RESULTS
            if state.target_repo.lower() in url.lower() and state.stage in (Stage.SEARCH_RESULTS, Stage.HOME):
                result.new_stage = Stage.REPO
            if "/releases" in url and state.stage in (Stage.REPO, Stage.SEARCH_RESULTS):
                result.new_stage = Stage.RELEASES
                result.should_extract = True

        return result, h
