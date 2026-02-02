from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os

from playwright.sync_api import Browser, Page, sync_playwright


@dataclass
class Observation:
    url: str
    title: str
    screenshot_png: bytes
    viewport: Dict[str, int]


class BrowserSession:
    def __init__(
        self,
        headed: bool = False,
        slow_mo_ms: int = 0,
        action_delay_ms: int = 250,
        user_data_dir: Optional[str] = None,
    ) -> None:
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._headed = headed
        self._slow_mo_ms = slow_mo_ms
        self._action_delay_s = action_delay_ms / 1000.0
        self._user_data_dir = user_data_dir

    def start(self, url: str) -> None:
        self._playwright = sync_playwright().start()
        if self._user_data_dir:
            os.makedirs(self._user_data_dir, exist_ok=True)
            context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=self._user_data_dir,
                headless=not self._headed,
                slow_mo=self._slow_mo_ms,
                viewport={"width": 1280, "height": 720},
            )
            self._browser = context.browser
            self._page = context.pages[0] if context.pages else context.new_page()
        else:
            self._browser = self._playwright.chromium.launch(headless=not self._headed, slow_mo=self._slow_mo_ms)
            self._page = self._browser.new_page(viewport={"width": 1280, "height": 720})
        self._page.goto(url, wait_until="domcontentloaded")

    def close(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser not started")
        return self._page

    def observe(self) -> Observation:
        page = self.page
        viewport = page.viewport_size or {"width": 1280, "height": 720}
        time.sleep(1.0)
        try:
            title = page.title()
        except Exception:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
                title = page.title()
            except Exception:
                title = ""
        png = page.screenshot(full_page=False)
        return Observation(url=page.url, title=title, screenshot_png=png, viewport=viewport)

    def wait_for_idle(self, timeout_ms: int = 8000) -> None:
        page = self.page
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            except Exception:
                pass
        self._delay()

    def back(self) -> None:
        page = self.page
        page.go_back(wait_until="domcontentloaded")
        self._delay()

    def press_key(self, key: str) -> None:
        self.page.keyboard.press(key)
        self._delay()

    def type_text(self, text: str) -> None:
        self.page.keyboard.type(text, delay=20)
        self._delay()
        time.sleep(0.2)

    def scroll(self, direction: str, amount: int = 400) -> None:
        sign = 1 if direction == "down" else -1
        self.page.mouse.wheel(0, sign * amount)
        self._delay()

    def click_bbox(self, bbox: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        self._ensure_in_viewport(cx, cy)
        jitter_x = max(-2, min(2, (x2 - x1) // 10))
        jitter_y = max(-2, min(2, (y2 - y1) // 10))
        self.page.mouse.click(cx + jitter_x, cy + jitter_y)
        self._delay()

    def click_point(self, x: int, y: int) -> None:
        self._ensure_in_viewport(x, y)
        self.page.mouse.click(x, y)
        self._delay()

    def type_into_bbox(self, bbox: Tuple[int, int, int, int], text: str) -> None:
        self.click_bbox(bbox)
        time.sleep(0.1)
        self.type_text(text)

    def _ensure_in_viewport(self, x: int, y: int) -> None:
        viewport = self.page.viewport_size or {"width": 1280, "height": 720}
        width = viewport["width"]
        height = viewport["height"]
        if x < 0 or x > width or y < 0 or y > height:
            self.scroll("down", 500)
            time.sleep(0.1)

    def _delay(self) -> None:
        if self._action_delay_s > 0:
            time.sleep(self._action_delay_s)
