from __future__ import annotations

from typing import Any, Dict, Tuple, List
import os
import time
from io import BytesIO
from PIL import Image, ImageDraw

from langgraph.graph import END, StateGraph

from agent.state import AgentState, Stage
from tools.browser import BrowserSession
from tools.extractor import ReleaseExtractor
from tools.validator import Validator
from tools.vision import Action, VisionClient
from utils.logging import log_event
from utils.bbox_guards import bbox_in_valid_column, blue_ratio
from utils.trace import TraceWriter


def build_graph(
    browser: BrowserSession,
    vision: VisionClient,
    validator: Validator,
    extractor: ReleaseExtractor,
    tracer: TraceWriter,
    logger,
):
    graph = StateGraph(dict)

    def observe_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState.model_validate(state_dict)
        obs = browser.observe()
        state.current_url = obs.url
        state.page_title = obs.title
        state.last_png_bytes = obs.screenshot_png
        state.last_png_bytes_raw = obs.screenshot_png
        if state.step_count == 0 and not state.run_dir:
            state.run_dir = tracer.run_dir
        if state.run_dir:
            path = tracer.save_image(state.step_count + 1, obs.screenshot_png)
            state.last_screenshot_path = path
            tracer.save_json(
                state.step_count + 1,
                "observation",
                {
                    "url": obs.url,
                    "title": obs.title,
                    "stage": state.stage,
                },
            )
        state.next_node = "decide"
        log_event(logger, "observe", url=obs.url, title=obs.title, stage=state.stage)
        return state.model_dump()

    def decide_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState.model_validate(state_dict)
        if state.current_url and "/search" in state.current_url:
            state.stage = Stage.SEARCH_RESULTS
        subgoal = _subgoal_for_stage(state)
        raw = state.last_png_bytes_raw or b""
        if state.pending_candidates and state.stage == Stage.SEARCH_RESULTS:
            action = Action.model_validate(
                {"type": "click", "reason": "pending_candidate", "bbox": state.pending_candidates[0]["bbox"]}
            )
        elif state.retry_count > 0:
            critique_goal = (
                subgoal
                + " The previous attempt did not change the URL. Return a corrected bbox or a new action using the current screenshot."
            )
            action = vision.get_action(raw, critique_goal, state.stage.value)
        else:
            action = vision.get_action(raw, subgoal, state.stage.value)
        state.last_action = action.model_dump()
        if action.bbox:
            state.last_bbox = tuple(action.bbox)
        if action.bbox and state.last_png_bytes:
            _save_bbox_overlay(state.last_png_bytes, action.bbox, state.step_count + 1)
        if state.run_dir:
            tracer.save_json(state.step_count + 1, "action", state.last_action)
        state.next_node = "act"
        log_event(logger, "decide", subgoal=subgoal, action=state.last_action)
        return state.model_dump()

    def act_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState.model_validate(state_dict)
        action = Action.model_validate(state.last_action or {"type": "noop", "reason": "missing"})
        if state.stage == Stage.SEARCH_RESULTS and action.type == "click" and action.bbox:
            success = _explore_click_points_multi(browser, tuple(action.bbox), state.target_repo, logger)
            if success:
                state.stage = Stage.REPO
            state.step_count += 1
            state.next_node = "validate"
            log_event(logger, "act", action=state.last_action, step=state.step_count)
            return state.model_dump()
        if state.stage == Stage.SEARCH_RESULTS and state.last_png_bytes_raw:
            img = Image.open(BytesIO(state.last_png_bytes_raw))
            img_w, img_h = img.size
            candidates = []
            if action.candidates:
                for c in action.candidates:
                    bbox = c.bbox if hasattr(c, "bbox") else None
                    if isinstance(bbox, list) and len(bbox) == 4:
                        candidates.append({"bbox": bbox, "confidence": getattr(c, "confidence", None), "reason": getattr(c, "reason", None)})
            if action.type == "click_candidates":
                candidates.extend([])
            if action.bbox:
                candidates.append({"bbox": action.bbox, "confidence": None, "reason": "primary"})

            if candidates and action.type in ("click", "click_candidates"):
                chosen = None
                reject_logs = []
                for c in candidates:
                    bbox = c["bbox"]
                    in_col = bbox_in_valid_column(tuple(bbox), img_w, img_h)
                    ratio = blue_ratio(state.last_png_bytes_raw, tuple(bbox))
                    reject_logs.append(
                        {
                            "bbox": bbox,
                            "center": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                            "in_valid_column": in_col,
                            "blue_ratio": ratio,
                        }
                    )
                    if in_col:
                        chosen = bbox
                        break
                if not chosen:
                    state.retry_count += 1
                    if state.run_dir:
                        rej_dir = os.path.join(state.run_dir, "rejections")
                        os.makedirs(rej_dir, exist_ok=True)
                        rej_path = os.path.join(rej_dir, f"step_{state.step_count + 1:02d}_rejected_action.json")
                        with open(rej_path, "w", encoding="utf-8") as f:
                            import json
                            json.dump(
                                {"action": state.last_action, "stage": state.stage, "candidates": reject_logs},
                                f,
                                ensure_ascii=True,
                                indent=2,
                            )
                    log_event(logger, "reject", reason={"candidates": reject_logs}, step=state.step_count + 1)
                    if state.retry_count > state.max_retries_per_stage:
                        state.stage = Stage.DONE
                        state.next_node = END
                        return state.model_dump()
                    state.next_node = "validate"
                    return state.model_dump()
                remaining = [c for c in candidates if c["bbox"] != chosen]
                state.pending_candidates = remaining
                action = Action.model_validate({**action.model_dump(), "bbox": chosen, "type": "click"})
        _execute_action(browser, action, state)
        state.step_count += 1
        state.next_node = "validate"
        log_event(logger, "act", action=state.last_action, step=state.step_count)
        return state.model_dump()

    def validate_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState.model_validate(state_dict)
        if not state.last_png_bytes:
            state.next_node = "observe"
            return state.model_dump()
        result, img_hash = validator.assess(state, state.last_png_bytes)
        log_event(logger, "validate", result=result.__dict__, hash=img_hash)

        if state.current_url and "/login" in state.current_url:
            state.next_node = "observe"
            return state.model_dump()

        if result.new_stage:
            state.stage = result.new_stage
            state.retry_count = 0
            state.refine_level = 0
            state.pending_candidates = []

        if result.should_stop:
            state.stage = Stage.DONE
            state.next_node = END
            return state.model_dump()

        if result.recovery_action:
            state.retry_count += 1
            if state.retry_count > state.max_retries_per_stage:
                state.stage = Stage.DONE
                state.next_node = END
                return state.model_dump()
            state.last_action = result.recovery_action
            state.next_node = "act"
            return state.model_dump()

        if (
            state.stage == Stage.SEARCH_RESULTS
            and state.last_action
            and state.last_action.get("type") == "click"
            and len(state.last_urls) >= 2
            and state.last_urls[-1] == state.last_urls[-2]
        ):
            if state.pending_candidates:
                next_cand = state.pending_candidates.pop(0)
                state.last_action = {"type": "click", "reason": "next_candidate", "bbox": next_cand["bbox"]}
                state.next_node = "act"
                return state.model_dump()
            state.refine_level = min(state.refine_level + 1, 2)
            state.retry_count += 1
            state.next_node = "observe"
            return state.model_dump()

        if result.should_extract and state.stage == Stage.RELEASES:
            state.next_node = "extract"
            return state.model_dump()

        state.next_node = "observe"
        return state.model_dump()

    def extract_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState.model_validate(state_dict)
        if not state.last_png_bytes:
            state.next_node = "observe"
            return state.model_dump()
        release = extractor.extract(state.last_png_bytes, state.target_repo)
        state.extracted_release = release
        state.stage = Stage.EXTRACTED
        state.next_node = END
        log_event(logger, "extract", release=release.model_dump())
        return state.model_dump()

    graph.add_node("observe", observe_node)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)
    graph.add_node("validate", validate_node)
    graph.add_node("extract", extract_node)

    graph.set_entry_point("observe")
    graph.add_conditional_edges(
        "observe",
        lambda s: s.get("next_node", "decide"),
        {"decide": "decide"},
    )
    graph.add_conditional_edges(
        "decide",
        lambda s: s.get("next_node", "act"),
        {"act": "act"},
    )
    graph.add_conditional_edges(
        "act",
        lambda s: s.get("next_node", "validate"),
        {"validate": "validate"},
    )
    graph.add_conditional_edges(
        "validate",
        lambda s: s.get("next_node", "observe"),
        {"observe": "observe", "act": "act", "extract": "extract", END: END},
    )
    graph.add_conditional_edges(
        "extract",
        lambda s: s.get("next_node", END),
        {END: END},
    )

    return graph.compile()


def _subgoal_for_stage(state: AgentState) -> str:
    if state.stage == Stage.HOME:
        return f"Find the search bar and search for {state.target_repo}."
    if state.stage == Stage.SEARCH_RESULTS:
        return (
            "You are on GitHub search results. "
            f"Find the FIRST result card for \"{state.target_repo}\" and return a coarse bbox around the entire card "
            "(rounded rectangle containing avatar, title, description). Do NOT return a tight link bbox."
        )
    if state.stage == Stage.REPO:
        return (
            "Find and click the Releases section in the right sidebar. "
            "If Releases is not visible, scroll down until it appears, then click it."
        )
    if state.stage == Stage.RELEASES:
        return "Ensure the latest release card is visible and readable."
    return "Wait or noop."


def _execute_action(browser: BrowserSession, action: Action, state: AgentState) -> None:
    if action.type == "click" and action.bbox:
        browser.click_bbox(tuple(action.bbox))
        if action.text and state.stage == Stage.HOME:
            browser.press_key("Meta+A")
            browser.press_key("Backspace")
            browser.type_text(action.text)
            browser.press_key("Enter")
        if action.key:
            browser.press_key(action.key)
        browser.wait_for_idle()
        return
    if action.type == "type":
        if action.bbox:
            browser.type_into_bbox(action.bbox, action.text or "")
        else:
            browser.type_text(action.text or "")
        if action.key:
            browser.press_key(action.key)
        elif action.text:
            browser.press_key("Enter")
        return
    if action.type == "press" and action.key:
        browser.press_key(action.key)
        browser.wait_for_idle()
        return
    if action.type == "scroll" and action.scroll:
        browser.scroll(action.scroll.direction, action.scroll.amount)
        return
    if action.type == "back":
        browser.back()
        return
    if action.type == "wait":
        browser.wait_for_idle()
        return
    # noop
    return


def _save_bbox_overlay(png_bytes: bytes, bbox: tuple[int, int, int, int], step: int) -> None:
    os.makedirs("bbox_img", exist_ok=True)
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 102, 255), width=2)
    out_path = os.path.join("bbox_img", f"step_{step:02d}_bbox.png")
    img.save(out_path)


def _explore_click_points_multi(
    browser: BrowserSession, bbox: Tuple[int, int, int, int], target_repo: str, logger
) -> bool:
    x1, y1, x2, y2 = _clamp_bbox_to_viewport(browser, bbox)
    if x2 <= x1 + 20 or y2 <= y1 + 20:
        return False

    rounds = []
    rounds.append(("A", (x1, y1, x2, y2), _grid_points(x1, y1, x2, y2, xs=3, ys=2)))

    bx1, by1, bx2, by2 = _shrink_top_left((x1, y1, x2, y2), 0.70)
    rounds.append(("B", (bx1, by1, bx2, by2), _grid_points(bx1, by1, bx2, by2, xs=5, ys=2)))

    cx1, cy1, cx2, cy2 = _shrink_top_left((bx1, by1, bx2, by2), 0.70)
    rounds.append(("C", (cx1, cy1, cx2, cy2), _grid_points(cx1, cy1, cx2, cy2, xs=5, ys=2)))

    target_path = f"/{target_repo}".lower()
    for label, (rx1, ry1, rx2, ry2), points in rounds:
        log_event(
            logger,
            "explore_round",
            round=label,
            bbox=[rx1, ry1, rx2, ry2],
            points=points,
        )
        for (px, py) in points:
            before = browser.page.url
            browser.click_point(px, py)
            time.sleep(0.8)
            try:
                browser.page.wait_for_load_state("domcontentloaded", timeout=1500)
            except Exception:
                pass
            after = browser.page.url
            log_event(logger, "explore_click", point=[px, py], before=before, after=after)
            if after != before and target_path in after.lower():
                return True
    return False


def _grid_points(x1: int, y1: int, x2: int, y2: int, xs: int, ys: int) -> List[List[int]]:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    xs_fracs = [(i + 1) / (xs + 1) for i in range(xs)]
    ys_fracs = [(i + 1) / (ys + 1) for i in range(ys)]
    points: List[List[int]] = []
    for fy in ys_fracs:
        for fx in xs_fracs:
            px = min(max(x1 + int(fx * w), x1 + 4), x2 - 4)
            py = min(max(y1 + int(fy * h), y1 + 4), y2 - 4)
            points.append([px, py])
    return points[:10]


def _shrink_top_left(bbox: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return (x1, y1, x1 + int(scale * w), y1 + int(scale * h))


def _clamp_bbox_to_viewport(
    browser: BrowserSession, bbox: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    viewport = browser.page.viewport_size or {"width": 1280, "height": 720}
    w = viewport["width"]
    h = viewport["height"]
    x1 = max(220, min(x1, w))
    x2 = max(220, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2
