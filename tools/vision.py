from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, ValidationError


class ScrollSpec(BaseModel):
    direction: str
    amount: int


class ExpectSpec(BaseModel):
    url_contains: Optional[str] = None
    page_contains_text: Optional[str] = None


class CandidateBBox(BaseModel):
    bbox: list[int]
    confidence: Optional[float] = None
    text: Optional[str] = None
    reason: Optional[str] = None


class Action(BaseModel):
    type: str
    reason: str
    bbox: Optional[list[int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    scroll: Optional[ScrollSpec] = None
    expect: Optional[ExpectSpec] = None
    candidates: Optional[list[CandidateBBox]] = None


class ReleaseExtract(BaseModel):
    version: Optional[str] = None
    tag: Optional[str] = None
    author: Optional[str] = None


class VisionClient:
    def __init__(self, model_nav: str = "gpt-5-mini", model_extract: Optional[str] = None) -> None:
        self._client = OpenAI()
        self._model_nav = model_nav
        self._model_extract = model_extract or model_nav

    @staticmethod
    def _b64(png_bytes: bytes) -> str:
        return base64.b64encode(png_bytes).decode("ascii")

    def get_action(self, png_bytes: bytes, subgoal: str, stage: str) -> Action:
        system = (
            "You are a VLM navigation agent. Use ONLY the screenshot to decide the next UI action. "
            "Return STRICT JSON that matches the Action schema. No prose. "
            "Choose large, unambiguous targets. If a search box is present, click it and type. "
            "If you need to scroll, return type=scroll with direction and amount."
        )
        user = (
            f"Stage: {stage}. Subgoal: {subgoal}. "
            "Return JSON: {type, reason, bbox?, text?, key?, scroll?, expect?}. "
            "bbox uses pixel coordinates [x1,y1,x2,y2] in the screenshot." 
        )
        return self._call_action(self._model_nav, system, user, png_bytes)

    def get_release_extract(self, png_bytes: bytes, repo: str) -> ReleaseExtract:
        system = (
            "You are a VLM extraction agent. Use ONLY the screenshot to read the latest release info. "
            "Return STRICT JSON with keys: version, tag, author. No prose."
        )
        user = f"Repository: {repo}. Extract latest release info from the page."
        return self._call_extract(self._model_extract, system, user, png_bytes)

    def _call_action(self, model: str, system: str, user: str, png_bytes: bytes) -> Action:
        payload = self._call_json(model, system, user, png_bytes)
        payload = self._normalize_action_payload(payload)
        try:
            return Action.model_validate(payload)
        except ValidationError as e:
            raise RuntimeError(f"Invalid Action JSON: {e}")

    def _call_extract(self, model: str, system: str, user: str, png_bytes: bytes) -> ReleaseExtract:
        payload = self._call_json(model, system, user, png_bytes)
        try:
            return ReleaseExtract.model_validate(payload)
        except ValidationError as e:
            raise RuntimeError(f"Invalid Release JSON: {e}")

    @staticmethod
    def _normalize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        action_type = payload.get("type")
        if isinstance(action_type, str) and action_type.lower() in ("bbox", "box"):
            payload["type"] = "click"
        if isinstance(action_type, str) and action_type.lower() in ("click_type", "click-and-type", "click_and_type"):
            payload["type"] = "click"
        # Map top-level coords -> bbox if present.
        if "coords" in payload and "bbox" not in payload:
            payload["bbox"] = payload.get("coords")
        # If bbox is a list of bboxes, move them into candidates.
        bbox_val = payload.get("bbox")
        if isinstance(bbox_val, list) and bbox_val and isinstance(bbox_val[0], list):
            payload["candidates"] = [{"bbox": b} for b in bbox_val]
            payload["bbox"] = None
        # Normalize candidates
        cands = payload.get("candidates")
        if isinstance(cands, list):
            norm = []
            for c in cands:
                if isinstance(c, dict):
                    if "coords" in c and "bbox" not in c:
                        c["bbox"] = c.get("coords")
                    if isinstance(c.get("bbox"), dict):
                        c = {**c, **c["bbox"]}
                    if "bbox" in c and isinstance(c["bbox"], list):
                        norm.append(c)
                elif isinstance(c, list) and len(c) == 4:
                    norm.append({"bbox": c})
            payload["candidates"] = norm
        expect = payload.get("expect")
        if isinstance(expect, str):
            payload["expect"] = {"page_contains_text": expect}
        scroll = payload.get("scroll")
        if scroll is False:
            payload["scroll"] = None
            scroll = None
        if isinstance(scroll, dict) and "amount" in scroll:
            amount = scroll.get("amount")
            if isinstance(amount, float):
                if 0 < amount < 1:
                    scroll["amount"] = 400
                else:
                    scroll["amount"] = int(round(amount))
        key = payload.get("key")
        if isinstance(key, str):
            allowed = {
                "Enter",
                "Tab",
                "Escape",
                "ArrowDown",
                "ArrowUp",
                "ArrowLeft",
                "ArrowRight",
                "PageDown",
                "PageUp",
                "Home",
                "End",
                "Backspace",
                "Delete",
            }
            if key not in allowed:
                payload["key"] = None
        return payload

    def _call_json(self, model: str, system: str, user: str, png_bytes: bytes) -> Dict[str, Any]:
        img_b64 = self._b64(png_bytes)
        img_data_url = f"data:image/png;base64,{img_b64}"
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self._client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": system},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user},
                                {"type": "input_image", "image_url": img_data_url},
                            ],
                        },
                    ],
                )
                text = resp.output_text or ""
                if not text.strip():
                    # Try to recover from empty output_text by inspecting output items.
                    try:
                        parts = []
                        for item in resp.output or []:
                            for c in getattr(item, "content", []) or []:
                                if getattr(c, "type", "") in ("output_text", "summary_text"):
                                    parts.append(getattr(c, "text", ""))
                        text = "\n".join(p for p in parts if p).strip()
                    except Exception:
                        text = ""
                if not text.strip():
                    raise RuntimeError(f"Empty model output_text (model={model})")
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    cleaned = cleaned.replace("json", "", 1).strip()
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError as je:
                    raise RuntimeError(f"Non-JSON model output: {text[:400]}") from je
                return payload
            except Exception as e:  # pragma: no cover - best effort
                last_err = e
                system += " Return valid JSON only."
        raise RuntimeError(f"Vision call failed: {last_err}")
