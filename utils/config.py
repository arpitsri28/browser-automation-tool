from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    model_vlm: str
    model_router: str
    model_prompt: str


DEFAULT_MODEL_VLM = "gpt-5-mini"
DEFAULT_MODEL_ROUTER = "gpt-5-nano"
DEFAULT_MODEL_PROMPT = "gpt-5-mini"


def load_config(
    vlm_model_override: str | None = None,
    router_model_override: str | None = None,
    prompt_model_override: str | None = None,
) -> AppConfig:
    model_vlm = vlm_model_override or os.getenv("OPENAI_MODEL_VLM", DEFAULT_MODEL_VLM)
    model_router = router_model_override or os.getenv("OPENAI_MODEL_ROUTER", DEFAULT_MODEL_ROUTER)
    model_prompt = prompt_model_override or os.getenv("OPENAI_MODEL_PROMPT", DEFAULT_MODEL_PROMPT)
    api_key = os.getenv("OPENAI_API_KEY")
    return AppConfig(
        openai_api_key=api_key,
        model_vlm=model_vlm,
        model_router=model_router,
        model_prompt=model_prompt,
    )
