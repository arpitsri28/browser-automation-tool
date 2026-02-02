from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from agent.graph import build_graph
from agent.state import AgentState
from tools.browser import BrowserSession
from tools.extractor import ReleaseExtractor
from tools.validator import Validator
from tools.vision import VisionClient
from utils.config import load_config
from utils.logging import get_logger, log_event
from utils.trace import TraceWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM GitHub Releases Navigator")
    parser.add_argument("--repo", default="openclaw/openclaw", help="owner/repo")
    parser.add_argument("--url", default="https://github.com", help="start URL")
    parser.add_argument("--prompt", default=None, help="optional custom prompt")
    parser.add_argument("--out", default=None, help="write output JSON to file")
    parser.add_argument("--headed", action="store_true", help="run browser headed")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--vlm-model", default=None, help="override VLM model name")
    parser.add_argument("--router-model", default=None, help="override router model name")
    parser.add_argument("--prompt-model", default=None, help="override prompt model name")
    return parser.parse_args()


def main() -> int:
    _try_load_dotenv()
    args = parse_args()
    config = load_config(args.vlm_model, args.router_model, args.prompt_model)
    logger = get_logger()
    tracer = TraceWriter.create()

    browser = BrowserSession(headed=args.headed)
    vision = VisionClient(model_nav=config.model_vlm, model_extract=config.model_vlm)
    validator = Validator()
    extractor = ReleaseExtractor(vision)

    state = AgentState(
        target_repo=args.repo,
        start_url=args.url,
        prompt=args.prompt,
        vlm_model=config.model_vlm,
        router_model=config.model_router,
        prompt_model=config.model_prompt,
        max_steps=args.max_steps,
        max_retries_per_stage=args.max_retries,
        run_dir=tracer.run_dir,
    )

    browser.start(args.url)
    try:
        graph = build_graph(browser, vision, validator, extractor, tracer, logger)
        final_state: Dict[str, Any] = graph.invoke(state.model_dump())
    finally:
        browser.close()

    result = _format_output(final_state)
    log_event(logger, "done", result=result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)
    return 0


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _format_output(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState.model_validate(state_dict)
    release = state.extracted_release
    return {
        "repository": state.target_repo,
        "latest_release": {
            "version": release.version if release else None,
            "tag": release.tag if release else None,
            "author": release.author if release else None,
        },
    }


if __name__ == "__main__":
    sys.exit(main())
