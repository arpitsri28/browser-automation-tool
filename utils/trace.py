from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class TraceWriter:
    run_dir: str

    @staticmethod
    def create(base_dir: str = "runs") -> "TraceWriter":
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        return TraceWriter(run_dir=run_dir)

    def save_image(self, step: int, png_bytes: bytes) -> str:
        path = os.path.join(self.run_dir, f"step_{step:02d}.png")
        with open(path, "wb") as f:
            f.write(png_bytes)
        return path

    def save_json(self, step: int, suffix: str, data: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, f"step_{step:02d}_{suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        return path
