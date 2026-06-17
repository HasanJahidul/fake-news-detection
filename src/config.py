"""Central config loader. Single source of truth = config.yaml at repo root."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Sets vars that aren't already set."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


@functools.lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def path(key: str) -> Path:
    """Resolve a path from config['paths'] relative to the repo root."""
    cfg = load_config()
    return (REPO_ROOT / cfg["paths"][key]).resolve()


LABELS = load_config()["labels"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
