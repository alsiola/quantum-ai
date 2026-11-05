"""Reproducibility records: every saved result carries code revision, environment and config."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run_metadata(config: dict[str, Any] | None = None, seed: int | None = None) -> dict[str, Any]:
    """Code revision, dirty flag, environment, timestamp, config and seed."""
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "seed": seed,
        "config": config,
    }


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    return path


def new_run_dir(experiment: str, root: str | Path | None = None) -> Path:
    """``results/<experiment>/<UTC timestamp>/`` (created)."""
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) if root is not None else REPO_ROOT / "results"
    d = base / experiment / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d
