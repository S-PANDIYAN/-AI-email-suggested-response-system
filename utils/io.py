"""
Small, dependency-light IO helpers used across the pipeline.

Kept in one place to avoid duplicating "make parent dir then write" logic
(DRY). Every function is pure-ish: it touches the filesystem but has no hidden
global state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (a directory) and parents if missing; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise ``obj`` to pretty JSON (utf-8). Creates parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(text: str, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(text)
