"""
Configuration loader.

Purpose
-------
Turn ``config/config.yaml`` + environment variables into a single, validated,
strongly-typed ``Settings`` object that the rest of the codebase imports.

Why a dedicated module (vs. reading yaml everywhere)?
    * Single source of truth — one place parses/validates config.
    * Fail fast — a bad weight sum or missing file raises at startup, not
      three minutes into a run.
    * Testability — code depends on a plain object, not on file I/O.

Inputs   : config/config.yaml, .env (via python-dotenv), OS environment.
Outputs  : a ``Settings`` dataclass instance (see ``load_settings``).
Depends  : pyyaml, python-dotenv (both core deps).

Complexity: O(1) — parses a tiny file once at process start.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = parent of the "config" directory. Resolved once, absolutely,
# so the program works regardless of the caller's current directory.
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Immutable, validated view of all configuration.

    ``frozen=True`` prevents accidental mutation of shared config at runtime —
    config should be read-only once loaded.
    """

    raw: dict[str, Any]                    # full parsed yaml (escape hatch)
    seed: int
    paths: dict[str, Path]
    dataset: dict[str, Any]
    ollama: dict[str, Any]
    embedding: dict[str, Any]
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    evaluation: dict[str, Any]
    logging: dict[str, Any]
    api_keys: dict[str, str] = field(default_factory=dict)

    # ----- convenience accessors (keep call sites readable) -----
    @property
    def weights(self) -> dict[str, float]:
        return self.evaluation["weights"]

    def key_for(self, provider: str) -> str | None:
        """Return the API key for a provider, or None if unset/placeholder."""
        val = self.api_keys.get(provider)
        if not val or val.strip() in {"", "your-actual-key-here", "your-key-here"}:
            return None
        return val


def _abs(p: str) -> Path:
    """Resolve a config path relative to the project root."""
    return (ROOT / p).resolve()


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load, validate, and return the global Settings object.

    Raises
    ------
    FileNotFoundError : config file missing.
    ValueError        : evaluation weights do not sum to 1.0.
    """
    # override=True so a real key in .env wins over a stale/placeholder value
    # already exported in the shell environment (e.g. GROQ_API_KEY=your-key-here).
    # Without this, dotenv keeps the pre-existing env var and the real key is ignored.
    load_dotenv(ROOT / ".env", override=True)  # no-op if .env absent; never raises

    cfg_file = Path(config_path) if config_path else ROOT / "config" / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # --- validation: fail fast on a misconfigured weighted score ---
    weights = raw["evaluation"]["weights"]
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"evaluation.weights must sum to 1.0, got {total:.4f} ({weights})"
        )

    paths = {k: _abs(v) for k, v in raw["paths"].items()}

    return Settings(
        raw=raw,
        seed=int(raw.get("seed", 42)),
        paths=paths,
        dataset=raw["dataset"],
        ollama=raw.get("ollama", {}),
        embedding=raw["embedding"],
        retrieval=raw["retrieval"],
        generation=raw["generation"],
        evaluation=raw["evaluation"],
        logging=raw["logging"],
        api_keys={
            "groq": os.getenv("GROQ_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
        },
    )
