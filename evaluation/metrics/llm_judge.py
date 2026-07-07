"""
Metric 3 — LLM-as-a-Judge  (the highest-weighted signal).

CONCEPT: Why an LLM judge
-------------------------
Cosine and BERTScore measure similarity to ONE reference reply. But a great
support answer can differ from the reference and still be great. A human
reviewer wouldn't diff tokens — they'd read it and rate correctness, tone,
completeness, etc. An LLM judge approximates that human reviewer: we give it a
rubric and ask it to score each dimension 1–5 with a short justification, and
return strict JSON.

Rubric (each 1–5):
    correctness    — is the information right and consistent with context?
    professionalism— appropriate business register?
    empathy        — does it acknowledge the customer's feelings?
    completeness   — does it address everything asked + a next step?
    grammar        — mechanics and fluency.
    tone           — warm, not robotic or curt.
    actionability  — is there a concrete action/resolution?

WHY it reflects human judgement
-------------------------------
It reads the reply holistically and reference-free-ish (we give the reference
as ONE valid example, not the only truth). It rewards a good NEW answer instead
of punishing it for differing from the gold text — the exact failure mode of
n-gram metrics.

RISKS (documented, mitigated)
-----------------------------
* Self-preference / verbosity bias: judges can favour longer or same-model
  text. Mitigation: explicit rubric, low temperature, small integer scale,
  averaging over 7 dimensions and 20 samples reduces variance.
* Non-determinism: temperature=0 + JSON-only instruction; we parse defensively.

FALLBACK (no LLM available)
---------------------------
A deterministic heuristic judge scores the same 7 dimensions from measurable
signals (semantic coverage of the reference, presence of greeting/closing,
length sanity, question-mark handling, hedging). It is weaker than a real
judge but keeps the pipeline whole and is clearly labelled ("heuristic").
"""
from __future__ import annotations

import json
import re

from config.settings import Settings
from embeddings.embedder import Embedder
from evaluation.metrics.bertscore import _tokenize
from generator.providers import LLMProvider, MockProvider
from utils.logger import get_logger

log = get_logger(__name__)

JUDGE_SYSTEM = (
    "You are a strict but fair QA reviewer for customer-support replies. "
    "You score a candidate reply against a rubric. Output ONLY valid JSON."
)


def _judge_user_prompt(email: str, candidate: str, reference: str,
                       rubric: list[str]) -> str:
    keys = ", ".join(f'"{d}"' for d in rubric)
    return (
        "Score the CANDIDATE reply to the customer email below.\n"
        "The REFERENCE is one example of a good reply (not the only correct "
        "answer) — reward a candidate that is correct and helpful even if it "
        "differs from the reference.\n\n"
        f"CUSTOMER EMAIL:\n{email}\n\n"
        f"REFERENCE REPLY:\n{reference}\n\n"
        f"CANDIDATE REPLY:\n{candidate}\n\n"
        "Rate each dimension from 1 (poor) to 5 (excellent). Return JSON with "
        f"exactly these integer keys: {{{keys}}} plus a short string key "
        '"rationale". No text outside the JSON.'
    )


def _parse_scores(text: str, rubric: list[str]) -> dict:
    """Extract the JSON object from an LLM response, defensively."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = json.loads(match.group(0)) if match else {}
    scores = {}
    for dim in rubric:
        try:
            scores[dim] = max(1, min(5, int(round(float(raw.get(dim, 3))))))
        except (TypeError, ValueError):
            scores[dim] = 3        # neutral default on malformed field
    scores["rationale"] = str(raw.get("rationale", ""))[:400]
    return scores


# ─────────────────────── heuristic fallback judge ──────────────────────────
def _heuristic_scores(email: str, candidate: str, reference: str,
                      rubric: list[str], embedder: Embedder) -> dict:
    """Deterministic 1–5 scores from measurable text signals."""
    cand = candidate.strip()
    cand_l = cand.lower()
    ref_tokens = set(_tokenize(reference))
    cand_tokens = set(_tokenize(cand))
    coverage = (len(ref_tokens & cand_tokens) / len(ref_tokens)) if ref_tokens else 0.0

    # Semantic closeness (document cosine) via the embedder.
    import numpy as np
    cv = embedder.encode([cand, reference])
    sem = float(max(0.0, min(1.0, np.dot(cv[0], cv[1]))))

    def band(x: float) -> int:                # map [0,1] -> 1..5
        return int(max(1, min(5, round(1 + 4 * x))))

    has_greeting = any(cand_l.startswith(g) for g in ("hi", "hello", "thanks", "thank"))
    has_closing = any(s in cand_l for s in ("let me know", "—", "regards", "happy to help"))
    has_action = any(v in cand_l for v in
                     ("i've", "i have", "i will", "i'll", "processed", "shipped",
                      "sent", "refund", "replacement", "cancelled", "reset", "arrange"))
    words = len(_tokenize(cand))
    length_ok = 20 <= words <= 160
    asks = email.count("?")
    addresses_q = (asks == 0) or has_action

    scores = {
        "correctness": band(0.5 * sem + 0.5 * coverage),
        "professionalism": 5 if (has_greeting and has_closing) else (4 if has_greeting else 3),
        "empathy": 5 if any(p in cand_l for p in
                            ("sorry", "understand", "apolog", "concern")) else 3,
        "completeness": band(0.4 * coverage + 0.3 * sem + (0.3 if addresses_q else 0.0)),
        "grammar": 4 if cand and cand[0].isupper() and cand.endswith((".", "!", "—")) else 3,
        "tone": band(sem) if has_greeting else max(2, band(sem) - 1),
        "actionability": 5 if has_action else 2,
    }
    scores = {d: scores.get(d, 3) for d in rubric}
    scores["rationale"] = (f"heuristic: sem={sem:.2f}, coverage={coverage:.2f}, "
                           f"len={words}, greeting={has_greeting}, action={has_action}")
    return scores


class LLMJudge:
    """Scores replies via a real LLM, or a heuristic fallback."""

    def __init__(self, provider: LLMProvider, settings: Settings, embedder: Embedder):
        self.rubric: list[str] = settings.evaluation["judge_rubric"]
        self.embedder = embedder
        backend = settings.evaluation.get("judge_backend", "auto")
        # Use the real judge only if we have a genuine LLM (not the mock).
        use_llm = backend != "heuristic" and not isinstance(provider, MockProvider)
        self.provider = provider if use_llm else None
        self.backend = provider.name if use_llm else "heuristic"
        log.info("LLM judge backend: %s", self.backend)

    def score_one(self, email: str, candidate: str, reference: str) -> dict:
        if self.provider is not None:
            try:
                raw = self.provider.complete(
                    JUDGE_SYSTEM,
                    _judge_user_prompt(email, candidate, reference, self.rubric),
                    temperature=0.0, max_tokens=300,
                )
                scores = _parse_scores(raw, self.rubric)
            except Exception as exc:  # noqa: BLE001 - fall back per-item
                log.warning("LLM judge failed (%s); using heuristic for this item", exc)
                scores = _heuristic_scores(email, candidate, reference,
                                           self.rubric, self.embedder)
        else:
            scores = _heuristic_scores(email, candidate, reference,
                                       self.rubric, self.embedder)

        dims = [scores[d] for d in self.rubric]
        scores["judge_avg_1to5"] = sum(dims) / len(dims)
        scores["judge_normalized"] = (scores["judge_avg_1to5"] - 1) / 4  # ->[0,1]
        scores["backend"] = self.backend
        return scores
