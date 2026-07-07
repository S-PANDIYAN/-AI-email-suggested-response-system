"""
Evaluator — orchestrates every metric over a batch and aggregates results.

Pipeline per run:
    for each (email, generated_reply, reference_reply):
        cosine        = Metric1(generated, reference)
        bert          = Metric2(generated, reference)
        judge         = Metric3(email, generated, reference)
        rouge         = Metric4(generated, reference)     # diagnostic
        final_score   = weighted_combine(judge, bert, cosine)
    aggregate -> mean / median / min / max, per-category breakdown.

Batch-embedding note: cosine and the fallback BERTScore both call the
embedder. We compute cosine for the whole batch in one call (vectorised), and
run BERTScore/judge per item (they need per-pair alignment / an LLM call).

Outputs a single ``EvalReport`` dataclass consumed by the reporting layer.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings
from embeddings.embedder import Embedder
from evaluation.metrics.bertscore import bertscore
from evaluation.metrics.cosine import cosine_similarity
from evaluation.metrics.lexical import rouge_l
from evaluation.metrics.llm_judge import LLMJudge
from evaluation.scorer import score_row
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EvalReport:
    per_email: list[dict]                       # one record per evaluated email
    summary: dict[str, Any]                     # overall + per-category stats
    meta: dict[str, Any] = field(default_factory=dict)


def _describe(values: list[float]) -> dict[str, float]:
    """mean/median/min/max/stdev for a list of scores (empty-safe)."""
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "stdev": 0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stdev": round(statistics.pstdev(values), 2),
    }


class Evaluator:
    def __init__(self, embedder: Embedder, judge: LLMJudge, settings: Settings):
        self.embedder = embedder
        self.judge = judge
        self.settings = settings
        self.bert_backend = settings.evaluation.get("bertscore_backend", "auto")

    def evaluate(self, records: list[dict]) -> EvalReport:
        """``records``: dicts with id, category, customer_email, generated,
        reference. Returns a full EvalReport."""
        emails = [r["customer_email"] for r in records]
        generated = [r["generated"] for r in records]
        reference = [r["reference"] for r in records]

        log.info("Scoring %d replies ...", len(records))
        cos = cosine_similarity(generated, reference, self.embedder)          # Metric 1
        bert = bertscore(generated, reference, self.embedder, self.bert_backend)  # Metric 2
        rouge = rouge_l(generated, reference)                                 # Metric 4

        per_email: list[dict] = []
        for i, rec in enumerate(records):
            judge = self.judge.score_one(emails[i], generated[i], reference[i])  # Metric 3
            row = score_row(cos[i], bert[i], judge, self.settings)
            row.update({
                "id": rec["id"],
                "category": rec["category"],
                "rouge_l_f1": round(rouge[i]["f1"], 4),
                "judge_rationale": judge.get("rationale", ""),
                "judge_dimensions": {d: judge[d]
                                     for d in self.settings.evaluation["judge_rubric"]},
            })
            per_email.append(row)
            log.info("  id=%-3s cat=%-18s final=%.1f (judge=%.2f bert=%.2f cos=%.2f)",
                     rec["id"], rec["category"], row["final_score"],
                     row["judge_normalized"], row["bertscore_f1"], row["cosine"])

        summary = self._summarize(per_email)
        return EvalReport(per_email=per_email, summary=summary, meta={
            "n_evaluated": len(per_email),
            "weights": self.settings.weights,
            "embedder": self.embedder.name,
            "judge_backend": self.judge.backend,
            "bertscore_backend": bert[0]["backend"] if bert else "n/a",
        })

    def _summarize(self, per_email: list[dict]) -> dict[str, Any]:
        finals = [r["final_score"] for r in per_email]
        overall = _describe(finals)

        # Per-category breakdown — reveals WHERE the system is weak.
        by_cat: dict[str, list[float]] = {}
        for r in per_email:
            by_cat.setdefault(r["category"], []).append(r["final_score"])
        per_category = {cat: _describe(v) for cat, v in sorted(by_cat.items())}

        # Component averages — diagnose WHICH metric drives the score.
        def avg(key: str) -> float:
            vals = [r[key] for r in per_email]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "overall": overall,
            "component_means": {
                "cosine": avg("cosine"),
                "bertscore_f1": avg("bertscore_f1"),
                "judge_normalized": avg("judge_normalized"),
                "rouge_l_f1": avg("rouge_l_f1"),
            },
            "per_category": per_category,
        }
