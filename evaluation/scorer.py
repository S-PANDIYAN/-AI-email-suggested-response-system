"""
Weighted scorer — combine the metrics into ONE 0–100 number per email.

THE FORMULA
-----------
    final = 100 · ( w_judge · judge_norm
                  + w_bert  · bert_f1
                  + w_cos   · cosine )

with weights from config (validated to sum to 1.0):
    judge = 0.50, bertscore = 0.30, cosine = 0.20.

WHY THESE WEIGHTS (the reasoning, not just the numbers)
-------------------------------------------------------
* LLM judge (0.50) — the only metric that reads for correctness, empathy, and
  actionability the way a human QA reviewer would. It's the best proxy for the
  quantity we actually care about, so it dominates. Not >0.5, because a single
  judge has known biases; the reference-based metrics keep it honest.
* BERTScore (0.30) — model-based, reference-grounded semantic overlap. It
  anchors the score to a real gold reply, catching a judge that is too
  generous. Second-highest because it's robust and paraphrase-aware.
* Cosine (0.20) — cheap, coarse floor. Guarantees an off-topic reply can't get
  a high score even if the judge misfires. Lowest weight: document-level and
  can't see correctness.

All three inputs are normalised to [0, 1] before weighting, so the weights
mean what they say. ROUGE-L is reported but weighted 0 (see lexical.py).

This is a linear (convex) combination: transparent, debuggable, and easy to
re-tune. Alternatives (learned regressor, geometric mean, min-gate) are noted
in the README; linear is the right call without labelled human scores to fit.
"""
from __future__ import annotations

from config.settings import Settings


def combine(judge_norm: float, bert_f1: float, cosine: float,
            weights: dict[str, float]) -> float:
    """Return the final weighted score in [0, 100]."""
    raw = (weights["llm_judge"] * judge_norm
           + weights["bertscore"] * bert_f1
           + weights["cosine"] * cosine)
    return round(100.0 * raw, 2)


def score_row(cosine: float, bert: dict, judge: dict, settings: Settings) -> dict:
    """Assemble the per-email score record from the three metric outputs."""
    final = combine(judge["judge_normalized"], bert["f1"], cosine, settings.weights)
    return {
        "cosine": round(cosine, 4),
        "bertscore_precision": round(bert["precision"], 4),
        "bertscore_recall": round(bert["recall"], 4),
        "bertscore_f1": round(bert["f1"], 4),
        "bertscore_backend": bert["backend"],
        "judge_avg_1to5": round(judge["judge_avg_1to5"], 3),
        "judge_normalized": round(judge["judge_normalized"], 4),
        "judge_backend": judge["backend"],
        "final_score": final,
    }
