"""
Metric 4 (diagnostic) — ROUGE-L, a lexical overlap metric.

CONCEPT: ROUGE-L
----------------
ROUGE-L is based on the Longest Common Subsequence (LCS) between candidate and
reference word sequences. LCS rewards in-order shared words without requiring
them to be contiguous, so it captures sentence-level structure overlap.

    R_lcs = LCS / len(reference)
    P_lcs = LCS / len(candidate)
    F_lcs = (1+β²)·P·R / (R + β²·P)      (β=1 here -> harmonic mean)

WHY it is DIAGNOSTIC ONLY (weight = 0)
--------------------------------------
Lexical metrics (ROUGE/BLEU/METEOR) reward word overlap. For open-ended
support replies with many valid phrasings, a perfect answer that paraphrases
the reference scores low — the classic failure this project is built to avoid.
We compute and REPORT ROUGE-L for transparency and comparison, but we do NOT
feed it into the final score. Including it as a reported-but-unweighted metric
demonstrates the point rather than just asserting it.

Strengths: cheap, deterministic, interpretable, standard in summarisation.
Weakness : blind to synonymy/paraphrase; penalises legitimate variation.

Complexity: O(|c| · |r|) for the LCS dynamic program.
"""
from __future__ import annotations

from evaluation.metrics.bertscore import _tokenize


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Classic DP for longest common subsequence length. O(|a|·|b|)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(generated: list[str], reference: list[str]) -> list[dict]:
    """Return per-pair ROUGE-L precision/recall/f1."""
    out = []
    for gen, ref in zip(generated, reference):
        c, r = _tokenize(gen), _tokenize(ref)
        lcs = _lcs_length(c, r)
        p = lcs / len(c) if c else 0.0
        rec = lcs / len(r) if r else 0.0
        f = 0.0 if (p + rec) == 0 else 2 * p * rec / (p + rec)
        out.append({"precision": p, "recall": rec, "f1": f})
    return out
