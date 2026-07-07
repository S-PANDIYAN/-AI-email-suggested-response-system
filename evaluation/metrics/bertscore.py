"""
Metric 2 — BERTScore (with an offline fallback).

CONCEPT: What BERTScore measures
--------------------------------
Instead of one vector per document (like cosine), BERTScore embeds EVERY TOKEN
with a contextual model, then greedily matches each generated token to its
most-similar reference token (and vice-versa):

    Precision = avg over generated tokens of max similarity to any ref token
                -> "how much of what I said is supported by the reference?"
    Recall    = avg over reference tokens of max similarity to any gen token
                -> "how much of the reference did I cover?"
    F1        = harmonic mean of the two -> the headline number.

WHY it's stronger than cosine or BLEU
-------------------------------------
* vs. cosine: token-level, so it localises overlap and rewards covering the
  key points rather than just matching overall topic.
* vs. BLEU/ROUGE: uses embeddings, so paraphrases ("refund" ≈ "money back")
  count as matches. n-gram metrics would miss them.

FALLBACK (this environment)
---------------------------
The real ``bert-score`` package needs transformers, which is broken here. We
provide a transparent fallback that keeps the SAME P/R/F1 shape using our
sentence-embedder at the WORD level: embed each unique word, greedy-match,
average. It is a genuine token-alignment score, just with our embedder instead
of RoBERTa. Clearly labelled in the output so results are never overstated.

Complexity (fallback): O((|g| + |r|) · d) to embed words + O(|g|·|r|) to match.
Reply lengths are short, so this is cheap.
"""
from __future__ import annotations

import re

import numpy as np

from embeddings.embedder import Embedder
from utils.logger import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _real_bertscore(generated: list[str], reference: list[str]) -> list[dict] | None:
    """Try the real bert-score library; return None if unavailable."""
    try:
        from bert_score import score as bs_score  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    P, R, F = bs_score(generated, reference, lang="en", rescale_with_baseline=False)
    return [{"precision": float(p), "recall": float(r), "f1": float(f),
             "backend": "bert-score"} for p, r, f in zip(P, R, F)]


def _fallback_pair(gen: str, ref: str, embedder: Embedder) -> dict:
    """Word-level greedy-matched P/R/F1 using our embedder."""
    g_words = _tokenize(gen) or [" "]
    r_words = _tokenize(ref) or [" "]
    # Deduplicate for embedding efficiency, keep a lookup back to vectors.
    vocab = list({*g_words, *r_words})
    vecs = embedder.encode(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    G = vecs[[idx[w] for w in g_words]]        # (|g|, d)
    Rm = vecs[[idx[w] for w in r_words]]       # (|r|, d)
    sim = G @ Rm.T                              # (|g|, |r|) cosine (unit vecs)
    precision = float(sim.max(axis=1).mean())   # best ref match per gen word
    recall = float(sim.max(axis=0).mean())      # best gen match per ref word
    denom = precision + recall
    f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
    return {"precision": max(0.0, precision), "recall": max(0.0, recall),
            "f1": max(0.0, f1), "backend": "fallback-wordsim"}


def bertscore(generated: list[str], reference: list[str], embedder: Embedder,
              backend: str = "auto") -> list[dict]:
    """Return a per-pair dict with precision/recall/f1/backend.

    ``backend``: "auto" tries real bert-score then falls back; "fallback"
    forces the offline word-similarity variant.
    """
    if backend != "fallback":
        real = _real_bertscore(generated, reference)
        if real is not None:
            log.info("BERTScore backend: bert-score (real)")
            return real
        log.info("BERTScore backend: fallback-wordsim (bert-score unavailable)")
    return [_fallback_pair(g, r, embedder) for g, r in zip(generated, reference)]
