"""
Metric 1 — Embedding Cosine Similarity.

CONCEPT
-------
Embed the generated reply and the reference reply into the same vector space,
then measure the angle between them:

    cosine(a, b) = (a · b) / (|a| |b|)   ∈ [-1, 1]   (≈ [0, 1] for text)

1.0 = same direction (same meaning), 0 = unrelated. Because our embedder
already L2-normalises, this is just a dot product.

WHY include it
--------------
It's the cheapest semantic signal and a good sanity FLOOR: if a reply is
totally off-topic, cosine catches it immediately, no LLM call needed. It is
document-level (one vector for the whole reply), so it's coarse — it can't
tell you *which part* is wrong. That's why it gets the smallest weight (0.20)
and is backed up by BERTScore and the judge.

LIMITATION
----------
Two fluent replies about the same topic score high even if one contains a
factual error, because topic ≈ direction. Cosine measures "about the same
thing", not "correct". Hence it is a floor, never the verdict.

Complexity: O(d) per pair after embedding (a single dot product).
"""
from __future__ import annotations

import numpy as np

from embeddings.embedder import Embedder


def cosine_similarity(generated: list[str], reference: list[str],
                      embedder: Embedder) -> list[float]:
    """Return per-pair cosine similarity in [0, 1] (clamped)."""
    gen = embedder.encode(generated)      # (n, d), already unit-normalised
    ref = embedder.encode(reference)      # (n, d)
    sims = np.sum(gen * ref, axis=1)      # row-wise dot == cosine
    return [float(max(0.0, min(1.0, s))) for s in sims]
