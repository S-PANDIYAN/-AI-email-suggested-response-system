"""
Vector store — index vectors, search by similarity.

CONCEPT: What is a vector database / index?
-------------------------------------------
Once every document is an embedding, "find similar text" becomes "find nearest
vectors". A vector index stores those vectors and answers nearest-neighbour
queries fast. FAISS (Facebook AI Similarity Search) is the de-facto library.

CONCEPT: Why IndexFlatIP + normalised vectors == cosine search
--------------------------------------------------------------
    cosine(a, b) = (a · b) / (|a| |b|)
If |a| = |b| = 1 (we L2-normalised in the embedder), then cosine(a, b) = a · b,
a plain inner product. ``IndexFlatIP`` is an exact inner-product index, so it
returns exact cosine-nearest neighbours. "Flat" = brute force, exact (no
approximation). For 100–100k vectors that's instant and perfectly accurate;
approximate indexes (IVF/HNSW) only matter at millions of vectors.

Complexity (Flat, N vectors, dim d, query batch q, top-k):
    search: O(q · N · d)  exact dot products.  Memory: O(N · d).
For N=100, d=768 this is ~77k multiply-adds per query — negligible.

Fallback: if faiss isn't importable we do the identical math with a single
numpy matmul, so the retriever's behaviour is unchanged.
"""
from __future__ import annotations

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _HAS_FAISS = False


class VectorStore:
    """Holds vectors + ids, answers top-k cosine search.

    Parameters
    ----------
    dim : embedding dimensionality (must match the embedder).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[int] = []
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(dim)     # exact inner-product
            self._backend = "faiss"
        else:
            self._matrix: np.ndarray | None = None   # (N, dim)
            self._backend = "numpy"
        log.info("VectorStore backend: %s (dim=%d)", self._backend, dim)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """Add a batch of (already unit-normalised) vectors with their ids."""
        assert vectors.shape[1] == self.dim, "vector dim mismatch"
        self._ids.extend(ids)
        if _HAS_FAISS:
            self._index.add(vectors.astype("float32"))
        else:
            self._matrix = (vectors if self._matrix is None
                            else np.vstack([self._matrix, vectors])).astype("float32")

    def search(self, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids) for the top-k neighbours of each query row.

        scores/ids have shape (n_queries, k). ids are the ORIGINAL ids passed
        to ``add`` (not internal offsets).
        """
        k = min(k, len(self._ids))
        if _HAS_FAISS:
            scores, offsets = self._index.search(query_vecs.astype("float32"), k)
        else:
            sims = query_vecs.astype("float32") @ self._matrix.T   # cosine
            offsets = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, offsets, axis=1)
        id_arr = np.array(self._ids)
        return scores, id_arr[offsets]
