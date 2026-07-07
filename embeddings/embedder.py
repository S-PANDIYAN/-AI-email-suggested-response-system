"""
Embedding layer — turn text into vectors.

CONCEPT: What is an embedding?
------------------------------
An embedding is a fixed-length vector of floats that represents the *meaning*
of a piece of text. Texts with similar meaning land close together in this
high-dimensional space. Analogy: a library where books aren't shelved
alphabetically but by topic — "refund" sits near "money back", far from
"bluetooth pairing", even though they share no words.

Why embeddings (vs. keyword search)?
    Keyword search matches surface tokens. "I want my money back" and
    "please refund my purchase" share almost no words but mean the same
    thing. Embeddings capture that; keywords don't.

CONCEPT: Why we L2-normalise every vector
------------------------------------------
Cosine similarity = dot product of unit vectors. If we normalise each vector
to length 1 up front, then a plain inner product IS the cosine similarity.
This lets us use FAISS's fast inner-product index and get cosine for free.

Design
------
``Embedder`` is an abstract Strategy. Three concrete backends implement it:
    * OllamaEmbedder            — real local model (embeddinggemma, 768-d)
    * SentenceTransformerEmbedder — real transformer (all-MiniLM, 384-d)
    * TfidfEmbedder             — deterministic lexical fallback (no models)
The rest of the codebase only ever sees ``Embedder.encode(texts) -> ndarray``.

Complexity (encode of n texts, dim d):
    Ollama/ST : O(n · model_cost); network/compute bound.
    TF-IDF    : O(n · avg_tokens) to vectorise; fit is O(corpus_tokens).
Space: O(n · d) floats for the returned matrix.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from config.settings import Settings
from utils.logger import get_logger
from utils.ollama_client import OllamaClient

log = get_logger(__name__)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so inner product == cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0            # guard against divide-by-zero on empties
    return (mat / norms).astype("float32")


class Embedder(ABC):
    """Strategy interface. ``dim`` is the vector length; ``name`` for logging."""

    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 matrix of unit vectors."""
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    """Real embeddings from a local Ollama model (preferred backend)."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model
        self.name = f"ollama:{model}"
        # Probe once to learn the dimensionality (embeddinggemma -> 768).
        self.dim = len(client.embed(model, "dimension probe"))

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = [self.client.embed(self.model, t if t.strip() else " ") for t in texts]
        return _l2_normalize(np.array(vecs, dtype="float32"))


class SentenceTransformerEmbedder(Embedder):
    """Real transformer embeddings via sentence-transformers (if installed)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import
        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True,
                                 show_progress_bar=False)
        return _l2_normalize(vecs.astype("float32"))


class TfidfEmbedder(Embedder):
    """Deterministic lexical fallback — needs NO model server or download.

    TF-IDF weights words by how distinctive they are, giving a sparse vector
    per document. We fit the vocabulary ONCE on the knowledge-base corpus,
    then transform queries into the same space. It captures lexical overlap
    (not deep semantics), which is a reasonable, fully-offline floor.

    Why it's a legitimate fallback and not a toy: TF-IDF + cosine was the
    backbone of production search for two decades. It just plateaus below
    transformer embeddings on paraphrase-heavy inputs.
    """

    def __init__(self, corpus: list[str], max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english",
        )
        self.vectorizer.fit(corpus)                    # learn vocab + idf
        self.name = "tfidf"
        self.dim = len(self.vectorizer.get_feature_names_out())

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        return _l2_normalize(mat)


def build_embedder(settings: Settings, corpus: list[str]) -> Embedder:
    """Factory implementing the "auto" preference order + explicit override.

    ``corpus`` is required only by the TF-IDF backend (to fit its vocab); the
    model-based backends ignore it. Passing it always keeps the factory's
    signature uniform.
    """
    backend = settings.embedding.get("backend", "auto")

    def make_ollama() -> Embedder:
        oc = OllamaClient(settings.ollama["base_url"], settings.ollama["timeout_s"])
        return OllamaEmbedder(oc, settings.ollama["embed_model"])

    def make_st() -> Embedder:
        return SentenceTransformerEmbedder(settings.embedding["st_model"])

    def make_tfidf() -> Embedder:
        return TfidfEmbedder(corpus, settings.embedding.get("tfidf_max_features", 4096))

    if backend == "ollama":
        return make_ollama()
    if backend == "sentence_transformer":
        return make_st()
    if backend == "tfidf":
        return make_tfidf()

    # ---- auto: try best -> worst, logging each fallback ----
    oc = OllamaClient(settings.ollama["base_url"], settings.ollama["timeout_s"])
    if oc.is_up():
        try:
            emb = OllamaEmbedder(oc, settings.ollama["embed_model"])
            log.info("Embedder backend: %s (dim=%d)", emb.name, emb.dim)
            return emb
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash
            log.warning("Ollama embedder failed (%s); trying next backend", exc)
    try:
        emb = make_st()
        log.info("Embedder backend: %s (dim=%d)", emb.name, emb.dim)
        return emb
    except Exception as exc:  # noqa: BLE001
        log.warning("sentence-transformers unavailable (%s); using TF-IDF", exc)

    emb = make_tfidf()
    log.info("Embedder backend: %s (dim=%d)", emb.name, emb.dim)
    return emb
