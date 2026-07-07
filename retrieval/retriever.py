"""
Retriever — the "R" in RAG.

Ties the embedder and the vector store together into one object that, given a
raw query email, returns the most relevant PAST cases (email + the reply an
agent actually sent). Those cases become few-shot grounding for the LLM.

CONCEPT: Why grounding (RAG) beats a bare LLM prompt
----------------------------------------------------
A bare LLM invents plausible-sounding but unfounded specifics ("refunds take
5 days" — says who?). By injecting real past replies, we ground the model in
THIS company's actual policies, tone, and phrasing. The model imitates
verified examples instead of hallucinating.

CONCEPT: top-k and the leakage trap
-----------------------------------
top-k = how many neighbours we retrieve. Small k -> tight, on-topic context
but risks missing a better match; large k -> more coverage but dilutes the
prompt with weak matches and costs tokens. k=3 is a good default for support.

Leakage: at EVALUATION time the query email is itself in the index (it's a
dataset row). If we let it retrieve itself, the LLM would just copy the gold
reply and every metric would be a meaningless ~1.0. ``exclude_self`` drops the
query's own id, so the model must generalise from *other* cases — exactly the
real deployment condition (a brand-new email).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedCase:
    """One grounding example returned by the retriever."""
    id: int
    score: float                 # cosine similarity to the query (0..1-ish)
    customer_email: str
    support_reply: str
    category: str


class Retriever:
    def __init__(self, df: pd.DataFrame, embedder: Embedder, exclude_self: bool = True):
        """Build the index over ``df`` (must have id/customer_email/... columns)."""
        self.df = df.set_index("id", drop=False)
        self.embedder = embedder
        self.exclude_self = exclude_self

        self.store = VectorStore(embedder.dim)
        vectors = embedder.encode(df["customer_email"].tolist())
        self.store.add(df["id"].tolist(), vectors)
        log.info("Indexed %d cases for retrieval", len(df))

    def retrieve(self, query_email: str, k: int, self_id: int | None = None
                 ) -> list[RetrievedCase]:
        """Return up to ``k`` most similar past cases to ``query_email``.

        ``self_id`` (the query's own dataset id, if it is one) is excluded when
        ``exclude_self`` is set. We over-fetch by one then filter, so we still
        return k results after dropping the self match.
        """
        qvec = self.embedder.encode([query_email])
        fetch = k + 1 if (self.exclude_self and self_id is not None) else k
        scores, ids = self.store.search(qvec, fetch)

        cases: list[RetrievedCase] = []
        for score, cid in zip(scores[0], ids[0]):
            cid = int(cid)
            if self.exclude_self and self_id is not None and cid == self_id:
                continue
            row = self.df.loc[cid]
            cases.append(RetrievedCase(
                id=cid, score=float(score),
                customer_email=row["customer_email"],
                support_reply=row["support_reply"],
                category=row["category"],
            ))
            if len(cases) == k:
                break
        return cases
