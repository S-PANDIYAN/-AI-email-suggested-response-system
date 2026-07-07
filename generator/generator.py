"""
Generator — orchestrates one email -> grounded reply.

Responsibilities (single, narrow):
    retrieve k cases -> build prompt -> call provider -> return reply + trace.

It deliberately owns NO retrieval or provider logic itself; it composes the
Retriever and LLMProvider it's given (dependency injection). That keeps it
trivially testable and swappable.

Inputs : a customer email (+ optional dataset id for self-exclusion).
Outputs: a ``Generation`` record capturing the reply AND the context used,
         so evaluation and debugging can see exactly what grounded the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import Settings
from generator.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from generator.providers import LLMProvider
from retrieval.retriever import RetrievedCase, Retriever
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Generation:
    query_id: int | None
    query_email: str
    reply: str
    provider: str
    retrieved: list[RetrievedCase] = field(default_factory=list)


class ReplyGenerator:
    def __init__(self, retriever: Retriever, provider: LLMProvider, settings: Settings):
        self.retriever = retriever
        self.provider = provider
        self.k = settings.retrieval["top_k"]
        self.temperature = settings.generation["temperature"]
        self.max_tokens = settings.generation["max_tokens"]

    def generate(self, query_email: str, query_id: int | None = None) -> Generation:
        cases = self.retriever.retrieve(query_email, self.k, self_id=query_id)
        user_prompt = build_user_prompt(query_email, cases)
        try:
            reply = self.provider.complete(
                SYSTEM_PROMPT, user_prompt,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - never let one email kill the run
            log.error("Generation failed for id=%s (%s); emitting fallback",
                      query_id, exc)
            reply = ("Thank you for reaching out. We've received your message and "
                     "a support specialist will follow up shortly.")
        return Generation(
            query_id=query_id, query_email=query_email, reply=reply,
            provider=self.provider.name, retrieved=cases,
        )
