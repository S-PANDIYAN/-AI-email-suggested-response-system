"""
Prompt construction — turn retrieved cases + the new email into an LLM prompt.

CONCEPT: Prompt engineering
---------------------------
An LLM's output is a function of its input. Prompt engineering is the practice
of structuring that input to reliably get the behaviour you want. For grounded
support replies, a strong prompt has four parts:

    1. SYSTEM message — the persona, rules, and hard constraints. Sets the
       policy the model must obey ("be concise, never invent policies").
    2. FEW-SHOT EXAMPLES — real (email -> reply) pairs from retrieval. The
       model learns the company's tone/format by imitation, not description.
    3. THE TASK — the new customer email to answer.
    4. OUTPUT INSTRUCTION — exactly what to return (just the reply body).

Why few-shot beats a longer instruction: showing 3 concrete gold replies
transfers tone, length, and structure far more reliably than any adjective
list ("be empathetic and professional"). Demonstration > description.

We wrap the single best reply in <<TOP_REPLY>> sentinels. Real LLMs ignore
them; the offline MockProvider uses them to recover a grounded answer. This
keeps ONE prompt format working across every backend.
"""
from __future__ import annotations

from retrieval.retriever import RetrievedCase

SYSTEM_PROMPT = (
    "You are a senior customer-support agent for an e-commerce company. "
    "Write the reply the company would send. Requirements:\n"
    "- Be warm, empathetic, and professional.\n"
    "- Be concise and specific; give the customer a clear next step.\n"
    "- Match the tone and policies shown in the examples.\n"
    "- NEVER invent policies, prices, or timelines not implied by the examples.\n"
    "- Output ONLY the reply body (no subject line, no meta commentary)."
)


def build_user_prompt(query_email: str, cases: list[RetrievedCase]) -> str:
    """Assemble few-shot examples + the new email into the user message."""
    blocks: list[str] = []
    blocks.append("Here are similar past tickets and the replies we sent:\n")

    for i, c in enumerate(cases, 1):
        top_open = "<<TOP_REPLY>>" if i == 1 else ""
        top_close = "<<END_TOP_REPLY>>" if i == 1 else ""
        blocks.append(
            f"--- Example {i} (category: {c.category}, similarity: {c.score:.2f}) ---\n"
            f"Customer: {c.customer_email}\n"
            f"Agent reply: {top_open}{c.support_reply}{top_close}\n"
        )

    blocks.append(
        "\nNow write the best reply to this NEW customer email:\n"
        f"Customer: {query_email}\n"
        "Agent reply:"
    )
    return "\n".join(blocks)
