"""
LLM provider abstraction.

CONCEPT: Why an abstraction layer over "just call the API"?
----------------------------------------------------------
Separation of concerns. The generator shouldn't know whether the text came
from Groq, OpenAI, Gemini, a local Ollama model, or a deterministic mock. It
should depend on ONE tiny contract:

    complete(system: str, user: str) -> str

Benefits:
    * Swap providers by changing one config line — no business-logic edits.
    * Test the pipeline offline with MockProvider (no network, no cost).
    * Same interface is reused by the LLM-as-judge (evaluation) — write the
      adapter once, use it twice.

This is the Strategy + Adapter pattern: each provider ADAPTS a vendor SDK to
our common STRATEGY interface.

Each real provider is imported lazily inside its constructor so the project
installs and runs with none of the vendor SDKs present.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from config.settings import Settings
from utils.logger import get_logger
from utils.ollama_client import OllamaClient

log = get_logger(__name__)


class LLMProvider(ABC):
    """Common contract for all text-generation backends."""

    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 400) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Local chat model via Ollama (preferred: free, private, no key)."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 400) -> str:
        return self.client.chat(self.model, system, user, temperature, max_tokens)


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        self.name = f"groq:{model}"

    def complete(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 400) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 400) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_obj = genai.GenerativeModel(model)
        self.name = f"gemini:{model}"

    def complete(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 400) -> str:
        resp = self.model_obj.generate_content(
            f"{system}\n\n{user}",
            generation_config={"temperature": temperature,
                               "max_output_tokens": max_tokens},
        )
        return resp.text.strip()


class MockProvider(LLMProvider):
    """Deterministic, offline provider.

    It does NOT call any model. Instead it adapts the single most-similar past
    reply (passed in the user prompt's context) into a fresh reply. This keeps
    the FULL pipeline runnable and testable with zero dependencies, and — since
    it grounds on *retrieved* text — it produces evaluation scores that are
    meaningful (high but not a perfect 1.0, because retrieval excludes the gold
    answer). It is intentionally simple: the mock exists to prove wiring, not
    to compete with a real LLM.
    """

    name = "mock"

    def complete(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 400) -> str:
        # The prompt builder tags the top example between sentinels so the mock
        # can recover a realistic reply without parsing free text.
        start, end = "<<TOP_REPLY>>", "<<END_TOP_REPLY>>"
        if start in user and end in user:
            reply = user.split(start, 1)[1].split(end, 1)[0].strip()
            return ("Thanks for reaching out — I understand the concern and I'm "
                    "happy to help. " + reply)
        return ("Thanks for reaching out. I've reviewed your message and our team "
                "will resolve this for you shortly. Please let me know if you need "
                "anything else in the meantime.")


def build_provider(settings: Settings) -> LLMProvider:
    """Factory implementing the "auto" preference + explicit override.

    Order for "auto": a cloud provider that has a REAL key (fastest to best
    quality for an assessment) -> local Ollama chat model -> Mock.
    """
    choice = settings.generation.get("provider", "auto")
    models = settings.generation["model"]

    def cloud(name: str) -> LLMProvider | None:
        key = settings.key_for(name)
        if not key:
            return None
        ctor = {"groq": GroqProvider, "openai": OpenAIProvider,
                "gemini": GeminiProvider}[name]
        try:
            return ctor(key, models[name])
        except Exception as exc:  # noqa: BLE001
            log.warning("%s provider init failed (%s)", name, exc)
            return None

    def ollama() -> LLMProvider | None:
        oc = OllamaClient(settings.ollama["base_url"], settings.ollama["timeout_s"])
        if oc.is_up():
            return OllamaProvider(oc, settings.ollama["chat_model"])
        return None

    if choice in {"groq", "openai", "gemini"}:
        p = cloud(choice)
        if p:
            return p
        log.warning("Requested %s but no usable key; falling back", choice)
    elif choice == "ollama":
        p = ollama()
        if p:
            return p
        log.warning("Requested ollama but server unreachable; using mock")
        return MockProvider()
    elif choice == "mock":
        return MockProvider()

    # auto
    for name in ("groq", "openai", "gemini"):
        p = cloud(name)
        if p:
            log.info("Generation provider: %s", p.name)
            return p
    p = ollama()
    if p:
        log.info("Generation provider: %s", p.name)
        return p
    log.info("Generation provider: mock (no keys, no local server)")
    return MockProvider()
