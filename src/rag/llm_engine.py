"""
Thin LLM abstraction so the rest of the RAG pipeline doesn't care which
backend is answering.

- If OPENAI_API_KEY is set, uses OpenAI's chat completion API (gpt-4o by
  default, per the spec).
- Otherwise, falls back to a deterministic local "extractive" responder so
  the whole application remains runnable and testable without any paid API
  key. Swap in Ollama here too if you prefer a fully local LLM.
"""
from config.settings import settings


class BaseLLM:
    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class LocalExtractiveLLM(BaseLLM):
    """No-API-key fallback. Not a real generative model — it deterministically
    assembles an answer from the retrieved context so the RAG pipeline can be
    demoed/tested end-to-end offline. Replace with Ollama/OpenAI for real
    natural-language generation."""

    def complete(self, prompt: str) -> str:
        # The qa_chain / summarizer / comparator modules pass already-built
        # prompts; we simply extract the "Context:" section and return the
        # most relevant sentences as a best-effort offline answer.
        if "Context:" in prompt and "Question:" in prompt:
            context = prompt.split("Context:")[1].split("Question:")[0].strip()
            question = prompt.split("Question:")[1].strip()
            if not context or context == "":
                return "I cannot determine the answer from the provided documents."
            sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
            top_sentences = sentences[:3]
            return (
                "[Offline mode - no LLM API key configured] Based on the retrieved "
                f"context, here is the most relevant information for '{question}':\n\n"
                + ". ".join(top_sentences) + "."
            )
        # Generic fallback for summarization/comparison prompts.
        return "[Offline mode - no LLM API key configured] " + prompt[-1500:]


def get_llm() -> BaseLLM:
    if settings.OPENAI_API_KEY:
        try:
            return OpenAILLM()
        except Exception:
            pass
    return LocalExtractiveLLM()
