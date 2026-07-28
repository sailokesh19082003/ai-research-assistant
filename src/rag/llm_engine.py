"""
Thin LLM abstraction so the rest of the RAG pipeline doesn't care which
backend is answering.

Priority: Gemini (if GEMINI_API_KEY is set) -> OpenAI (if OPENAI_API_KEY is
set) -> offline extractive fallback (always works, no key needed).
"""
from config.settings import settings


class BaseLLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

    def complete(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


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
    def complete(self, prompt: str) -> str:
        if "Context:" in prompt and "Question:" in prompt:
            context = prompt.split("Context:")[1].split("Question:")[0].strip()
            question = prompt.split("Question:")[1].strip()
            if not context:
                return "I cannot determine the answer from the provided documents."
            sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
            top_sentences = sentences[:3]
            return (
                "[Offline mode - no LLM API key configured] Based on the retrieved "
                f"context, here is the most relevant information for '{question}':\n\n"
                + ". ".join(top_sentences) + "."
            )
        return "[Offline mode - no LLM API key configured] " + prompt[-1500:]


def get_llm() -> BaseLLM:
    if settings.GEMINI_API_KEY:
        try:
            return GeminiLLM()
        except Exception:
            pass
    if settings.OPENAI_API_KEY:
        try:
            return OpenAILLM()
        except Exception:
            pass
    return LocalExtractiveLLM()
