import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.llm_engine import LocalExtractiveLLM


def test_local_llm_fallback_returns_cannot_determine_when_no_context():
    llm = LocalExtractiveLLM()
    prompt = "Conversation History:\n\nContext:\n\nQuestion: What is X?"
    answer = llm.complete(prompt)
    assert "cannot determine" in answer.lower()


def test_local_llm_fallback_uses_context_when_present():
    llm = LocalExtractiveLLM()
    prompt = (
        "Conversation History:\n\n"
        "Context:\nThe sky is blue during a clear day. Water boils at 100C.\n"
        "Question: What color is the sky?"
    )
    answer = llm.complete(prompt)
    assert "sky is blue" in answer.lower()
