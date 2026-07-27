"""
Generates multi-tier summaries (Executive Summary, Technical Summary,
Bullet-Point Breakdown, Key Takeaways) for a single document, using all
indexed chunks belonging to that doc_id.
"""
from typing import Dict, Any

from src.vector_store.manager import get_vector_store
from src.rag.llm_engine import get_llm

SUMMARY_PROMPT_TEMPLATE = """You are an AI Research Assistant summarizing a technical document.
Use ONLY the content provided below. Do not invent facts not present in the text.

Document Content:
{context}

Produce a structured summary with these exact sections:
1. Executive Summary (2-3 sentences, non-technical)
2. Technical Summary (detailed, technical audience)
3. Bullet Point Breakdown (5-8 concise bullets of key points)
4. Key Takeaways (top 3 most important conclusions)
"""


class DocumentSummarizer:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()

    def _get_all_chunks_text(self, doc_id: str, max_chunks: int = 40) -> str:
        # Pull a representative sample of the document's chunks straight from
        # the vector collection (bypassing similarity search since we want
        # the whole document, not a query-relevant subset).
        data = self.vector_store.collection.get(
            where={"doc_id": doc_id}, include=["documents"]
        )
        chunks = data.get("documents", []) or []
        return "\n\n".join(chunks[:max_chunks])

    def summarize_document(self, doc_id: str, file_name: str) -> Dict[str, Any]:
        context = self._get_all_chunks_text(doc_id)
        if not context:
            return {"error": f"No indexed content found for doc_id={doc_id}"}

        prompt = SUMMARY_PROMPT_TEMPLATE.format(context=context)
        summary_text = self.llm.complete(prompt)

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "summary": summary_text,
        }
