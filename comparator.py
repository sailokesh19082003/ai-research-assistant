"""
Compares research methodologies / content across multiple documents:
methodologies, advantages/disadvantages, similarities, and implementation
approaches.
"""
from typing import List, Dict, Any

from src.vector_store.manager import get_vector_store
from src.rag.llm_engine import get_llm

COMPARISON_PROMPT_TEMPLATE = """You are an AI Research Assistant comparing multiple technical documents.
Use ONLY the content provided below for each document. Do not hallucinate facts absent from the text.

{documents_block}

Compare the documents above and produce a structured analysis covering:
1. Methodologies used in each document
2. Advantages / Disadvantages of each approach
3. Similarities across the documents
4. Implementation Approaches (practical differences)

If any document lacks sufficient content to compare on a given dimension, state that explicitly rather than guessing.
"""


class DocumentComparator:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()

    def _get_doc_text(self, doc_id: str, max_chunks: int = 25) -> str:
        data = self.vector_store.collection.get(where={"doc_id": doc_id}, include=["documents"])
        chunks = data.get("documents", []) or []
        return "\n".join(chunks[:max_chunks])

    def compare_documents(self, doc_infos: List[Dict[str, str]]) -> Dict[str, Any]:
        """doc_infos: list of {"doc_id": ..., "file_name": ...}"""
        documents_block = ""
        missing = []
        for info in doc_infos:
            text = self._get_doc_text(info["doc_id"])
            if not text:
                missing.append(info["file_name"])
                continue
            documents_block += f"\n=== Document: {info['file_name']} (doc_id={info['doc_id']}) ===\n{text}\n"

        if not documents_block:
            return {"error": "None of the requested documents have indexed content."}

        prompt = COMPARISON_PROMPT_TEMPLATE.format(documents_block=documents_block)
        comparison_text = self.llm.complete(prompt)

        return {
            "documents_compared": [d["file_name"] for d in doc_infos if d["file_name"] not in missing],
            "documents_missing_content": missing,
            "comparison": comparison_text,
        }
