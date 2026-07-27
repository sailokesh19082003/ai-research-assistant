"""
Retrieval-Augmented Generation pipeline: retrieves top-K chunks, builds a
strictly-grounded prompt with citations, and answers using session-based
conversation memory so follow-up questions ("summarize it", "its
limitations?") resolve correctly.
"""
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from config.settings import settings
from src.vector_store.manager import get_vector_store
from src.rag.llm_engine import get_llm
from src.database.models import ChatSession, QueryLog

PROMPT_TEMPLATE = """You are an AI Research Assistant. Answer the user's question using ONLY the provided document context below.
If the context does not contain sufficient information to answer, state clearly: "I cannot determine the answer from the provided documents."

Conversation History:
{history}

Context:
{context}

Question: {question}

Provide a clear, direct answer followed by an explicit list of source documents and page references.
"""


class RAGQuestionAnswering:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()

    def _get_or_create_session(self, db: Session, session_id: Optional[str]) -> ChatSession:
        if session_id:
            session = db.get(ChatSession, session_id)
            if session:
                return session
        session = ChatSession(history="")
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def answer_question(
        self,
        db: Session,
        query: str,
        session_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        search_mode: str = "hybrid",
    ) -> Dict[str, Any]:
        session = self._get_or_create_session(db, session_id)

        if search_mode == "semantic":
            docs = self.vector_store.semantic_search(query, k=settings.TOP_K, doc_ids=doc_ids)
        elif search_mode == "keyword":
            docs = self.vector_store.keyword_search(query, k=settings.TOP_K, doc_ids=doc_ids)
        else:
            docs = self.vector_store.hybrid_search(query, k=settings.TOP_K, doc_ids=doc_ids)

        context_str = ""
        citations = []
        for d in docs:
            file_name = d["metadata"].get("file_name", "Unknown")
            page_no = d["metadata"].get("page_number", "N/A")
            context_str += f"\n--- Source: {file_name} (Page {page_no}) ---\n{d['text']}\n"
            citations.append({"document": file_name, "page": page_no, "relevance_score": round(d["score"], 4)})

        prompt = PROMPT_TEMPLATE.format(
            history=session.history or "(no prior turns)",
            context=context_str if context_str else "(no relevant context retrieved)",
            question=query,
        )

        answer = self.llm.complete(prompt)

        # Update conversational memory.
        session.history = (session.history or "") + f"\nQ: {query}\nA: {answer}\n"
        db.add(session)

        # Log for analytics.
        log = QueryLog(
            session_id=session.session_id,
            doc_id=doc_ids[0] if doc_ids else None,
            question=query,
            answer=answer,
        )
        db.add(log)
        db.commit()

        return {
            "session_id": session.session_id,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": len(docs),
        }
