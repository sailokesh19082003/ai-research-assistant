"""
Endpoints: semantic/keyword/hybrid search, and RAG-based QA with citations.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.vector_store.manager import get_vector_store
from src.rag.qa_chain import RAGQuestionAnswering

router = APIRouter(prefix="/search", tags=["Search & QA"])


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # semantic | keyword | hybrid
    k: int = 4
    doc_ids: Optional[List[str]] = None


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    mode: str = "hybrid"


@router.post("/")
def search(req: SearchRequest):
    store = get_vector_store()
    if req.mode == "semantic":
        results = store.semantic_search(req.query, k=req.k, doc_ids=req.doc_ids)
    elif req.mode == "keyword":
        results = store.keyword_search(req.query, k=req.k, doc_ids=req.doc_ids)
    else:
        results = store.hybrid_search(req.query, k=req.k, doc_ids=req.doc_ids)
    return {"query": req.query, "mode": req.mode, "results": results}


@router.post("/ask")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    rag = RAGQuestionAnswering()
    return rag.answer_question(
        db=db,
        query=req.query,
        session_id=req.session_id,
        doc_ids=req.doc_ids,
        search_mode=req.mode,
    )
