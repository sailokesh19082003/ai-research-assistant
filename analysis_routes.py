"""
Endpoints: document summarization, multi-document comparison, and standalone
text classification (useful for testing the ML model without an upload).
"""
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import Document
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.ml.isolated_client import predict_category_isolated

router = APIRouter(prefix="/analysis", tags=["Analysis"])


class CompareRequest(BaseModel):
    doc_ids: List[str]


class ClassifyRequest(BaseModel):
    text: str


@router.get("/summarize/{doc_id}")
def summarize(doc_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.processing_status != "PROCESSED":
        raise HTTPException(status_code=400, detail=f"Document status is '{doc.processing_status}', not ready to summarize.")

    summarizer = DocumentSummarizer()
    return summarizer.summarize_document(doc_id, doc.file_name)


@router.post("/compare")
def compare(req: CompareRequest, db: Session = Depends(get_db)):
    if len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 doc_ids to compare.")

    doc_infos = []
    for doc_id in req.doc_ids:
        doc = db.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
        doc_infos.append({"doc_id": doc.doc_id, "file_name": doc.file_name})

    comparator = DocumentComparator()
    return comparator.compare_documents(doc_infos)


@router.post("/classify")
def classify_text(req: ClassifyRequest):
    category, confidence = predict_category_isolated(req.text)
    return {"category": category, "confidence": round(confidence, 4)}
