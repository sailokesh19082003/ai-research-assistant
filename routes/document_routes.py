"""
Document management endpoints: upload (with background processing),
list, get, delete, and reprocess.
"""
import os
import uuid
import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from config.settings import settings
from src.database.base import get_db
from src.database.models import Document
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.vector_store.manager import get_vector_store
from src.ml.isolated_client import predict_category_isolated

router = APIRouter(prefix="/documents", tags=["Document Management"])

os.makedirs(settings.RAW_DOCS_DIR, exist_ok=True)


def process_pdf_pipeline(doc_id: str, file_path: str, file_name: str):
    """Background task: parse -> classify -> chunk -> embed & index."""
    from src.database.base import SessionLocal

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            return

        parser = PDFParser()
        pages_data, total_pages = parser.extract_text_with_metadata(file_path, doc_id)

        vector_store = get_vector_store()

        # ML classification runs in an isolated subprocess (see
        # src/ml/isolated_client.py) so TensorFlow's native libraries never
        # load inside the main server process alongside ChromaDB's.
        full_text = "\n".join(p["text"] for p in pages_data)
        category, confidence = predict_category_isolated(full_text)

        # Chunking.
        chunker = DocumentChunker()
        chunks = chunker.create_chunks(pages_data, file_name=file_name)

        # Embedding + vector indexing.
        vector_store.index_chunks(chunks)

        doc.total_pages = total_pages
        doc.total_chunks = len(chunks)
        doc.category = category
        doc.category_confidence = int(confidence * 100)
        doc.processing_status = "PROCESSED"
        db.add(doc)
        db.commit()

    except Exception as e:  # pragma: no cover
        doc = db.get(Document, doc_id)
        if doc:
            doc.processing_status = "FAILED"
            doc.error_message = f"{e}\n{traceback.format_exc()}"
            db.add(doc)
            db.commit()
    finally:
        db.close()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(settings.RAW_DOCS_DIR, f"{doc_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc = Document(
        doc_id=doc_id,
        file_name=file.filename,
        file_path=file_path,
        processing_status="PENDING",
    )
    db.add(doc)
    db.commit()

    background_tasks.add_task(process_pdf_pipeline, doc_id, file_path, file.filename)

    return {
        "message": "Document uploaded successfully. Processing started in background.",
        "doc_id": doc_id,
        "status": "PENDING",
    }


@router.get("/")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.upload_timestamp.desc()).all()
    return [
        {
            "doc_id": d.doc_id,
            "file_name": d.file_name,
            "status": d.processing_status,
            "category": d.category,
            "category_confidence_pct": d.category_confidence,
            "total_pages": d.total_pages,
            "total_chunks": d.total_chunks,
            "upload_timestamp": d.upload_timestamp,
        }
        for d in docs
    ]


@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "status": doc.processing_status,
        "category": doc.category,
        "category_confidence_pct": doc.category_confidence,
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "error_message": doc.error_message,
    }


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    get_vector_store().delete_document(doc_id)

    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    db.delete(doc)
    db.commit()
    return {"message": f"Document {doc_id} deleted."}


@router.post("/{doc_id}/reprocess")
def reprocess_document(doc_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.processing_status = "PENDING"
    doc.error_message = None
    db.add(doc)
    db.commit()

    background_tasks.add_task(process_pdf_pipeline, doc.doc_id, doc.file_path, doc.file_name)
    return {"message": "Reprocessing started.", "doc_id": doc_id}
