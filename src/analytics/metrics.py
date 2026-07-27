"""
System analytics: usage stats, total indexed chunks, category distribution,
and top-referenced documents.
"""
from collections import Counter
from typing import Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.models import Document, QueryLog
from src.vector_store.manager import get_vector_store


def get_system_analytics(db: Session) -> Dict[str, Any]:
    total_documents = db.query(func.count(Document.doc_id)).scalar() or 0
    total_chunks = db.query(func.sum(Document.total_chunks)).scalar() or 0
    total_queries = db.query(func.count(QueryLog.id)).scalar() or 0

    category_rows = db.query(Document.category, func.count(Document.doc_id)).group_by(Document.category).all()
    category_distribution = {cat or "Uncategorized": count for cat, count in category_rows}

    top_docs_rows = (
        db.query(QueryLog.doc_id, func.count(QueryLog.id).label("query_count"))
        .filter(QueryLog.doc_id.isnot(None))
        .group_by(QueryLog.doc_id)
        .order_by(func.count(QueryLog.id).desc())
        .limit(5)
        .all()
    )

    top_documents = []
    for doc_id, query_count in top_docs_rows:
        doc = db.get(Document, doc_id)
        if doc:
            top_documents.append({
                "doc_id": doc_id,
                "file_name": doc.file_name,
                "query_count": query_count,
            })

    # Live count straight from the vector index, as a cross-check.
    try:
        vector_store = get_vector_store()
        live_vector_count = vector_store.collection.count()
    except Exception:
        live_vector_count = None

    return {
        "total_documents_indexed": total_documents,
        "total_text_chunks": int(total_chunks),
        "total_queries_answered": total_queries,
        "documents_by_category": category_distribution,
        "top_referenced_documents": top_documents,
        "live_vector_index_size": live_vector_count,
    }
