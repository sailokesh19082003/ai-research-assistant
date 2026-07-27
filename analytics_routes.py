"""
System analytics endpoint: usage stats, chunk counts, category distribution,
top-referenced documents.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.analytics.metrics import get_system_analytics

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/")
def analytics(db: Session = Depends(get_db)):
    return get_system_analytics(db)
