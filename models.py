"""
ORM schemas: Document metadata, chat sessions, and query analytics.
"""
import uuid
import datetime as dt

from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship

from src.database.base import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, default=gen_uuid)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=dt.datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")  # PENDING/PROCESSED/FAILED
    category = Column(String, default="Uncategorized")
    category_confidence = Column(Integer, default=0)  # stored as % (0-100)
    error_message = Column(Text, nullable=True)

    query_logs = relationship("QueryLog", back_populates="document")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, default=gen_uuid)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    history = Column(Text, default="")  # newline-delimited "Q:...\nA:..." transcript


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)

    document = relationship("Document", back_populates="query_logs")
