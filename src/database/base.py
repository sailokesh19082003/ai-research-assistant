"""
Database engine + session management (SQLite by default, Postgres-ready).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import settings

DATABASE_URL = f"sqlite:///{settings.SQLITE_DB_PATH}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once on application startup."""
    from src.database import models  # noqa: F401 (register models)
    Base.metadata.create_all(bind=engine)
