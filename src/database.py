from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.models import Base

DATABASE_URL = "sqlite:///./db/database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Session:
    """Dependency function to get a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        # This is a simplified version for scripts.
        # In a web app, you'd close it after the request.
        pass


def create_tables() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
