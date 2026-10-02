"""Database Configuration and SQLAlchemy Session Management for Supabase PostgreSQL.

Provides database engine configuration, scoped session management, and context managers
connecting to Supabase PostgreSQL via DATABASE_URL with zero credential hardcoding.
"""

from contextlib import contextmanager
import os
import re
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Base class for all SQLAlchemy ORM models
Base = declarative_base()


def get_database_url() -> str:
    """Resolve database URL from environment with safe fallback.

    Normalizes 'postgres://' URI scheme to 'postgresql://' as required by SQLAlchemy 2.x.
    """
    raw_url = os.getenv("DATABASE_URL", "").strip()

    if not raw_url:
        # Fallback to local SQLite for isolated test/dev when live Supabase URL is not set
        fallback_url = "sqlite:///ai_screening_local.db"
        return fallback_url

    # SQLAlchemy 2.0 requires postgresql:// rather than postgres://
    if raw_url.startswith("postgres://"):
        raw_url = re.sub(r"^postgres://", "postgresql://", raw_url, count=1)

    return raw_url


def create_db_engine(db_url: str = None):
    """Create a configured SQLAlchemy engine."""
    url = db_url or get_database_url()

    engine_kwargs = {}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL / Supabase pool configuration
        engine_kwargs["pool_pre_ping"] = True
        engine_kwargs["pool_recycle"] = 300

    return create_engine(url, **engine_kwargs)


# Module-level engine and sessionmaker
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session scope with automatic rollback and cleanup."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """Yield database session for Flask request lifecycle or dependency injection."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(target_engine=None):
    """Create all registered database tables (used primarily in development and test environments)."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
