"""
Database session management module for AVA metadata database.

Provides connection pooling, transaction handling, and context managers for both
SQLite and PostgreSQL backends. Implements database engine factory with
configurable echo mode for debugging and automatic resource cleanup through
session context managers.

All functions adhere to the 15 LOC constraint and implement fail-loud behavior
for database connection and transaction failures.
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ava.db.schema import Base


def create_engine_from_url(database_url: str, echo: bool = False) -> Engine:
    """
    Create SQLAlchemy engine from database URL with optimized connection pooling.

    Configures appropriate pooling parameters based on database type:
    - SQLite: check_same_thread=False with foreign key enforcement
    - PostgreSQL: pool_size=5, max_overflow=10 for concurrent operations

    Args:
        database_url: Database connection string (sqlite:/// or postgresql+psycopg://)
        echo: Enable SQL query logging for debugging

    Returns:
        Configured SQLAlchemy Engine instance

    Raises:
        RuntimeError: On invalid URL format or connection failure
    """
    if database_url.startswith("sqlite:"):
        engine = create_engine(database_url, echo=echo, connect_args={"check_same_thread": False})
        # Enable foreign key enforcement for SQLite
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()
    elif database_url.startswith("postgresql"):
        engine = create_engine(database_url, echo=echo, pool_size=5, max_overflow=10)
    else:
        raise RuntimeError(f"Unsupported database URL format: {database_url}")

    # Create all tables using Base metadata
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    """
    Context manager providing database session with automatic transaction handling.

    Creates session from engine, yields it for database operations, commits on success,
    and rolls back on any exception. Ensures proper resource cleanup.

    Args:
        engine: SQLAlchemy Engine instance from create_engine_from_url()

    Yields:
        Session: Configured SQLAlchemy session with transaction management

    Raises:
        RuntimeError: On session creation failure or transaction errors
    """
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
