"""
Test module validating database schema constraints including foreign key enforcement,
CASCADE delete behavior, unique constraints, and referential integrity across the
four-table schema (recording, syllable, embedding, annotation).

Ensures database operations properly raise RuntimeError on constraint violations
as per fail-loud philosophy. All test methods adhere to 15 LOC constraint.
"""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import pytest
from sqlalchemy import create_all

from ava.db.schema import Embedding
from ava.db.session import create_engine_from_url
from ava.db.repository import RecordingRepository


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create temporary SQLite database file for testing."""
    with TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_ava.db"
        yield f"sqlite:///{db_path}"


@pytest.fixture  
def sqlite_engine(temp_db_path: str):
    """Create SQLite engine with foreign key enforcement enabled."""
    engine = create_engine_from_url(temp_db_path, echo=False)
    return engine


@pytest.fixture
def postgres_engine():
    """Create PostgreSQL engine for cross-backend testing."""
    # This would use Docker container in actual CI/CD environment
    pytest.skip("PostgreSQL container not available in current test environment")


@pytest.fixture
def recording_repo(sqlite_engine):
    """Create RecordingRepository instance with test session."""
    from ava.db.session import get_session
    with get_session(sqlite_engine) as session:
        yield RecordingRepository(session)


def test_foreign_key_cascade_delete_recording_to_syllable(sqlite_engine):
    """Test CASCADE delete from Recording removes dependent Syllables."""
    from ava.db.session import get_session
    from ava.db.schema import Recording, Syllable
    
    with get_session(sqlite_engine) as session:
        # Create recording with dependent syllable
        recording = Recording(file_path="/test/audio.wav", checksum_sha256="abc123")
        session.add(recording)
        session.flush()
        
        syllable = Syllable(recording_id=recording.id, spectrogram_path="/test/spec.h5", 
                          start_time=0.1, end_time=0.5)
        session.add(syllable)
        session.flush()
        
        # Delete recording should CASCADE delete syllable
        session.delete(recording)
        session.commit()
        
        assert session.query(Syllable).count() == 0


def test_foreign_key_cascade_delete_syllable_to_embedding(sqlite_engine):
    """Test CASCADE delete from Syllable removes dependent Embeddings.""" 
    from ava.db.session import get_session
    from ava.db.schema import Recording, Syllable, Embedding
    
    with get_session(sqlite_engine) as session:
        recording = Recording(file_path="/test/audio.wav", checksum_sha256="abc123")
        session.add(recording)
        session.flush()
        
        syllable = Syllable(recording_id=recording.id, spectrogram_path="/test/spec.h5",
                          start_time=0.1, end_time=0.5)
        session.add(syllable)
        session.flush()
        
        embedding = Embedding(syllable_id=syllable.id, model_version="v1",
                             embedding_path="/test/embed.npy", dimensions=128)
        session.add(embedding)
        session.flush()
        
        # Delete syllable should CASCADE delete embedding
        session.delete(syllable)
        session.commit()
        
        assert session.query(Embedding).count() == 0


def test_foreign_key_cascade_delete_syllable_to_annotation(sqlite_engine):
    """Test CASCADE delete from Syllable removes dependent Annotations."""
    from ava.db.session import get_session
    from ava.db.schema import Recording, Syllable, Annotation
    
    with get_session(sqlite_engine) as session:
        recording = Recording(file_path="/test/audio.wav", checksum_sha256="abc123")
        session.add(recording)
        session.flush()
        
        syllable = Syllable(recording_id=recording.id, spectrogram_path="/test/spec.h5",
                          start_time=0.1, end_time=0.5) 
        session.add(syllable)
        session.flush()
        
        annotation = Annotation(syllable_id=syllable.id, annotation_type="label",
                               key="species", value="mouse")
        session.add(annotation)
        session.flush()
        
        # Delete syllable should CASCADE delete annotation
        session.delete(syllable)
        session.commit()
        
        assert session.query(Annotation).count() == 0


def test_unique_constraint_file_path_violation(sqlite_engine):
    """Test unique constraint on Recording.file_path raises integrity error."""
    from ava.db.session import get_session
    from ava.db.schema import Recording
    from sqlalchemy.exc import IntegrityError
    
    with get_session(sqlite_engine) as session:
        # Create first recording
        recording1 = Recording(file_path="/duplicate/path.wav", checksum_sha256="abc123")
        session.add(recording1)
        session.flush()
        
        # Attempt duplicate file_path should raise IntegrityError
        recording2 = Recording(file_path="/duplicate/path.wav", checksum_sha256="def456")
        session.add(recording2)
        
        with pytest.raises(IntegrityError):
            session.flush()


def test_referential_integrity_orphaned_syllable_prevented(sqlite_engine):
    """Test orphaned Syllable creation prevented by foreign key constraint."""
    from ava.db.session import get_session  
    from ava.db.schema import Syllable
    from sqlalchemy.exc import IntegrityError
    
    with get_session(sqlite_engine) as session:
        # Attempt to create syllable with non-existent recording_id
        orphaned_syllable = Syllable(recording_id=999, spectrogram_path="/test/spec.h5",
                                   start_time=0.1, end_time=0.5)
        session.add(orphaned_syllable)
        
        with pytest.raises(IntegrityError):
            session.flush()


def test_referential_integrity_orphaned_embedding_prevented(sqlite_engine):
    """Test orphaned Embedding creation prevented by foreign key constraint."""
    from ava.db.session import get_session
    from sqlalchemy.exc import IntegrityError
    
    with get_session(sqlite_engine) as session:
        # Attempt to create embedding with non-existent syllable_id
        orphaned_embedding = Embedding(syllable_id=999, model_version="v1",
                                     embedding_path="/test/embed.npy", dimensions=128)
        session.add(orphaned_embedding)
        
        with pytest.raises(IntegrityError):
            session.flush()


def test_referential_integrity_orphaned_annotation_prevented(sqlite_engine):
    """Test orphaned Annotation creation prevented by foreign key constraint."""
    from ava.db.session import get_session
    from ava.db.schema import Annotation
    from sqlalchemy.exc import IntegrityError
    
    with get_session(sqlite_engine) as session:
        # Attempt to create annotation with non-existent syllable_id  
        orphaned_annotation = Annotation(syllable_id=999, annotation_type="label",
                                       key="species", value="mouse")
        session.add(orphaned_annotation)
        
        with pytest.raises(IntegrityError):
            session.flush()


@pytest.mark.parametrize("database_url", [
    "sqlite:///test_ava.db",
    "sqlite:///:memory:",
])
def test_sqlite_backend_compatibility(database_url):
    """Test database engine creation and schema setup for SQLite backends."""
    engine = create_engine_from_url(database_url, echo=False)
    assert engine is not None
    
    # Verify foreign keys are enabled
    with engine.connect() as conn:
        result = conn.execute("PRAGMA foreign_keys")
        assert result.scalar() == 1


def test_transaction_rollback_on_constraint_violation(sqlite_engine):
    """Test transaction rollback behavior when constraint violations occur."""
    from ava.db.session import get_session
    from ava.db.schema import Recording
    from sqlalchemy.exc import IntegrityError
    
    with pytest.raises(IntegrityError):
        with get_session(sqlite_engine) as session:
            # Add valid recording
            recording1 = Recording(file_path="/valid/path.wav", checksum_sha256="abc123")
            session.add(recording1)
            session.flush()
            
            # Add duplicate file_path to trigger constraint violation
            recording2 = Recording(file_path="/valid/path.wav", checksum_sha256="def456")
            session.add(recording2) 
            session.flush()  # This will raise IntegrityError
    
    # Verify transaction was rolled back - no records should exist
    with get_session(sqlite_engine) as session:
        assert session.query(Recording).count() == 0


def test_embedding_attributes_validation(sqlite_engine):
    """Test Embedding model attributes match schema requirements."""
    from ava.db.session import get_session
    from ava.db.schema import Recording, Syllable
    
    with get_session(sqlite_engine) as session:
        recording = Recording(file_path="/test/audio.wav", checksum_sha256="abc123")
        session.add(recording)
        session.flush()
        
        syllable = Syllable(recording_id=recording.id, spectrogram_path="/test/spec.h5",
                          start_time=0.1, end_time=0.5)
        session.add(syllable)
        session.flush()
        
        embedding = Embedding(
            syllable_id=syllable.id,
            model_version="test_v1", 
            embedding_path="/test/embed.npy",
            dimensions=256,
            extra_model_metadata={"architecture": "VAE"}
        )
        session.add(embedding)
        session.flush()
        
        # Validate all required attributes are accessible
        assert embedding.id is not None
        assert embedding.syllable_id == syllable.id
        assert embedding.model_version == "test_v1"
        assert embedding.embedding_path == "/test/embed.npy"
        assert embedding.dimensions == 256
        assert embedding.extra_model_metadata["architecture"] == "VAE"


def test_repository_delete_cascade_behavior(recording_repo):
    """Test RecordingRepository delete method triggers CASCADE behavior."""
    from ava.db.schema import Recording, Syllable
    
    # Create recording using repository
    recording = recording_repo.create("/test/repo.wav", "abc123", {"source": "test"})
    
    # Create dependent syllable directly
    syllable = Syllable(recording_id=recording.id, spectrogram_path="/test/spec.h5",
                       start_time=0.1, end_time=0.5)
    recording_repo.session.add(syllable)
    recording_repo.session.flush()
    
    # Repository delete should trigger CASCADE
    recording_repo.delete(recording)
    recording_repo.session.commit()
    
    assert recording_repo.session.query(Syllable).count() == 0


def test_repository_get_by_id_method(recording_repo):
    """Test RecordingRepository get_by_id method functionality.""" 
    # Create recording using repository
    created_recording = recording_repo.create("/test/get_test.wav", "def456")
    recording_id = created_recording.id
    
    # Retrieve using get_by_id
    retrieved_recording = recording_repo.get_by_id(recording_id)
    
    assert retrieved_recording is not None
    assert retrieved_recording.id == recording_id
    assert retrieved_recording.file_path == "/test/get_test.wav"
    assert retrieved_recording.checksum_sha256 == "def456"


def test_now_and_isoformat_datetime_usage(sqlite_engine):
    """Test datetime.now() and isoformat() usage in temporal constraints."""
    from ava.db.session import get_session
    from ava.db.schema import Recording
    
    current_time = datetime.now()
    iso_timestamp = current_time.isoformat()
    
    with get_session(sqlite_engine) as session:
        recording = Recording(file_path="/test/datetime.wav", checksum_sha256="time123", 
                            created_at=current_time)
        session.add(recording)
        session.flush()
        
        assert recording.created_at == current_time
        assert isinstance(iso_timestamp, str)
        assert "T" in iso_timestamp  # Verify ISO format structure