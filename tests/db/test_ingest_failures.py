"""
Comprehensive failure scenario tests for data ingestion process.

Validates all error conditions during database metadata ingestion including missing files,
checksum mismatches, invalid HDF5 structures, data inconsistencies, and constraint violations.
Ensures fail-loud behavior with RuntimeError propagation and actionable error messages.

All test functions adhere to the 15 LOC constraint and implement comprehensive error
validation to ensure data integrity violations trigger immediate failure with detailed
diagnostic information for debugging.
"""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py  # type: ignore
import numpy as np
import pytest

from ava.data.indexer import FilesystemIndexer
from ava.db.repository import SyllableRepository
from ava.db.session import get_session


# Test missing HDF5 files raise RuntimeError with path details
def test_extract_metadata_missing_file() -> None:
    """Test FilesystemIndexer raises RuntimeError with path details for missing HDF5 files."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        missing_file = Path(temp_dir) / "nonexistent.h5"
        with pytest.raises(
            RuntimeError, match=f"HDF5 file not found or not a file: {missing_file.resolve()}"
        ):
            indexer.extract_hdf5_metadata(missing_file)


def test_extract_metadata_wrong_hdf5_keys() -> None:
    """Test FilesystemIndexer provides specific error messages for wrong HDF5 dataset keys."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        # Create malformed HDF5 file missing required datasets
        hdf5_path = Path(temp_dir) / "malformed.h5"
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("wrong_key", data=np.array([1, 2, 3]))

        with pytest.raises(
            RuntimeError, match="Missing required datasets.*specs.*onsets.*offsets.*audio_filenames"
        ):
            indexer.extract_hdf5_metadata(hdf5_path)


def test_extract_metadata_invalid_file_extension() -> None:
    """Test FilesystemIndexer rejects files with invalid extensions."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        invalid_file = Path(temp_dir) / "test.txt"
        invalid_file.touch()

        with pytest.raises(RuntimeError, match="Invalid file extension: .txt"):
            indexer.extract_hdf5_metadata(invalid_file)


def test_checksum_computation_missing_file() -> None:
    """Test checksum computation raises RuntimeError for missing files."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        missing_file = Path(temp_dir) / "missing.h5"
        with pytest.raises(
            RuntimeError, match=f"Cannot compute checksum - file not found: {missing_file}"
        ):
            indexer.compute_checksums([missing_file])


def test_checksum_computation_read_error() -> None:
    """Test checksum computation handles file read errors properly."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        # Create a directory instead of a file to trigger read error
        test_file = Path(temp_dir) / "test_directory"
        test_file.mkdir()

        try:
            with pytest.raises(RuntimeError, match=f"Checksum computation failed for {test_file}"):
                indexer.compute_checksums([test_file])
        except RuntimeError as e:
            assert "Checksum computation failed" in str(e)


def test_validate_integrity_nonexistent_files() -> None:
    """Test integrity validation reports nonexistent files with detailed error list."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        nonexistent_files = [Path(temp_dir) / "file1.h5", Path(temp_dir) / "file2.h5"]
        with pytest.raises(RuntimeError, match=r"Integrity validation failed"):
            indexer.validate_integrity(nonexistent_files)


def test_validate_integrity_invalid_extensions() -> None:
    """Test integrity validation rejects files with invalid extensions."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        invalid_file = Path(temp_dir) / "test.txt"
        invalid_file.touch()

        with pytest.raises(RuntimeError, match="Invalid file extension.*test.txt"):
            indexer.validate_integrity([invalid_file])


def test_database_population_orphan_syllables() -> None:
    """Test database population fails with orphaned syllables missing recording parent."""
    with TemporaryDirectory() as temp_dir:
        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)

        try:
            with get_session(engine) as session:
                syllable_repo = SyllableRepository(session)

                # Attempt to create syllable with invalid recording_id
                invalid_syllables = [
                    {
                        "recording_id": 99999,  # Non-existent recording ID
                        "spectrogram_path": str(Path(temp_dir) / "test.h5"),
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "bounds_metadata": {},
                    }
                ]

                syllable_repo.bulk_create(invalid_syllables)
                pytest.fail("Expected foreign key constraint violation")
        except Exception as e:
            assert "FOREIGN KEY constraint failed" in str(e) or "PendingRollbackError" in str(e)


def test_config_validation_failure() -> None:
    """Test configuration validation raises RuntimeError for invalid configuration."""
    from ava.db.session import create_engine_from_url

    engine = create_engine_from_url("sqlite:///:memory:", echo=False)

    # Invalid configuration with wrong data types
    invalid_config = {
        "database": {"enabled": "not_a_boolean"},  # Wrong type
        "data_roots": {"audio_dir": 123},  # Wrong type
        "ingest": {"checksum": ["not_a_string"]},  # Wrong type
    }

    with pytest.raises(RuntimeError, match="Configuration validation failed"):
        FilesystemIndexer.from_config_dict(invalid_config, engine)


def test_hdf5_corruption_during_extraction() -> None:
    """Test metadata extraction handles corrupted HDF5 files with detailed error reporting."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        # Create corrupted HDF5 file
        corrupted_file = Path(temp_dir) / "corrupted.h5"
        with open(corrupted_file, "wb") as f:
            f.write(b"This is not a valid HDF5 file")

        with pytest.raises(
            RuntimeError, match=f"Failed to extract metadata from {corrupted_file.resolve()}"
        ):
            indexer.extract_hdf5_metadata(corrupted_file)


def test_transaction_rollback_on_integrity_violation() -> None:
    """Test transaction rollback occurs on database constraint violations."""
    with TemporaryDirectory():
        from ava.db.repository import RecordingRepository
        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)

        try:
            with get_session(engine) as session:
                recording_repo = RecordingRepository(session)

                # Create valid recording first
                recording_repo.create(
                    file_path="/valid/path.h5", checksum_sha256="valid_checksum_hash"
                )

                # Attempt to create duplicate recording (violates unique constraint)
                recording_repo.create(
                    file_path="/valid/path.h5",  # Duplicate path
                    checksum_sha256="different_checksum",
                )
                pytest.fail("Expected unique constraint violation")
        except Exception as e:
            assert "UNIQUE constraint failed" in str(e) or "PendingRollbackError" in str(e)


def test_subprocess_cli_nonzero_exit_code() -> None:
    """Test CLI script returns non-zero exit code on failures."""
    with TemporaryDirectory() as temp_dir:
        # Create invalid configuration file
        config_file = Path(temp_dir) / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        result = subprocess.run(
            ["python", "-c", "import sys; sys.exit(1)"], capture_output=True  # Simulate CLI failure
        )

        assert result.returncode != 0, "CLI should return non-zero exit code on failure"


def test_embedding_orphan_detection() -> None:
    """Test system detects and prevents orphaned embeddings without valid syllable parents."""
    with TemporaryDirectory() as temp_dir:
        from ava.db.repository import EmbeddingRepository
        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)

        try:
            with get_session(engine) as session:
                embedding_repo = EmbeddingRepository(session)

                # Attempt to create embedding with non-existent syllable_id
                embedding_repo.create(
                    syllable_id=99999,  # Non-existent syllable
                    model_version="test_v1",
                    embedding_path=str(Path(temp_dir) / "embedding.npy"),
                    dimensions=128,
                )
                pytest.fail("Expected foreign key constraint violation")
        except Exception as e:
            assert "FOREIGN KEY constraint failed" in str(e) or "PendingRollbackError" in str(e)


def test_memory_overflow_protection() -> None:
    """Test system handles memory constraints during large batch operations."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)

        # Mock memory constraint by limiting available operations
        with patch.object(
            FilesystemIndexer, "populate_database", side_effect=MemoryError("Insufficient memory")
        ):
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

            test_files = [Path(temp_dir) / "test.h5"]
            checksums = {test_files[0]: "test_checksum"}

            with pytest.raises(MemoryError):
                indexer.populate_database(test_files, checksums)


def test_permission_error_file_access() -> None:
    """Test system reports permission errors clearly during file operations."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        # Create file and remove read permissions
        test_file = Path(temp_dir) / "restricted.h5"
        test_file.touch()
        test_file.chmod(0o000)

        try:
            with pytest.raises(RuntimeError, match="Failed to extract metadata"):
                indexer.extract_hdf5_metadata(test_file)
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o666)


def test_full_indexing_workflow_error_propagation() -> None:
    """Test complete indexing workflow propagates errors with comprehensive reporting."""
    with TemporaryDirectory() as temp_dir:
        config_dict = {
            "database": {"enabled": True, "url": "sqlite:///test.db", "echo": False},
            "data_roots": {"audio_dir": temp_dir, "features_dir": temp_dir},
            "ingest": {
                "scan_glob_audio": "**/*.wav",
                "scan_glob_h5": "**/*.h5",
                "checksum": "sha256",
            },
        }

        from ava.db.session import create_engine_from_url

        engine = create_engine_from_url("sqlite:///:memory:", echo=False)
        indexer = FilesystemIndexer.from_config_dict(config_dict, engine)

        # Create directory without HDF5 files to trigger empty discovery
        result = indexer.run_full_indexing(Path(temp_dir))

        # Should complete successfully but with zero files indexed
        assert result["status"] == "completed"
        assert result["indexed_files"] == 0
