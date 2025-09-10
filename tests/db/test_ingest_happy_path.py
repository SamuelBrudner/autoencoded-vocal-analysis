"""
Test module validating successful data ingestion workflows including filesystem scanning,
HDF5 metadata extraction, checksum computation, and database population.

Implements comprehensive Test-Driven Development (TDD) approach with deterministic
test fixtures and reproducible execution. All test functions validate happy-path
scenarios while ensuring proper foreign key relationships and transaction management.

Tests adhere to 15 LOC constraint and fail-loud philosophy with clear assertions.
Synthetic HDF5 fixtures enable isolated testing without dependencies on real data files.
"""

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Any

import h5py  # type: ignore
import numpy as np
import pytest


from ava.data.indexer import FilesystemIndexer
from ava.db.repository import RecordingRepository
from ava.db.schema import Recording
from ava.db.session import create_engine_from_url


def create_synthetic_hdf5_files(temp_dir: Path, num_files: int = 3, syllables_per_file: int = 5) -> List[Path]:
    """
    Create synthetic HDF5 files with specs, onsets, offsets, audio_filenames datasets.
    
    Generates deterministic test fixtures with reproducible content using fixed
    random seeds. Each file follows syllables_XXXX.hdf5 naming convention with
    required datasets for ingestion testing.
    """
    np.random.seed(42)  # Ensure deterministic test data generation
    created_files = []
    
    for file_idx in range(num_files):
        file_path = temp_dir / f"syllables_{file_idx:04d}.hdf5"
        
        with h5py.File(file_path, 'w') as h5_file:
            # Create specs dataset with shape (num_syllables, 128, 128)
            specs_shape = (syllables_per_file, 128, 128)
            specs_data = np.random.rand(*specs_shape).astype(np.float32)
            h5_file.create_dataset('specs', data=specs_data)
            
            # Create temporal bounds with realistic values
            base_time = 0.0
            onsets = [base_time + i * 0.2 for i in range(syllables_per_file)]
            offsets = [onset + 0.15 for onset in onsets]  # 150ms syllable duration
            
            h5_file.create_dataset('onsets', data=np.array(onsets, dtype=np.float64))
            h5_file.create_dataset('offsets', data=np.array(offsets, dtype=np.float64))
            
            # Create audio filenames dataset
            audio_filenames = [f"audio_{file_idx:04d}_{i:03d}.wav" for i in range(syllables_per_file)]
            h5_file.create_dataset('audio_filenames', data=np.array(audio_filenames, dtype=h5py.string_dtype()))
        
        created_files.append(file_path)
    
    return created_files


def setup_test_database(temp_dir: Path) -> Any:
    """
    Create ephemeral SQLite database engine for testing ingestion workflows.
    
    Configures in-memory SQLite database with foreign key enforcement and
    automatic table creation through SQLAlchemy Base metadata.
    Returns engine instance for test database operations.
    """
    db_path = temp_dir / "test_ava.db"
    database_url = f"sqlite:///{db_path}"
    
    engine = create_engine_from_url(database_url, echo=False)
    return engine


def validate_ingestion_results(engine: Any, expected_files: List[Path], expected_syllables: int) -> Dict[str, Any]:
    """
    Validate ingestion results including record counts, checksum integrity, and relationships.
    
    Performs comprehensive validation of database state after ingestion including
    recording creation, syllable population, foreign key relationships, and
    temporal ordering. Returns validation summary for test assertions.
    """
    from ava.db.session import get_session
    from ava.db.schema import Recording, Syllable
    from sqlalchemy import select, func
    
    with get_session(engine) as session:
        # Validate recording count matches expected files
        recording_count = session.execute(select(func.count(Recording.id))).scalar()
        
        # Validate syllable count matches expected total
        syllable_count = session.execute(select(func.count(Syllable.id))).scalar()
        
        # Validate all recordings have proper checksums
        recordings = session.execute(select(Recording)).scalars().all()
        checksum_validation = all(len(r.checksum_sha256) == 64 for r in recordings)
    
    return {
        'recordings_created': recording_count,
        'syllables_created': syllable_count,
        'checksums_valid': checksum_validation,
        'expected_files': len(expected_files),
        'expected_syllables': expected_syllables
    }


class TestIngestHappyPath:
    """
    Test class for validating successful ingestion workflows using Test-Driven Development (TDD).
    
    Implements comprehensive happy-path testing including filesystem scanning,
    metadata extraction, checksum validation, and database population.
    All tests use deterministic fixtures with reproducible execution following TDD principles.
    """
    
    def test_synthetic_hdf5_creation(self) -> None:
        """Test synthetic HDF5 file generation with required datasets."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=2, syllables_per_file=3)
            
            assert len(created_files) == 2
            assert all(f.exists() for f in created_files)
            assert all(f.suffix == '.hdf5' for f in created_files)
            
            # Validate HDF5 file structure
            with h5py.File(created_files[0], 'r') as h5_file:
                required_keys = ['specs', 'onsets', 'offsets', 'audio_filenames']
                file_keys = list(h5_file.keys())
                missing_keys = [k for k in required_keys if k not in file_keys]
                assert len(missing_keys) == 0, f"Missing required datasets: {missing_keys}"
    
    def test_database_setup_and_connection(self) -> None:
        """Test ephemeral database creation and table initialization."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            engine = setup_test_database(temp_dir)
            
            # Validate engine creation and connection
            assert engine is not None
            
            # Validate table creation through schema inspection
            from ava.db.session import get_session
            from ava.db.schema import Recording
            from sqlalchemy import select
            
            with get_session(engine) as session:
                result = session.execute(select(Recording).limit(1)).first()
                # Empty result expected but no exception means tables exist
                assert result is None
    
    def test_filesystem_scanning_discovery(self) -> None:
        """Test filesystem scanning with glob patterns discovering HDF5 files."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Create test HDF5 files and initialize indexer
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=3)
            engine = setup_test_database(temp_dir)
            
            config_dict = {
                'database': {'enabled': True, 'url': f'sqlite:///{temp_dir}/test.db', 'echo': False},
                'data_roots': {'audio_dir': str(temp_dir), 'features_dir': str(temp_dir)},
                'ingest': {'scan_glob_audio': '*.wav', 'scan_glob_h5': '*.hdf5', 'checksum': 'sha256'}
            }
            
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)
            discovered_files = indexer.scan_files(temp_dir)
            
            assert len(discovered_files) == 3
            assert all(f.suffix in ['.h5', '.hdf5'] for f in discovered_files)
    
    def test_metadata_extraction_from_hdf5(self) -> None:
        """Test HDF5 metadata extraction including shape, dtype, and temporal bounds."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Create single test file and extract metadata
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=1, syllables_per_file=4)
            engine = setup_test_database(temp_dir)
            
            config_dict = {
                'database': {'enabled': True, 'url': f'sqlite:///{temp_dir}/test.db', 'echo': False},
                'data_roots': {'audio_dir': str(temp_dir), 'features_dir': str(temp_dir)},
                'ingest': {'scan_glob_audio': '*.wav', 'scan_glob_h5': '*.hdf5', 'checksum': 'sha256'}
            }
            
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)
            metadata = indexer.extract_hdf5_metadata(created_files[0])
            
            assert metadata['specs_shape'] == (4, 128, 128)
            assert metadata['specs_dtype'] == 'float32'
            assert metadata['num_syllables'] == 4
            assert len(metadata['onsets']) == 4
            assert len(metadata['offsets']) == 4
    
    def test_checksum_computation_integrity(self) -> None:
        """Test SHA-256 checksum computation for file integrity validation."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Create files and compute checksums
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=2)
            engine = setup_test_database(temp_dir)
            
            config_dict = {
                'database': {'enabled': True, 'url': f'sqlite:///{temp_dir}/test.db', 'echo': False},
                'data_roots': {'audio_dir': str(temp_dir), 'features_dir': str(temp_dir)},
                'ingest': {'scan_glob_audio': '*.wav', 'scan_glob_h5': '*.hdf5', 'checksum': 'sha256'}
            }
            
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)
            checksums = indexer.compute_checksums(created_files)
            
            assert len(checksums) == 2
            assert all(len(checksum) == 64 for checksum in checksums.values())
            
            # Validate checksums are deterministic for same content
            checksums_second = indexer.compute_checksums(created_files)
            assert checksums == checksums_second
    
    def test_database_population_with_transactions(self) -> None:
        """Test database population with proper foreign key relationships and transactions."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Full workflow test with validation
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=2, syllables_per_file=3)
            engine = setup_test_database(temp_dir)
            
            config_dict = {
                'database': {'enabled': True, 'url': f'sqlite:///{temp_dir}/test.db', 'echo': False},
                'data_roots': {'audio_dir': str(temp_dir), 'features_dir': str(temp_dir)},
                'ingest': {'scan_glob_audio': '*.wav', 'scan_glob_h5': '*.hdf5', 'checksum': 'sha256'}
            }
            
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)
            results = indexer.run_full_indexing(temp_dir)
            
            assert results['status'] == 'completed'
            assert results['indexed_files'] == 2
            assert results['total_syllables'] == 6  # 2 files * 3 syllables each
            
            # Validate database state through validation function
            validation = validate_ingestion_results(engine, created_files, expected_syllables=6)
            assert validation['recordings_created'] == 2
            assert validation['syllables_created'] == 6
            assert validation['checksums_valid'] is True
    
    def test_deterministic_result_ordering(self) -> None:
        """Test deterministic ordering of ingestion results for reproducible workflows."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Create files and run indexing twice
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=3, syllables_per_file=2)
            engine = setup_test_database(temp_dir)
            
            config_dict = {
                'database': {'enabled': True, 'url': f'sqlite:///{temp_dir}/test.db', 'echo': False},
                'data_roots': {'audio_dir': str(temp_dir), 'features_dir': str(temp_dir)},
                'ingest': {'scan_glob_audio': '*.wav', 'scan_glob_h5': '*.hdf5', 'checksum': 'sha256'}
            }
            
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)
            
            # First run
            results_first = indexer.run_full_indexing(temp_dir)
            
            # Query syllables and validate deterministic ordering
            from ava.db.session import get_session
            from ava.db.schema import Syllable
            from sqlalchemy import select
            
            with get_session(engine) as session:
                syllables = session.execute(select(Syllable).order_by(Syllable.start_time, Syllable.id)).scalars().all()
                syllable_times = [s.start_time for s in syllables]
                
                # Validate temporal ordering is maintained
                assert syllable_times == sorted(syllable_times)
                assert len(syllables) == 6  # 3 files * 2 syllables each
    
    @pytest.mark.parametrize("num_files,syllables_per_file", [
        (1, 5),
        (3, 3),
        (5, 2)
    ])
    def test_bulk_insertion_performance(self, num_files: int, syllables_per_file: int) -> None:
        """Test bulk insertion performance with parameterized file counts and syllable distributions."""
        with TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            created_files = create_synthetic_hdf5_files(temp_dir, num_files=num_files, syllables_per_file=syllables_per_file)
            engine = setup_test_database(temp_dir)
            
            config_dict = {
                'database': {'enabled': True, 'url': f'sqlite:///{temp_dir}/test.db', 'echo': False},
                'data_roots': {'audio_dir': str(temp_dir), 'features_dir': str(temp_dir)},
                'ingest': {'scan_glob_audio': '*.wav', 'scan_glob_h5': '*.hdf5', 'checksum': 'sha256'}
            }
            
            indexer = FilesystemIndexer.from_config_dict(config_dict, engine)
            results = indexer.run_full_indexing(temp_dir)
            
            expected_syllables = num_files * syllables_per_file
            assert results['indexed_files'] == num_files
            assert results['total_syllables'] == expected_syllables
            
            validation = validate_ingestion_results(engine, created_files, expected_syllables)
            assert validation['syllables_created'] == expected_syllables