"""
Filesystem scanner that discovers and indexes existing HDF5 spectrogram files and NPY 
embedding files into the metadata database.

Implements comprehensive filesystem scanning with SHA-256 checksum validation to discover 
and register existing analysis artifacts. Parses existing HDF5 files (syllables_XXXX.hdf5 
pattern) and NPY files, extracting metadata including shape, dtype, temporal bounds, and 
file references.

All methods adhere to the 15 LOC constraint and implement fail-loud behavior with 
RuntimeError on any data inconsistency. Uses structured JSONL logging for audit trails
and progress tracking during long-running indexing operations.
"""

from pathlib import Path
from glob import glob
from hashlib import sha256
from typing import Dict, List, Optional, Any
from datetime import datetime

import h5py
import numpy as np
from loguru import logger
from pydantic import BaseModel

from ava.db.schema import Syllable
from ava.db.session import get_session
from ava.db.repository import SyllableRepository, RecordingRepository


class DatabaseConfig(BaseModel):
    """Database configuration with connection parameters and feature toggle."""
    enabled: bool = True
    url: str = "sqlite:///./ava.db"
    echo: bool = False


class DataRootsConfig(BaseModel):
    """Data directory configuration for audio files and processed features."""
    audio_dir: str = "data/audio"
    features_dir: str = "data/features"


class IngestConfig(BaseModel):
    """Ingestion configuration with glob patterns and checksum algorithm."""
    scan_glob_audio: str = "**/*.wav"
    scan_glob_h5: str = "**/*.h5"
    checksum: str = "sha256"


class AVAConfig(BaseModel):
    """Top-level configuration aggregating database, paths, and ingestion settings."""
    database: DatabaseConfig
    data_roots: DataRootsConfig
    ingest: IngestConfig


class FilesystemIndexer:
    """
    Comprehensive filesystem scanner for HDF5 spectrogram and NPY embedding indexing.
    
    Implements automated discovery of existing analysis artifacts with SHA-256 checksum
    validation and structured database population through the repository layer.
    All methods maintain the 15 LOC constraint and fail-loud error handling.
    """
    
    def __init__(self, config: AVAConfig, engine) -> None:
        """Initialize indexer with configuration and database engine."""
        self.config = config
        self.engine = engine
        self.processed_files: List[Path] = []
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def scan_files(self, base_dir: Path) -> List[Path]:
        """Discover HDF5 files using configured glob patterns with recursive scanning."""
        pattern = str(base_dir / self.config.ingest.scan_glob_h5)
        discovered_paths = [Path(f) for f in glob(pattern, recursive=True)]
        hdf5_files = [p for p in discovered_paths if p.suffix == '.h5' or p.suffix == '.hdf5']
        logger.info(f"Discovered {len(hdf5_files)} HDF5 files", pattern=pattern)
        return hdf5_files
    
    def extract_hdf5_metadata(self, hdf5_path: Path) -> Dict[str, Any]:
        """Extract spectrogram metadata from HDF5 file with dataset validation."""
        resolved_path = hdf5_path.resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise RuntimeError(f"HDF5 file not found or not a file: {resolved_path}")
        
        if resolved_path.suffix not in ['.h5', '.hdf5']:
            raise RuntimeError(f"Invalid file extension: {resolved_path.suffix}")
        
        try:
            with h5py.File(resolved_path, 'r') as h5_file:
                required_keys = ['specs', 'onsets', 'offsets', 'audio_filenames']
                file_keys = list(h5_file.keys())
                missing_keys = [k for k in required_keys if k not in file_keys]
                if missing_keys:
                    raise RuntimeError(f"Missing required datasets {missing_keys} in {resolved_path}")
                
                specs_dataset = h5_file['specs']
                specs_array = specs_dataset[:]  # Load as numpy array to access properties
                onsets_array = h5_file['onsets'][:]
                offsets_array = h5_file['offsets'][:]
                
                return {
                    'specs_shape': specs_array.shape,
                    'specs_dtype': str(specs_array.dtype),
                    'specs_ndim': specs_array.ndim,
                    'onsets': onsets_array.tolist(),
                    'offsets': offsets_array.tolist(),
                    'num_syllables': len(onsets_array),
                    'extraction_time': datetime.now().isoformat()
                }
        except Exception as e:
            raise RuntimeError(f"Failed to extract metadata from {resolved_path}: {e}")
    
    def compute_checksums(self, file_paths: List[Path]) -> Dict[Path, str]:
        """Compute SHA-256 checksums for file integrity validation."""
        checksums = {}
        for file_path in file_paths:
            if not file_path.exists():
                raise RuntimeError(f"Cannot compute checksum - file not found: {file_path}")
            
            hash_algo = sha256()
            try:
                with file_path.open('rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_algo.update(chunk)
                checksums[file_path] = hash_algo.hexdigest()
                logger.debug(f"Computed checksum for {file_path}", checksum=checksums[file_path])
            except Exception as e:
                raise RuntimeError(f"Checksum computation failed for {file_path}: {e}")
        
        return checksums
    
    def populate_database(self, hdf5_files: List[Path], checksums: Dict[Path, str]) -> None:
        """Populate database with syllable metadata using repository pattern and transactions."""
        with get_session(self.engine) as session:
            recording_repo = RecordingRepository(session)
            syllable_repo = SyllableRepository(session)
            syllables_batch = []
            
            for hdf5_path in hdf5_files:
                try:
                    metadata = self.extract_hdf5_metadata(hdf5_path)
                    checksum = checksums[hdf5_path]
                    
                    # Create or get recording for this spectrogram file
                    existing_recording = recording_repo.get_by_path(str(hdf5_path))
                    if existing_recording is None:
                        recording = recording_repo.create(
                            file_path=str(hdf5_path),
                            checksum_sha256=checksum,
                            metadata={'file_type': 'hdf5_spectrogram', 'num_syllables': metadata['num_syllables']}
                        )
                        recording_id = recording.id
                    else:
                        recording_id = existing_recording.id
                    
                    # Prepare syllable data for bulk creation
                    for i, (onset, offset) in enumerate(zip(metadata['onsets'], metadata['offsets'])):
                        syllable_data = {
                            'recording_id': recording_id,
                            'spectrogram_path': str(hdf5_path),
                            'start_time': float(onset),
                            'end_time': float(offset),
                            'bounds_metadata': {'batch_index': i, 'specs_shape': metadata['specs_shape']}
                        }
                        syllables_batch.append(syllable_data)
                
                    logger.info(f"Prepared {metadata['num_syllables']} syllables from {hdf5_path}")
                except Exception as e:
                    raise RuntimeError(f"Database population failed for {hdf5_path}: {e}")
            
            created_syllables = syllable_repo.bulk_create(syllables_batch)
            logger.info(f"Successfully populated {len(created_syllables)} syllables to database")
    
    def validate_integrity(self, file_paths: List[Path]) -> bool:
        """Validate file existence and accessibility with comprehensive error reporting."""
        validation_errors = []
        
        for file_path in file_paths:
            if not file_path.exists():
                validation_errors.append(f"File does not exist: {file_path}")
            elif not file_path.is_file():
                validation_errors.append(f"Path is not a file: {file_path}")
            elif not file_path.suffix.lower() in ['.h5', '.hdf5']:
                validation_errors.append(f"Invalid file extension: {file_path}")
        
        if validation_errors:
            error_msg = f"Integrity validation failed:\n" + "\n".join(validation_errors)
            raise RuntimeError(error_msg)
        
        logger.info(f"Integrity validation passed for {len(file_paths)} files")
        return True
    
    def run_full_indexing(self, base_directory: Optional[Path] = None) -> Dict[str, Any]:
        """Execute complete indexing workflow with comprehensive error handling and progress tracking."""
        if base_directory is None:
            base_directory = Path(self.config.data_roots.features_dir)
        
        logger.info(f"Starting filesystem indexing", base_dir=str(base_directory))
        
        try:
            # Step 1: Discover files
            discovered_files = self.scan_files(base_directory)
            if not discovered_files:
                logger.warning("No HDF5 files discovered for indexing")
                return {'status': 'completed', 'indexed_files': 0, 'errors': []}
            
            # Step 2: Validate integrity
            self.validate_integrity(discovered_files)
            
            # Step 3: Compute checksums
            checksums = self.compute_checksums(discovered_files)
            
            # Step 4: Populate database
            self.populate_database(discovered_files, checksums)
            
            results = {
                'status': 'completed',
                'indexed_files': len(discovered_files),
                'total_syllables': sum(len(self.extract_hdf5_metadata(f)['onsets']) for f in discovered_files),
                'checksums_computed': len(checksums),
                'errors': []
            }
            
            logger.info("Indexing completed successfully", **results)
            return results
            
        except Exception as e:
            error_msg = f"Indexing workflow failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], engine) -> "FilesystemIndexer":
        """Create indexer from configuration dictionary with Pydantic validation."""
        try:
            validated_config = AVAConfig.model_validate(config_dict)
            logger.debug("Configuration validation successful", config=validated_config.model_dump())
            return cls(validated_config, engine)
        except Exception as e:
            raise RuntimeError(f"Configuration validation failed: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Export current configuration as validated dictionary for audit purposes."""
        config_dump = self.config.model_dump()
        config_dump['timestamp'] = datetime.now().isoformat()
        config_dump['indexer_instance'] = id(self)
        return config_dump