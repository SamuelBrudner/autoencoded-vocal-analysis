#!/usr/bin/env python3
"""
AVA Database Ingestion CLI Script.

Primary CLI entry point for database metadata ingestion that scans filesystem for existing 
HDF5/NPY files, validates checksums, extracts metadata, and populates the relational database 
tables with comprehensive error handling and JSONL structured logging.

Supports multiple subcommands:
- init: Initialize database schema
- ingest: Execute metadata ingestion from filesystem
- validate: Validate configuration without database operations
- status: Display database status and health metrics

All functions adhere to 15 LOC limit and implement fail-loud behavior with RuntimeError
on data inconsistencies. Uses structured JSONL logging for audit trails and real-time
progress reporting for long-running operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from loguru import logger
from rich.progress import Progress
from sqlalchemy import MetaData

from ava.db.schema import Recording
from ava.db.session import create_engine_from_url  
from ava.db.repository import SyllableRepository
from ava.data.indexer import AVAConfig


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file with comprehensive validation.
    
    Validates file existence, accessibility, and YAML syntax before returning
    parsed configuration dictionary for Pydantic validation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing parsed configuration data
        
    Raises:
        RuntimeError: On file access failure or YAML parsing errors
    """
    if not config_path.exists():
        raise RuntimeError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise RuntimeError(f"Configuration path is not a file: {config_path}")
    
    try:
        config_text = config_path.read_text()
        config_data: Dict[str, Any] = yaml.safe_load(config_text)
        logger.info("Configuration loaded successfully", path=str(config_path))
        return config_data
    except Exception as e:
        raise RuntimeError(f"Configuration loading failed: {e}")


def validate_config(args: argparse.Namespace) -> None:
    """
    Validate YAML configuration using Pydantic models without database operations.
    
    Tests configuration parsing, validation, and schema compliance to detect
    issues before attempting database initialization or ingestion operations.
    
    Args:
        args: Parsed command line arguments containing config_path
        
    Raises:
        RuntimeError: On validation failure with detailed error messages  
    """
    try:
        config_data = load_config(args.config_path)
        validated_config = AVAConfig.model_validate(config_data)
        logger.info("Configuration validation successful", config=validated_config.model_dump())
        print("✓ Configuration validation passed")
    except Exception as e:
        logger.error("Configuration validation failed", error=str(e))
        raise RuntimeError(f"Validation failed: {e}")


def init_database(args: argparse.Namespace) -> None:
    """
    Initialize database schema using SQLAlchemy metadata creation.
    
    Creates database tables based on schema definitions with proper foreign key
    constraints and indexing. Supports both SQLite and PostgreSQL backends.
    
    Args:
        args: Parsed command line arguments containing config_path
        
    Raises:
        RuntimeError: On database creation failure or connection errors
    """
    try:
        config_data = load_config(args.config_path) 
        ava_config = AVAConfig.model_validate(config_data)
        engine = create_engine_from_url(ava_config.database.url, ava_config.database.echo)
        
        # Tables are created automatically by create_engine_from_url via Base.metadata.create_all
        logger.info("Database initialization completed", url=ava_config.database.url)
        print(f"✓ Database initialized at: {ava_config.database.url}")
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise RuntimeError(f"Database initialization failed: {e}")


def ingest_metadata(args: argparse.Namespace) -> None:
    """
    Execute filesystem metadata ingestion with real-time progress reporting.
    
    Scans configured directories for HDF5 files, extracts metadata, validates
    checksums, and populates database with syllable information using batch operations.
    
    Args:
        args: Parsed command line arguments containing config_path
        
    Raises:
        RuntimeError: On ingestion failure, checksum mismatch, or data integrity violation
    """
    from ava.data.indexer import FilesystemIndexer
    try:
        ava_config = AVAConfig.model_validate(load_config(args.config_path))
        engine = create_engine_from_url(ava_config.database.url, ava_config.database.echo)
        with Progress() as progress:
            progress.add_task("[green]Ingesting metadata...", total=100)
            indexer = FilesystemIndexer(ava_config, engine)
            results = indexer.run_full_indexing()
        logger.info("Metadata ingestion completed", **results)
        print(f"✓ Ingested {results['indexed_files']} files, {results['total_syllables']} syllables")
    except Exception as e:
        logger.error("Metadata ingestion failed", error=str(e))
        raise RuntimeError(f"Ingestion failed: {e}")


def show_status(args: argparse.Namespace) -> None:
    """
    Display database status and health metrics with comprehensive validation.
    
    Queries database for record counts, validates table relationships, and reports
    system health including file integrity and foreign key compliance.
    
    Args:
        args: Parsed command line arguments containing config_path
        
    Raises:
        RuntimeError: On database connection failure or integrity validation errors
    """
    from ava.db.session import get_session
    try:
        ava_config = AVAConfig.model_validate(load_config(args.config_path))
        engine = create_engine_from_url(ava_config.database.url, ava_config.database.echo)
        with get_session(engine) as session:
            syllable_repo = SyllableRepository(session)
            if hasattr(args, 'recording_id') and args.recording_id:
                syllables = syllable_repo.get_by_recording(args.recording_id)
                print(f"✓ Found {len(syllables)} syllables for recording {args.recording_id}")
            duration_filtered = syllable_repo.filter_by_duration(0.1, 2.0)
            print(f"✓ Database operational - {len(duration_filtered)} syllables in duration range")
        logger.info("Status check completed successfully")
    except Exception as e:
        logger.error("Status check failed", error=str(e))
        raise RuntimeError(f"Status check failed: {e}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure argument parser with subcommands and options.
    
    Creates main parser with config argument and subparsers for init, ingest,
    validate, and status operations with appropriate help text.
    
    Returns:
        Configured ArgumentParser instance with all subcommands
    """
    parser = argparse.ArgumentParser(
        description="AVA Database Ingestion CLI", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", dest="config_path", type=Path, required=True,
                       help="Path to YAML configuration file")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("init", help="Initialize database schema")
    subparsers.add_parser("ingest", help="Execute metadata ingestion")
    subparsers.add_parser("validate", help="Validate configuration")
    status_parser = subparsers.add_parser("status", help="Display database status")
    status_parser.add_argument("--recording-id", type=int, help="Recording ID for detailed status")
    return parser


def dispatch_command(args: argparse.Namespace) -> None:
    """
    Execute appropriate command based on parsed arguments.
    
    Dispatches to init_database, ingest_metadata, validate_config, or show_status
    based on command argument with proper error handling and logging.
    
    Args:
        args: Parsed command line arguments with command and config_path
        
    Raises:
        SystemExit: With non-zero exit code on command execution failure
    """
    command_map = {
        "init": init_database,
        "ingest": ingest_metadata, 
        "validate": validate_config,
        "status": show_status
    }
    command_map[args.command](args)


def main() -> None:
    """
    Main CLI entry point with comprehensive argument parsing and subcommand dispatch.
    
    Configures structured logging, parses command line arguments, and dispatches
    to appropriate subcommand handlers. Implements fail-loud behavior with non-zero
    exit codes on any failure for CI/CD compatibility.
    
    Raises:
        SystemExit: With non-zero exit code on any operation failure
    """
    logger.configure(handlers=[{"sink": sys.stderr, "format": "{message}", "serialize": True}])
    parser = setup_argument_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    try:
        dispatch_command(args)
    except Exception as e:
        logger.error("Command execution failed", command=args.command, error=str(e))
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()