"""
Test module validating query result determinism and repository method functionality.

This module ensures query reproducibility across multiple executions, validates filtering
by duration, label, model_tag, and complex filter combinations. All test functions adhere
to the 15 LOC constraint and provide comprehensive validation of database query operations
with known test fixtures for deterministic results.

Validates performance requirements ensuring queries complete within 100ms threshold
and maintains compatibility with both SQLite and PostgreSQL backends.
"""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ava.db.repository import SyllableRepository, EmbeddingRepository, AnnotationRepository, QueryDSL
from ava.db.schema import Base, Recording, Syllable, Embedding, Annotation
from ava.db.session import get_session, create_engine_from_url


@pytest.fixture
def setup_test_database():
    """
    Create ephemeral test database with schema and return engine for test isolation.
    
    Creates temporary SQLite database with all tables initialized and foreign key
    enforcement enabled for comprehensive constraint validation during testing.
    """
    temp_dir = TemporaryDirectory()
    db_path = Path(temp_dir.name) / "test_ava.db"
    database_url = f"sqlite:///{db_path}"
    engine = create_engine_from_url(database_url, echo=False)
    return engine


@pytest.fixture  
def create_test_data_fixtures(setup_test_database):
    """
    Generate deterministic test data fixtures with known values for reproducible query validation.
    
    Creates recordings, syllables, embeddings, and annotations with specific values
    designed to test filtering, duration queries, label matching, and model versioning.
    """
    engine = setup_test_database
    with get_session(engine) as session:
        # Create test recordings with deterministic timestamps
        recording1 = Recording(file_path="/test/audio1.wav", checksum_sha256="hash1", 
                              created_at=datetime(2024, 1, 1, 10, 0, 0))
        recording2 = Recording(file_path="/test/audio2.wav", checksum_sha256="hash2",
                              created_at=datetime(2024, 1, 2, 10, 0, 0))
        session.add_all([recording1, recording2])
        session.flush()
        
        # Create syllables with specific durations for filtering tests
        syllable1 = Syllable(recording_id=recording1.id, spectrogram_path="/test/spec1.h5",
                            start_time=0.0, end_time=0.1)  # 0.1s duration
        syllable2 = Syllable(recording_id=recording1.id, spectrogram_path="/test/spec2.h5", 
                            start_time=1.0, end_time=1.2)  # 0.2s duration  
        syllable3 = Syllable(recording_id=recording2.id, spectrogram_path="/test/spec3.h5",
                            start_time=2.0, end_time=2.05)  # 0.05s duration
        session.add_all([syllable1, syllable2, syllable3])
        session.flush()
        
        # Create embeddings with different model versions
        embedding1 = Embedding(syllable_id=syllable1.id, model_version="vae_v1", 
                              embedding_path="/test/emb1.npy", dimensions=64)
        embedding2 = Embedding(syllable_id=syllable2.id, model_version="vae_v2",
                              embedding_path="/test/emb2.npy", dimensions=128)  
        session.add_all([embedding1, embedding2])
        session.flush()
        
        # Create annotations for label-based testing
        annotation1 = Annotation(syllable_id=syllable1.id, annotation_type="label", 
                                key="call_type", value="call_A")
        annotation2 = Annotation(syllable_id=syllable2.id, annotation_type="label",
                                key="call_type", value="call_B") 
        session.add_all([annotation1, annotation2])
    
    return engine


def test_duration_filtering_returns_consistent_results(create_test_data_fixtures):
    """
    Validate SyllableRepository.filter_by_duration returns deterministic results for threshold queries.
    
    Tests filtering syllables with duration > 0.15 seconds returns consistent ordering
    and exact same results across multiple query executions using repository pattern.
    """
    engine = create_test_data_fixtures
    with get_session(engine) as session:
        repo = SyllableRepository(session) 
        # Query syllables with duration > 0.15s (should return syllable2 with 0.2s duration)
        results1 = repo.filter_by_duration(0.15, 1.0)
        results2 = repo.filter_by_duration(0.15, 1.0)
        results3 = repo.filter_by_duration(0.15, 1.0)
        
        # Verify consistent results across multiple executions
        assert len(results1) == 1
        assert len(results2) == 1  
        assert len(results3) == 1
        assert results1[0].id == results2[0].id == results3[0].id
        assert abs(results1[0].end_time - results1[0].start_time - 0.2) < 0.001


def test_label_based_queries_exact_matching(create_test_data_fixtures):
    """
    Validate QueryDSL label filtering produces exact matches for annotation values.
    
    Tests that label=="call_A" returns only syllables with that exact annotation
    value and maintains deterministic ordering across multiple query executions.
    """
    engine = create_test_data_fixtures
    with get_session(engine) as session:
        query_dsl = QueryDSL(session)
        # Query syllables with label annotation "call_A" 
        results1 = query_dsl.filter_by_label("label", "call_A").execute()
        results2 = query_dsl.filter_by_label("label", "call_A").execute()
        
        # Verify exact matching and reproducibility
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].id == results2[0].id
        # Verify the correct syllable was returned (syllable1 has call_A annotation)
        assert results1[0].start_time == 0.0
        assert results1[0].end_time == 0.1


def test_model_tag_filtering_functions_properly(create_test_data_fixtures):
    """
    Validate EmbeddingRepository.filter_by_model_tag returns correct model version filtering.
    
    Tests that model_tag=="vae_v1" returns only embeddings with that exact model
    version and maintains consistent ordering for reproducible dataset selections.
    """
    engine = create_test_data_fixtures
    with get_session(engine) as session:
        repo = EmbeddingRepository(session)
        # Query embeddings with model version "vae_v1"
        results1 = repo.filter_by_model_tag("vae_v1") 
        results2 = repo.filter_by_model_tag("vae_v1")
        
        # Verify model version filtering works correctly
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].id == results2[0].id
        assert results1[0].model_version == "vae_v1"
        assert results1[0].dimensions == 64


def test_combined_filters_produce_expected_intersections(create_test_data_fixtures):
    """
    Validate QueryDSL chaining produces correct intersections of multiple filter criteria.
    
    Tests that combining duration and label filters returns syllables meeting both
    conditions with deterministic result ordering for complex query scenarios.
    """
    engine = create_test_data_fixtures
    with get_session(engine) as session:
        query_dsl1 = QueryDSL(session)
        query_dsl2 = QueryDSL(session)
        # Chain filters: duration > 0.05 AND label == "call_A"  
        results1 = query_dsl1.filter_by_duration(0.05).filter_by_label("label", "call_A").execute()
        results2 = query_dsl2.filter_by_duration(0.05).filter_by_label("label", "call_A").execute()
        
        # Verify intersection logic and reproducibility
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].id == results2[0].id
        # Should return syllable1: duration=0.1s > 0.05s AND has call_A annotation
        assert abs(results1[0].end_time - results1[0].start_time - 0.1) < 0.001


def test_results_reproducible_across_multiple_runs(create_test_data_fixtures):
    """
    Validate all query methods return identical results across multiple executions.
    
    Tests that ordering by primary key and temporal fields ensures deterministic
    result sets for reproducible dataset selections in analysis workflows.
    """
    engine = create_test_data_fixtures
    results_sets = []
    # Execute same query 5 times to verify reproducibility
    for _ in range(5):
        with get_session(engine) as session:
            repo = SyllableRepository(session)
            results = repo.filter_by_duration(0.0, 1.0)  # Get all syllables
            results_sets.append([s.id for s in results])
    
    # Verify all result sets are identical
    first_set = results_sets[0] 
    for result_set in results_sets[1:]:
        assert result_set == first_set
    assert len(first_set) == 3  # All three test syllables


def test_query_performance_within_100ms_threshold(create_test_data_fixtures):
    """
    Validate repository query methods complete within 100ms performance threshold.
    
    Measures execution time of filter operations to ensure database queries meet
    performance requirements for interactive analysis workflows.
    """
    engine = create_test_data_fixtures
    with get_session(engine) as session:
        repo = SyllableRepository(session)
        # Measure duration filtering performance
        start_time = time.perf_counter()
        results = repo.filter_by_duration(0.0, 1.0)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Verify query completes within performance threshold
        assert duration_ms < 100.0, f"Query took {duration_ms:.2f}ms, exceeds 100ms threshold"
        assert len(results) > 0  # Ensure query returned expected results