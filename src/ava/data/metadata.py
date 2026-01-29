"""
Metadata field definitions and alignment checks for HDF5 syllable data.
"""
from __future__ import annotations

from typing import Dict
import os

import h5py


CANONICAL_METADATA_FIELDS = ("audio_filenames", "onsets", "offsets")
CANONICAL_SPEC_FIELDS = ("specs",) + CANONICAL_METADATA_FIELDS


def _dataset_length(dataset: h5py.Dataset) -> int:
    """Return the number of rows in a dataset (first dimension)."""
    shape = dataset.shape
    if shape is None or len(shape) == 0:
        return 1
    return shape[0]


def _format_lengths(lengths: Dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(lengths.items()))


def _validate_equal_lengths(lengths: Dict[str, int], label: str) -> int:
    unique = set(lengths.values())
    if len(unique) != 1:
        detail = _format_lengths(lengths)
        raise ValueError(
            f"Found mismatched dataset lengths in {label}: {detail}"
        )
    return unique.pop()


def validate_spec_metadata(hdf5_path: str, require_metadata: bool = True) -> int:
    """
    Validate metadata alignment for a spec HDF5 file.

    Returns the number of syllables (length of the spec dimension).
    """
    required_fields = ["specs"]
    optional_fields = []
    if require_metadata:
        required_fields.extend(CANONICAL_METADATA_FIELDS)
    else:
        optional_fields.extend(CANONICAL_METADATA_FIELDS)

    with h5py.File(hdf5_path, "r") as f:
        missing = [field for field in required_fields if field not in f]
        if missing:
            raise KeyError(
                f"Missing required metadata fields in {hdf5_path}: {missing}"
            )
        lengths = {field: _dataset_length(f[field]) for field in required_fields}
        for field in optional_fields:
            if field in f:
                lengths[field] = _dataset_length(f[field])

    return _validate_equal_lengths(lengths, hdf5_path)


def validate_projection_alignment(
    spec_hdf5_path: str,
    projection_hdf5_path: str,
    require_metadata: bool = True,
    allow_missing_projection: bool = False,
) -> int:
    """
    Validate that projection/features datasets align with the spec file.

    Returns the number of syllables from the spec file.
    """
    spec_length = validate_spec_metadata(
        spec_hdf5_path,
        require_metadata=require_metadata,
    )
    if not os.path.exists(projection_hdf5_path):
        if allow_missing_projection:
            return spec_length
        raise FileNotFoundError(
            f"Missing projection file {projection_hdf5_path} for {spec_hdf5_path}"
        )

    with h5py.File(projection_hdf5_path, "r") as f:
        lengths = {
            key: _dataset_length(obj)
            for key, obj in f.items()
            if isinstance(obj, h5py.Dataset)
        }

    if lengths:
        mismatched = {key: value for key, value in lengths.items()
            if value != spec_length}
        if mismatched:
            detail = _format_lengths(mismatched)
            raise ValueError(
                "Projection metadata misaligned with specs in "
                f"{projection_hdf5_path}: {detail}"
            )

    return spec_length
