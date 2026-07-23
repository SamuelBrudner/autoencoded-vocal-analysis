"""Validation for the public ``ava_latent_sequence_v1`` handoff contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np


SCHEMA_VERSION = "ava_latent_sequence_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LatentSequenceContractError(ValueError):
	"""Raised when an AVA latent-sequence pair violates the public contract."""


@dataclass(frozen=True)
class ValidatedLatentSequence:
	"""A validated latent-sequence pair without dtype coercion."""

	npz_path: Path
	metadata_path: Path
	start_times_sec: np.ndarray
	window_length_sec: np.ndarray
	hop_length_sec: np.ndarray
	mu: np.ndarray
	logvar: np.ndarray
	metadata: Mapping[str, Any]
	energy: Optional[np.ndarray] = None
	gating_weight: Optional[np.ndarray] = None


def _fail(message: str) -> None:
	raise LatentSequenceContractError(message)


def _require_exact_dtype(
	arrays: Mapping[str, np.ndarray],
	name: str,
	dtype: np.dtype,
) -> np.ndarray:
	if name not in arrays:
		_fail(f"Missing required NPZ array: {name}.")
	array = np.asarray(arrays[name])
	if array.dtype != np.dtype(dtype):
		_fail(
			f"{name} must have dtype {np.dtype(dtype).name}; "
			f"found {array.dtype.name}."
		)
	if not np.isfinite(array).all():
		_fail(f"{name} contains nonfinite values.")
	return array


def _validate_metadata(metadata: Any) -> Dict[str, Any]:
	if not isinstance(metadata, dict):
		_fail("The JSON sidecar must contain an object.")

	required = (
		"schema_version",
		"created_utc",
		"clip_id",
		"audio_path",
		"audio_sha256",
		"sample_rate_hz",
	)
	missing = [name for name in required if name not in metadata]
	if missing:
		_fail(f"Missing required metadata fields: {', '.join(missing)}.")

	if metadata["schema_version"] != SCHEMA_VERSION:
		_fail(
			f"schema_version must be {SCHEMA_VERSION!r}; "
			f"found {metadata['schema_version']!r}."
		)

	for name in ("clip_id", "audio_path"):
		value = metadata[name]
		if not isinstance(value, str) or not value.strip():
			_fail(f"{name} must be a nonempty string.")

	created_utc = metadata["created_utc"]
	if not isinstance(created_utc, str):
		_fail("created_utc must be an ISO-8601 UTC string.")
	try:
		parsed = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
	except ValueError:
		_fail("created_utc must be an ISO-8601 UTC string.")
	if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
		_fail("created_utc must include a UTC offset or trailing Z.")

	audio_sha256 = metadata["audio_sha256"]
	if audio_sha256 is not None and (
		not isinstance(audio_sha256, str) or _SHA256_RE.fullmatch(audio_sha256) is None
	):
		_fail("audio_sha256 must be null or a lowercase 64-character SHA-256.")

	sample_rate_hz = metadata["sample_rate_hz"]
	if sample_rate_hz is not None and (
		not isinstance(sample_rate_hz, int)
		or isinstance(sample_rate_hz, bool)
		or sample_rate_hz <= 0
	):
		_fail("sample_rate_hz must be null or a positive integer.")

	return metadata


def validate_latent_sequence_pair(
	npz_path: Union[str, Path],
	metadata_path: Optional[Union[str, Path]] = None,
) -> ValidatedLatentSequence:
	"""Load and validate one same-stem NPZ/JSON latent-sequence pair.

	The validator deliberately performs no dtype conversion. A producer or
	consumer that converts timestamps to float32 before validation is therefore
	nonconformant.
	"""

	npz_path = Path(npz_path)
	if npz_path.suffix != ".npz":
		_fail(f"Expected an .npz path; found {npz_path.name!r}.")
	if not npz_path.is_file():
		raise FileNotFoundError(npz_path)

	expected_metadata_path = npz_path.with_suffix(".json")
	metadata_path = (
		expected_metadata_path if metadata_path is None else Path(metadata_path)
	)
	if metadata_path != expected_metadata_path:
		_fail(
			"Latent metadata must be a same-directory, same-stem JSON sidecar "
			f"({expected_metadata_path.name})."
		)
	if not metadata_path.is_file():
		_fail(f"Missing required JSON sidecar: {metadata_path}.")

	try:
		with metadata_path.open("r", encoding="utf-8") as handle:
			metadata = _validate_metadata(json.load(handle))
	except json.JSONDecodeError as exc:
		raise LatentSequenceContractError(
			f"Invalid JSON sidecar {metadata_path}: {exc}."
		) from exc

	try:
		with np.load(npz_path, allow_pickle=False) as loaded:
			arrays = {name: loaded[name] for name in loaded.files}
	except (OSError, ValueError) as exc:
		raise LatentSequenceContractError(
			f"Unable to read latent NPZ {npz_path}: {exc}."
		) from exc

	start_times = _require_exact_dtype(arrays, "start_times_sec", np.float64)
	window_length = _require_exact_dtype(arrays, "window_length_sec", np.float64)
	hop_length = _require_exact_dtype(arrays, "hop_length_sec", np.float64)
	mu = _require_exact_dtype(arrays, "mu", np.float32)
	logvar = _require_exact_dtype(arrays, "logvar", np.float32)

	if mu.ndim != 2 or mu.shape[0] < 1 or mu.shape[1] < 1:
		_fail("mu must have shape [T, z_dim] with T >= 1 and z_dim >= 1.")
	if logvar.shape != mu.shape:
		_fail(f"logvar shape {logvar.shape} does not match mu shape {mu.shape}.")
	if start_times.shape != (mu.shape[0],):
		_fail(
			f"start_times_sec must have shape ({mu.shape[0]},); "
			f"found {start_times.shape}."
		)
	if start_times.size > 1 and not np.all(np.diff(start_times) > 0):
		_fail("start_times_sec must be strictly increasing.")

	for name, array in (
		("window_length_sec", window_length),
		("hop_length_sec", hop_length),
	):
		if array.shape != ():
			_fail(f"{name} must be a float64 scalar; found shape {array.shape}.")
		if float(array) <= 0:
			_fail(f"{name} must be greater than zero.")

	energy = None
	if "energy" in arrays:
		energy = _require_exact_dtype(arrays, "energy", np.float32)
		if energy.shape != (mu.shape[0],):
			_fail(f"energy must have shape ({mu.shape[0]},); found {energy.shape}.")

	gating_weight = None
	if "gating_weight" in arrays:
		gating_weight = _require_exact_dtype(
			arrays, "gating_weight", np.float32
		)
		if gating_weight.shape != (mu.shape[0],):
			_fail(
				f"gating_weight must have shape ({mu.shape[0]},); "
				f"found {gating_weight.shape}."
			)
		if np.any(gating_weight < 0) or np.any(gating_weight > 1):
			_fail("gating_weight values must lie in the closed interval [0, 1].")

	return ValidatedLatentSequence(
		npz_path=npz_path,
		metadata_path=metadata_path,
		start_times_sec=start_times,
		window_length_sec=window_length,
		hop_length_sec=hop_length,
		mu=mu,
		logvar=logvar,
		energy=energy,
		gating_weight=gating_weight,
		metadata=metadata,
	)
