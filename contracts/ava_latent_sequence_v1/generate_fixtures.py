"""Regenerate deterministic public conformance fixtures."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent / "fixtures"
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def _write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
		for name in sorted(arrays):
			buffer = io.BytesIO()
			np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
			info = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIME)
			info.compress_type = zipfile.ZIP_DEFLATED
			info.external_attr = 0o644 << 16
			archive.writestr(info, buffer.getvalue())


def _metadata(version: str = "ava_latent_sequence_v1") -> dict:
	return {
		"schema_version": version,
		"created_utc": "2026-01-01T00:00:00Z",
		"clip_id": "fixture_clip",
		"audio_path": "fixtures/fixture_clip.wav",
		"audio_sha256": "a" * 64,
		"sample_rate_hz": 32000,
	}


def _arrays() -> Dict[str, np.ndarray]:
	return {
		"start_times_sec": np.asarray([0.0, 0.01, 0.02], dtype=np.float64),
		"window_length_sec": np.asarray(0.03, dtype=np.float64),
		"hop_length_sec": np.asarray(0.01, dtype=np.float64),
		"mu": np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float32),
		"logvar": np.asarray(
			[[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0]], dtype=np.float32
		),
		"energy": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
		"gating_weight": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
	}


def _write_case(
	name: str,
	arrays: Dict[str, np.ndarray],
	metadata: Optional[dict],
) -> None:
	case_dir = ROOT / name
	_write_npz(case_dir / "fixture_clip.npz", arrays)
	if metadata is not None:
		(case_dir / "fixture_clip.json").write_text(
			json.dumps(metadata, indent=2, sort_keys=True) + "\n",
			encoding="utf-8",
		)


def main() -> None:
	valid = _arrays()
	_write_case("valid", valid, _metadata())
	_write_case("missing_sidecar", valid, None)
	_write_case("wrong_version", valid, _metadata("ava_latent_sequence_v0"))

	nonfinite = dict(valid)
	nonfinite["mu"] = valid["mu"].copy()
	nonfinite["mu"][1, 0] = np.nan
	_write_case("nonfinite", nonfinite, _metadata())

	nonmonotonic = dict(valid)
	nonmonotonic["start_times_sec"] = np.asarray(
		[0.0, 0.02, 0.01], dtype=np.float64
	)
	_write_case("nonmonotonic_times", nonmonotonic, _metadata())

	shape_error = dict(valid)
	shape_error["logvar"] = valid["logvar"][:2]
	_write_case("shape_error", shape_error, _metadata())

	invalid_gating = dict(valid)
	invalid_gating["gating_weight"] = np.asarray(
		[0.0, 1.01, 1.0], dtype=np.float32
	)
	_write_case("invalid_gating_weight", invalid_gating, _metadata())

	files = sorted(
		path
		for path in ROOT.rglob("*")
		if path.is_file() and path.name != "SHA256SUMS.json"
	)
	manifest = {
		path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
		for path in files
	}
	(ROOT / "SHA256SUMS.json").write_text(
		json.dumps(manifest, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)


if __name__ == "__main__":
	main()
