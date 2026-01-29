import numpy as np
import h5py
import pytest

from ava.data.metadata import validate_projection_alignment, validate_spec_metadata


def _write_spec_file(path, length=3, include_metadata=True):
	with h5py.File(path, "w") as f:
		f.create_dataset("specs", data=np.zeros((length, 2, 2)))
		if include_metadata:
			f.create_dataset(
				"audio_filenames",
				data=np.array([f"file_{i}.wav" for i in range(length)]).astype("S"),
			)
			f.create_dataset("onsets", data=np.arange(length, dtype=float))
			f.create_dataset("offsets", data=np.arange(length, dtype=float) + 1.0)


def test_validate_spec_metadata_missing_fields(tmp_path):
	spec_path = tmp_path / "specs_missing.hdf5"
	_write_spec_file(spec_path, include_metadata=False)
	with pytest.raises(KeyError):
		validate_spec_metadata(str(spec_path), require_metadata=True)
	assert validate_spec_metadata(str(spec_path), require_metadata=False) == 3


def test_validate_projection_alignment_mismatch(tmp_path):
	spec_path = tmp_path / "specs.hdf5"
	proj_path = tmp_path / "projections.hdf5"
	_write_spec_file(spec_path, length=3, include_metadata=True)
	with h5py.File(proj_path, "w") as f:
		f.create_dataset("latent_means", data=np.zeros((2, 4)))
	with pytest.raises(ValueError):
		validate_projection_alignment(
			str(spec_path),
			str(proj_path),
			require_metadata=True,
			allow_missing_projection=False,
		)


def test_validate_projection_alignment_missing_ok(tmp_path):
	spec_path = tmp_path / "specs.hdf5"
	proj_path = tmp_path / "missing.hdf5"
	_write_spec_file(spec_path, length=2, include_metadata=True)
	assert validate_projection_alignment(
		str(spec_path),
		str(proj_path),
		require_metadata=True,
		allow_missing_projection=True,
	) == 2
	with pytest.raises(FileNotFoundError):
		validate_projection_alignment(
			str(spec_path),
			str(proj_path),
			require_metadata=True,
			allow_missing_projection=False,
		)
