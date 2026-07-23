from pathlib import Path

import numpy as np
import pytest

from ava.models.latent_sequence_contract import (
	LatentSequenceContractError,
	validate_latent_sequence_pair,
)


FIXTURES = (
	Path(__file__).resolve().parents[2]
	/ "contracts"
	/ "ava_latent_sequence_v1"
	/ "fixtures"
)


def test_public_valid_fixture_preserves_contract_dtypes():
	artifact = validate_latent_sequence_pair(FIXTURES / "valid" / "fixture_clip.npz")

	assert artifact.mu.dtype == np.float32
	assert artifact.logvar.dtype == np.float32
	assert artifact.start_times_sec.dtype == np.float64
	assert artifact.window_length_sec.shape == ()
	assert artifact.hop_length_sec.shape == ()
	assert artifact.gating_weight is not None


@pytest.mark.parametrize(
	"case, message",
	[
		("missing_sidecar", "Missing required JSON sidecar"),
		("wrong_version", "schema_version must be"),
		("nonfinite", "mu contains nonfinite"),
		("nonmonotonic_times", "strictly increasing"),
		("shape_error", "does not match mu shape"),
		("invalid_gating_weight", "closed interval"),
	],
)
def test_public_invalid_fixtures_are_rejected(case, message):
	with pytest.raises(LatentSequenceContractError, match=message):
		validate_latent_sequence_pair(FIXTURES / case / "fixture_clip.npz")


def test_rejects_explicit_non_same_stem_sidecar(tmp_path):
	npz_path = FIXTURES / "valid" / "fixture_clip.npz"
	other_sidecar = tmp_path / "other.json"
	other_sidecar.write_text("{}", encoding="utf-8")

	with pytest.raises(LatentSequenceContractError, match="same-stem"):
		validate_latent_sequence_pair(npz_path, other_sidecar)
