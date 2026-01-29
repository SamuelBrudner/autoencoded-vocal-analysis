import numpy as np
import pytest
from scipy.io import wavfile

from ava.preprocessing.preprocess import read_onsets_offsets_from_file


def test_read_onsets_offsets_empty_file(tmp_path):
	seg_path = tmp_path / "segments.txt"
	seg_path.write_text("# header only\n")
	onsets, offsets = read_onsets_offsets_from_file(str(seg_path), {})
	assert onsets.size == 0
	assert offsets.size == 0


def test_get_shotgun_partition_skips_empty_roi(tmp_path):
	pytest.importorskip("torch")
	pytest.importorskip("affinewarp")
	from ava.models.shotgun_vae_dataset import get_shotgun_partition

	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()
	audio = np.zeros(8000, dtype=np.float32)
	wav_path = audio_dir / "sample.wav"
	wavfile.write(str(wav_path), 8000, audio)
	roi_path = roi_dir / "sample.txt"
	roi_path.write_text("# empty\n")
	partition = get_shotgun_partition(
		[str(audio_dir)],
		[str(roi_dir)],
		split=0.8,
		shuffle=False,
		exclude_empty_roi_files=True,
	)
	assert len(partition["train"]["audio"]) == 0
	assert len(partition["test"]["audio"]) == 0
