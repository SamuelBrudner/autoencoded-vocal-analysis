import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("torch")

from ava.models.window_vae_dataset import FixedWindowDataset


def _write_silence(path, fs=1000, duration=1.0):
	audio = np.zeros(int(fs * duration), dtype=np.int16)
	wavfile.write(path, fs, audio)


def test_fixed_window_dataset_aligns_and_filters(tmp_path):
	a_wav = tmp_path / "a.wav"
	b_wav = tmp_path / "b.wav"
	_write_silence(b_wav)
	_write_silence(a_wav)

	a_roi = tmp_path / "a.txt"
	b_roi = tmp_path / "b.txt"
	np.savetxt(a_roi, np.array([[0.0, 0.2], [0.0, 0.5]]))
	np.savetxt(b_roi, np.array([[0.1, 0.5]]))

	dataset = FixedWindowDataset(
		audio_filenames=[str(b_wav), str(a_wav)],
		roi_filenames=[str(b_roi), str(a_roi)],
		p={"window_length": 0.3},
	)

	assert list(dataset.filenames) == [str(a_wav), str(b_wav)]
	assert list(dataset.roi_filenames) == [str(a_roi), str(b_roi)]
	assert dataset.rois[0].shape[0] == 1
	roi_lengths = np.diff(dataset.rois[0], axis=1).flatten()
	assert np.allclose(roi_lengths, np.array([0.5]))
