import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("torch")

from ava.models.window_vae_dataset import FixedWindowDataset


def _write_silence(path, fs=1000, duration=1.0):
	audio = np.zeros(int(fs * duration), dtype=np.int16)
	wavfile.write(path, fs, audio)


def _write_sine(path, fs=1000, duration=1.0, freq=10.0, amplitude=0.5):
	t = np.arange(int(fs * duration), dtype=np.float32) / fs
	audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
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


def test_fixed_window_dataset_sampling_weights(tmp_path):
	a_wav = tmp_path / "a.wav"
	b_wav = tmp_path / "b.wav"
	_write_silence(a_wav)
	_write_silence(b_wav)

	a_roi = tmp_path / "a.txt"
	b_roi = tmp_path / "b.txt"
	np.savetxt(a_roi, np.array([[0.0, 0.6], [0.1, 0.3]]))
	np.savetxt(b_roi, np.array([[0.0, 0.2]]))

	dataset = FixedWindowDataset(
		audio_filenames=[str(a_wav), str(b_wav)],
		roi_filenames=[str(a_roi), str(b_roi)],
		p={
			"window_length": 0.1,
			"file_weight_cap": 0.3,
			"roi_weight_mode": "uniform",
		},
	)

	assert np.allclose(dataset.file_weights, np.array([0.6, 0.4]))
	assert np.allclose(dataset.roi_weights[0], np.array([0.5, 0.5]))


def test_fixed_window_dataset_rejects_low_energy_windows(tmp_path):
	quiet_wav = tmp_path / "quiet.wav"
	loud_wav = tmp_path / "loud.wav"
	_write_silence(quiet_wav)
	_write_sine(loud_wav, amplitude=0.5)

	quiet_roi = tmp_path / "quiet.txt"
	loud_roi = tmp_path / "loud.txt"
	np.savetxt(quiet_roi, np.array([[0.0, 0.25]]))
	np.savetxt(loud_roi, np.array([[0.0, 0.6]]))

	def _fake_get_spec(t1, t2, audio, p, fs=32000, target_times=None, **kwargs):
		spec = np.ones((p["num_freq_bins"], p["num_time_bins"]), dtype=np.float32)
		return spec, True

	dataset = FixedWindowDataset(
		audio_filenames=[str(quiet_wav), str(loud_wav)],
		roi_filenames=[str(quiet_roi), str(loud_roi)],
		p={
			"window_length": 0.2,
			"num_time_bins": 4,
			"num_freq_bins": 3,
			"get_spec": _fake_get_spec,
		},
		min_audio_energy=0.05,
	)

	_, file_index, _, _ = dataset.__getitem__(0, seed=0, return_seg_info=True)
	assert dataset.filenames[file_index] == str(loud_wav)
