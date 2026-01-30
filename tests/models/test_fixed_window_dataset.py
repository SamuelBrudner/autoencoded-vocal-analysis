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


def _write_constant(path, fs=1000, duration=1.0, value=1.0):
	audio = np.full(int(fs * duration), value, dtype=np.float32)
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


def test_fixed_window_dataset_global_normalization(tmp_path):
	audio_wav = tmp_path / "a.wav"
	_write_constant(audio_wav, value=0.4)
	roi = tmp_path / "a.txt"
	np.savetxt(roi, np.array([[0.0, 0.5]]))

	def _fake_get_spec(t1, t2, audio, p, fs=32000, target_times=None, **kwargs):
		value = float(np.mean(audio))
		spec = np.full((p["num_freq_bins"], p["num_time_bins"]), value,
			dtype=np.float32)
		return spec, True

	dataset = FixedWindowDataset(
		audio_filenames=[str(audio_wav)],
		roi_filenames=[str(roi)],
		p={
			"window_length": 0.2,
			"num_time_bins": 4,
			"num_freq_bins": 3,
			"get_spec": _fake_get_spec,
			"normalization_mode": "global",
			"normalization_method": "mean_std",
			"normalization_num_samples": 1,
			"normalization_seed": 0,
		},
	)

	stats = dataset.normalization_stats
	assert stats["mode"] == "global"
	assert np.isclose(stats["center"], 0.4)

	spec = dataset[0]
	assert np.allclose(spec, 0.0)


def test_fixed_window_dataset_per_file_normalization(tmp_path):
	a_wav = tmp_path / "a.wav"
	b_wav = tmp_path / "b.wav"
	_write_constant(a_wav, value=0.1)
	_write_constant(b_wav, value=0.6)

	a_roi = tmp_path / "a.txt"
	b_roi = tmp_path / "b.txt"
	np.savetxt(a_roi, np.array([[0.0, 0.5]]))
	np.savetxt(b_roi, np.array([[0.0, 0.5]]))

	def _fake_get_spec(t1, t2, audio, p, fs=32000, target_times=None, **kwargs):
		value = float(np.mean(audio))
		spec = np.full((p["num_freq_bins"], p["num_time_bins"]), value,
			dtype=np.float32)
		return spec, True

	dataset = FixedWindowDataset(
		audio_filenames=[str(b_wav), str(a_wav)],
		roi_filenames=[str(b_roi), str(a_roi)],
		p={
			"window_length": 0.2,
			"num_time_bins": 3,
			"num_freq_bins": 2,
			"get_spec": _fake_get_spec,
			"normalization_mode": "per_file",
			"normalization_method": "mean_std",
			"normalization_num_samples": 1,
			"normalization_seed": 1,
		},
	)

	stats = dataset.normalization_stats
	expected = []
	for filename in dataset.filenames:
		if filename.endswith("a.wav"):
			expected.append(0.1)
		else:
			expected.append(0.6)
	assert np.allclose(stats["center"], np.array(expected))

	spec, _, _, _ = dataset.__getitem__(0, seed=0, return_seg_info=True)
	assert np.allclose(spec, 0.0)


def test_fixed_window_dataset_deterministic_sampling_logs(tmp_path):
	audio_wav = tmp_path / "a.wav"
	_write_constant(audio_wav, value=0.3)
	roi = tmp_path / "a.txt"
	np.savetxt(roi, np.array([[0.0, 0.6]]))

	def _fake_get_spec(t1, t2, audio, p, fs=32000, target_times=None, **kwargs):
		spec = np.ones((p["num_freq_bins"], p["num_time_bins"]), dtype=np.float32)
		return spec, True

	dataset = FixedWindowDataset(
		audio_filenames=[str(audio_wav)],
		roi_filenames=[str(roi)],
		p={
			"window_length": 0.2,
			"num_time_bins": 4,
			"num_freq_bins": 3,
			"get_spec": _fake_get_spec,
			"sampling_seed": 123,
			"log_window_indices": True,
		},
	)

	dataset.set_epoch(0)
	_, file_idx0, onset0, offset0 = dataset.__getitem__(
		0, return_seg_info=True
	)
	_, file_idx1, onset1, offset1 = dataset.__getitem__(
		0, return_seg_info=True
	)
	assert file_idx0 == file_idx1
	assert np.isclose(onset0, onset1)
	assert np.isclose(offset0, offset1)
	log_epoch0 = dataset.get_window_log(0)
	assert log_epoch0
	seed0 = log_epoch0[-1]["seed"]

	dataset.set_epoch(1)
	_, _, onset2, offset2 = dataset.__getitem__(0, return_seg_info=True)
	_, _, onset3, offset3 = dataset.__getitem__(0, return_seg_info=True)
	assert np.isclose(onset2, onset3)
	assert np.isclose(offset2, offset3)
	log_epoch1 = dataset.get_window_log(1)
	assert log_epoch1
	seed1 = log_epoch1[-1]["seed"]
	assert seed0 != seed1
