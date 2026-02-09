import numpy as np
import pytest
from scipy.io import wavfile


torch = pytest.importorskip("torch")

from ava.models.fixed_window_config import FixedWindowAugmentationConfig
from ava.models.window_vae_dataset import get_fixed_window_data_loaders, get_window_partition


def _write_sine(path, fs=1000, duration=0.5, freq=10.0, amplitude=0.5):
    t = np.arange(int(fs * duration), dtype=np.float32) / fs
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wavfile.write(path, fs, audio)


def _fake_get_spec(t1, t2, audio, p, fs=32000, target_times=None, **kwargs):
    spec = np.ones((p["num_freq_bins"], p["num_time_bins"]), dtype=np.float32)
    return spec, True


def test_fixed_window_data_loaders_smoke(tmp_path):
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "rois"
    audio_dir.mkdir()
    roi_dir.mkdir()

    for name in ("a", "b"):
        wav_path = audio_dir / f"{name}.wav"
        roi_path = roi_dir / f"{name}.txt"
        _write_sine(wav_path)
        np.savetxt(roi_path, np.array([[0.0, 0.4]]))

    partition = get_window_partition(
        [str(audio_dir)],
        [str(roi_dir)],
        split=0.5,
        shuffle=False,
    )
    params = {
        "window_length": 0.2,
        "num_time_bins": 4,
        "num_freq_bins": 3,
        "get_spec": _fake_get_spec,
    }
    loaders = get_fixed_window_data_loaders(
        partition,
        params,
        batch_size=2,
        num_workers=0,
    )

    assert set(loaders) == {"train", "test"}
    batch = next(iter(loaders["train"]))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[1:] == (params["num_freq_bins"], params["num_time_bins"])


def test_fixed_window_data_loaders_train_only_augmentations(tmp_path):
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "rois"
    audio_dir.mkdir()
    roi_dir.mkdir()

    for name in ("a", "b"):
        wav_path = audio_dir / f"{name}.wav"
        roi_path = roi_dir / f"{name}.txt"
        _write_sine(wav_path)
        np.savetxt(roi_path, np.array([[0.0, 0.4]]))

    partition = get_window_partition(
        [str(audio_dir)],
        [str(roi_dir)],
        split=0.5,
        shuffle=False,
    )
    params = {
        "window_length": 0.2,
        "num_time_bins": 4,
        "num_freq_bins": 3,
        "get_spec": _fake_get_spec,
    }
    augmentations = FixedWindowAugmentationConfig(
        enabled=True,
        amplitude_scale=(2.0, 2.0),
    )
    loaders = get_fixed_window_data_loaders(
        partition,
        params,
        batch_size=2,
        num_workers=0,
        augmentations=augmentations,
    )

    train_batch = next(iter(loaders["train"]))
    test_batch = next(iter(loaders["test"]))
    assert torch.allclose(train_batch, torch.full_like(train_batch, 2.0))
    assert torch.allclose(test_batch, torch.ones_like(test_batch))


def test_fixed_window_data_loaders_return_pair(tmp_path):
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "rois"
    audio_dir.mkdir()
    roi_dir.mkdir()

    for name in ("a", "b"):
        wav_path = audio_dir / f"{name}.wav"
        roi_path = roi_dir / f"{name}.txt"
        _write_sine(wav_path)
        np.savetxt(roi_path, np.array([[0.0, 0.4]]))

    partition = get_window_partition(
        [str(audio_dir)],
        [str(roi_dir)],
        split=0.5,
        shuffle=False,
    )
    params = {
        "window_length": 0.2,
        "num_time_bins": 4,
        "num_freq_bins": 3,
        "get_spec": _fake_get_spec,
    }
    loaders = get_fixed_window_data_loaders(
        partition,
        params,
        batch_size=2,
        num_workers=0,
        return_pair=True,
    )

    batch = next(iter(loaders["train"]))
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    assert torch.is_tensor(batch[0])
    assert torch.is_tensor(batch[1])
    assert batch[0].shape[1:] == (params["num_freq_bins"], params["num_time_bins"])
    assert batch[1].shape[1:] == (params["num_freq_bins"], params["num_time_bins"])
