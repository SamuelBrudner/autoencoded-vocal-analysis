from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

torch = pytest.importorskip("torch")

from ava.models.sequence_window_dataset import (
    FixedWindowSequenceDataset,
    get_sequence_window_data_loaders,
)


def _write_sine(path: Path, fs: int = 1000, duration: float = 0.5, freq: float = 10.0) -> None:
    t = np.arange(int(fs * duration), dtype=np.float32) / fs
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wavfile.write(path.as_posix(), fs, audio)


def _fake_get_spec(t1, t2, audio, p, fs=32000, target_times=None, **kwargs):
    center = 0.5 * (float(t1) + float(t2))
    spec = np.full(
        (p["num_freq_bins"], p["num_time_bins"]),
        center,
        dtype=np.float32,
    )
    return spec, True


def test_sequence_dataset_returns_ordered_windows(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "roi"
    audio_dir.mkdir()
    roi_dir.mkdir()

    _write_sine(audio_dir / "a.wav")
    np.savetxt(roi_dir / "a.txt", np.array([[0.0, 0.25]]))

    params = {
        "fs": 1000,
        "window_length": 0.1,
        "sequence_hop_length": 0.05,
        "num_time_bins": 4,
        "num_freq_bins": 3,
        "get_spec": _fake_get_spec,
    }
    dataset = FixedWindowSequenceDataset(
        [str(audio_dir / "a.wav")],
        [str(roi_dir / "a.txt")],
        params,
    )

    sample = dataset[0]

    assert sample["x"].shape == (4, 3, 4)
    assert sample["mask"].tolist() == [True, True, True, True]
    assert torch.all(torch.diff(sample["start_times"]) > 0)
    assert float(sample["x"][1, 0, 0]) > float(sample["x"][0, 0, 0])


def test_sequence_data_loader_pads_variable_lengths(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "roi"
    audio_dir.mkdir()
    roi_dir.mkdir()

    for name in ("a", "b"):
        _write_sine(audio_dir / f"{name}.wav")
    np.savetxt(roi_dir / "a.txt", np.array([[0.0, 0.25]]))
    np.savetxt(roi_dir / "b.txt", np.array([[0.0, 0.15]]))

    params = {
        "fs": 1000,
        "window_length": 0.1,
        "sequence_hop_length": 0.05,
        "num_time_bins": 4,
        "num_freq_bins": 3,
        "get_spec": _fake_get_spec,
    }
    partition = {
        "train": {
            "audio": [str(audio_dir / "a.wav"), str(audio_dir / "b.wav")],
            "rois": [str(roi_dir / "a.txt"), str(roi_dir / "b.txt")],
        },
        "test": {},
    }
    loaders = get_sequence_window_data_loaders(
        partition,
        params,
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(loaders["train"]))

    assert batch["x"].shape == (2, 4, 3, 4)
    assert batch["mask"].shape == (2, 4)
    assert batch["mask"].sum(dim=1).tolist() == [4, 2]
    assert torch.allclose(batch["x"][1, 2:], torch.zeros_like(batch["x"][1, 2:]))
