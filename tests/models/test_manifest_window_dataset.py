from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from ava.models.manifest_window_dataset import (
    ManifestFixedWindowDataset,
    get_manifest_fixed_window_data_loaders,
)
from ava.preprocessing.utils import get_spec


def _write_roi_parquet(path: Path) -> None:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    table = pa.table(
        {
            "clip_stem": ["sample"],
            "onsets_sec": [[0.2]],
            "offsets_sec": [[0.5]],
        }
    )
    pq.write_table(table, path.as_posix())


def _make_params(fs: int, *, sampling_seed: int | None = None) -> dict:
    params = {
        "fs": fs,
        "get_spec": get_spec,
        "num_freq_bins": 64,
        "num_time_bins": 64,
        "nperseg": 256,
        "noverlap": 128,
        "max_dur": 1e9,
        "window_length": 0.12,
        "min_freq": 300.0,
        "max_freq": 10000.0,
        "spec_min_val": 0.0,
        "spec_max_val": 10.0,
        "mel": False,
        "time_stretch": False,
        "within_syll_normalize": False,
        "normalization_mode": "none",
        "roi_weight_mode": "uniform",
    }
    if sampling_seed is not None:
        params["sampling_seed"] = int(sampling_seed)
    return params


def _write_chirp_wav(path: Path, fs: int, duration_sec: float = 3.0) -> None:
    t = np.arange(int(fs * duration_sec), dtype=np.float32) / float(fs)
    phase = 2.0 * np.pi * (350.0 * t + 900.0 * np.square(t))
    audio = (0.6 * np.sin(phase) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(path.as_posix(), fs, audio)


def test_manifest_dataset_parquet_is_lazy_about_roi(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "roi"
    audio_dir.mkdir()
    roi_dir.mkdir()

    fs = 44100
    audio = np.zeros(fs, dtype=np.int16)
    wavfile.write((audio_dir / "sample.wav").as_posix(), fs, audio)

    entries = [
        {
            "audio_dir": audio_dir.as_posix(),
            "roi_dir": roi_dir.as_posix(),
            "audio_dir_rel": ".",
            "num_files": 1,
        }
    ]

    p = _make_params(fs)

    dataset = ManifestFixedWindowDataset(
        entries,
        p,
        roi_format="parquet",
        roi_parquet_name="roi.parquet",
        dataset_length=8,
        roi_cache_size=2,
    )

    # No roi.parquet exists yet; dataset should construct successfully and only
    # fail when sampling.
    with pytest.raises(ValueError):
        _ = dataset[0]

    _write_roi_parquet(roi_dir / "roi.parquet")

    sample = dataset[0]
    assert hasattr(sample, "shape")
    assert tuple(sample.shape) == (p["num_freq_bins"], p["num_time_bins"])

    sampled_batch = dataset[np.array([0, 1])]
    assert isinstance(sampled_batch, list)
    assert len(sampled_batch) == 2
    assert all(tuple(spec.shape) == (p["num_freq_bins"], p["num_time_bins"]) for spec in sampled_batch)


def test_manifest_dataset_set_epoch_changes_deterministic_windows(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "roi"
    audio_dir.mkdir()
    roi_dir.mkdir()

    fs = 44100
    _write_chirp_wav(audio_dir / "sample.wav", fs)
    _write_roi_parquet(roi_dir / "roi.parquet")

    entries = [
        {
            "audio_dir": audio_dir.as_posix(),
            "roi_dir": roi_dir.as_posix(),
            "audio_dir_rel": ".",
            "num_files": 1,
        }
    ]
    params = _make_params(fs, sampling_seed=17)
    dataset = ManifestFixedWindowDataset(
        entries,
        params,
        roi_format="parquet",
        roi_parquet_name="roi.parquet",
        dataset_length=4,
        roi_cache_size=1,
    )

    dataset.set_epoch(0)
    epoch0 = dataset[0].detach().cpu().numpy()
    dataset.set_epoch(1)
    epoch1 = dataset[0].detach().cpu().numpy()
    dataset.set_epoch(0)
    epoch0_repeat = dataset[0].detach().cpu().numpy()

    assert not np.allclose(epoch0, epoch1)
    assert np.allclose(epoch0, epoch0_repeat)


def test_manifest_data_loaders_accept_split_dataset_lengths(tmp_path: Path) -> None:
    train_audio_dir = tmp_path / "train_audio"
    train_roi_dir = tmp_path / "train_roi"
    test_audio_dir = tmp_path / "test_audio"
    test_roi_dir = tmp_path / "test_roi"
    train_audio_dir.mkdir()
    train_roi_dir.mkdir()
    test_audio_dir.mkdir()
    test_roi_dir.mkdir()

    fs = 44100
    _write_chirp_wav(train_audio_dir / "sample.wav", fs)
    _write_chirp_wav(test_audio_dir / "sample.wav", fs)
    _write_roi_parquet(train_roi_dir / "roi.parquet")
    _write_roi_parquet(test_roi_dir / "roi.parquet")

    train_entries = [
        {
            "audio_dir": train_audio_dir.as_posix(),
            "roi_dir": train_roi_dir.as_posix(),
            "audio_dir_rel": ".",
            "num_files": 1,
        }
    ]
    test_entries = [
        {
            "audio_dir": test_audio_dir.as_posix(),
            "roi_dir": test_roi_dir.as_posix(),
            "audio_dir_rel": ".",
            "num_files": 1,
        }
    ]

    loaders = get_manifest_fixed_window_data_loaders(
        train_entries,
        test_entries,
        _make_params(fs, sampling_seed=3),
        roi_format="parquet",
        roi_parquet_name="roi.parquet",
        dataset_length=8,
        train_dataset_length=5,
        test_dataset_length=2,
        batch_size=2,
        num_workers=0,
    )

    assert len(loaders["train"].dataset) == 5
    assert len(loaders["test"].dataset) == 2
