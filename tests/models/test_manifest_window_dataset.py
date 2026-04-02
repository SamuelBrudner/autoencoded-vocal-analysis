from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from ava.models.manifest_window_dataset import ManifestFixedWindowDataset
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

    p = {
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
