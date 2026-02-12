from __future__ import annotations

from pathlib import Path

import pytest

from ava.models.roi_preflight import (
    assert_window_length_compatible_parquet_sample,
    sample_parquet_roi_duration_stats,
)


def test_parquet_sample_preflight_stats_and_guardrail(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    roi_path = tmp_path / "roi.parquet"
    table = pa.Table.from_pydict(
        {
            "clip_stem": ["a", "b"],
            "onsets_sec": [[0.0, 1.0], [0.0]],
            "offsets_sec": [[0.5, 1.4], [0.2]],
        }
    )
    pq.write_table(table, roi_path.as_posix())

    stats = sample_parquet_roi_duration_stats(
        [roi_path.as_posix()],
        window_length=0.25,
        sample_dirs=1,
        sample_segments=10,
        seed=0,
    )
    assert stats["roi_segments_sampled"] > 0
    assert stats["roi_segments_compatible"] > 0

    with pytest.raises(ValueError):
        assert_window_length_compatible_parquet_sample(
            [roi_path.as_posix()],
            window_length=10.0,
            sample_dirs=1,
            sample_segments=10,
            seed=0,
        )

