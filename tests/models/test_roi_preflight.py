import numpy as np
import pytest

from ava.models.roi_preflight import assert_window_length_compatible


def test_assert_window_length_compatible_passes_with_valid_segments(tmp_path):
    roi_a = tmp_path / "a.txt"
    roi_b = tmp_path / "b.txt"
    np.savetxt(roi_a, np.array([[0.0, 0.05], [0.1, 0.16]]))
    np.savetxt(roi_b, np.array([[0.0, 0.03], [0.05, 0.08]]))

    stats = assert_window_length_compatible(
        [roi_a.as_posix(), roi_b.as_posix()],
        window_length=0.03,
    )

    assert stats["roi_segments_total"] == 4
    assert stats["roi_segments_compatible"] == 4
    assert stats["max_duration_sec"] >= 0.05


def test_assert_window_length_compatible_fails_when_no_segment_is_long_enough(tmp_path):
    roi = tmp_path / "short.txt"
    np.savetxt(roi, np.array([[0.0, 0.01], [0.02, 0.03]]))

    with pytest.raises(ValueError, match="window_length is incompatible"):
        assert_window_length_compatible([roi.as_posix()], window_length=0.02)
