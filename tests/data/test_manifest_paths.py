from pathlib import Path

from ava.data.manifest_paths import resolve_manifest_entry_paths


def test_resolve_manifest_entry_paths_overrides_absolute_dirs(tmp_path: Path) -> None:
    entry = {
        "audio_dir_rel": "birdA/session1",
        "audio_dir": "/old/audio/birdA/session1",
        "roi_dir": "/old/roi/birdA/session1",
    }

    audio_root = tmp_path / "audio_root"
    roi_root = tmp_path / "roi_root"

    audio_dir, roi_dir = resolve_manifest_entry_paths(
        entry, audio_root=audio_root, roi_root=roi_root
    )

    assert audio_dir == (audio_root / "birdA/session1").as_posix()
    assert roi_dir == (roi_root / "birdA/session1").as_posix()


def test_resolve_manifest_entry_paths_falls_back_without_rel() -> None:
    entry = {
        "audio_dir": "/abs/audio",
        "roi_dir": "/abs/roi",
    }

    audio_dir, roi_dir = resolve_manifest_entry_paths(
        entry, audio_root=Path("/new/audio"), roi_root=Path("/new/roi")
    )

    assert audio_dir == "/abs/audio"
    assert roi_dir == "/abs/roi"


def test_resolve_manifest_entry_paths_dot_rel_resolves_to_root(tmp_path: Path) -> None:
    entry = {
        "audio_dir_rel": ".",
        "audio_dir": "/abs/audio",
        "roi_dir": "/abs/roi",
    }

    audio_root = tmp_path / "audio_root"
    roi_root = tmp_path / "roi_root"

    audio_dir, roi_dir = resolve_manifest_entry_paths(
        entry, audio_root=audio_root, roi_root=roi_root
    )

    assert audio_dir == audio_root.as_posix()
    assert roi_dir == roi_root.as_posix()

