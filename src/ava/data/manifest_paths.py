"""Helpers for resolving absolute paths from birdsong manifest entries.

These helpers centralize the CLI override behavior for scripts that consume the
birdsong manifest (docs/birdsong_manifest.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


def _resolve_dir_from_root(root: Path, rel: str) -> str:
    rel = str(rel)
    if rel in (".", ""):
        return root.as_posix()
    return (root / rel).as_posix()


def resolve_manifest_entry_paths(
    entry: Mapping[str, Any],
    audio_root: Optional[Path] = None,
    roi_root: Optional[Path] = None,
) -> Tuple[str, str]:
    """Resolve (audio_dir, roi_dir) for a manifest entry.

    Override behavior:
    - If audio_root is provided AND entry has audio_dir_rel, audio_dir is
      resolved as <audio_root>/<audio_dir_rel> even if entry["audio_dir"] is set.
    - If roi_root is provided AND entry has audio_dir_rel, roi_dir is
      resolved as <roi_root>/<audio_dir_rel> even if entry["roi_dir"] is set.

    Fallback behavior:
    - If audio_dir_rel is missing, fall back to entry["audio_dir"]/entry["roi_dir"].
    - If a required path cannot be resolved, raise ValueError.
    """

    audio_dir_rel = entry.get("audio_dir_rel")
    audio_dir = entry.get("audio_dir")
    roi_dir = entry.get("roi_dir")

    if audio_root is not None and audio_dir_rel is not None:
        audio_dir = _resolve_dir_from_root(audio_root, str(audio_dir_rel))
    elif not audio_dir:
        if audio_dir_rel is None or audio_root is None:
            raise ValueError("Manifest entry missing audio_dir and audio_root.")
        audio_dir = _resolve_dir_from_root(audio_root, str(audio_dir_rel))
    else:
        audio_dir = str(audio_dir)

    if roi_root is not None and audio_dir_rel is not None:
        roi_dir = _resolve_dir_from_root(roi_root, str(audio_dir_rel))
    elif not roi_dir:
        if audio_dir_rel is None or roi_root is None:
            raise ValueError("Manifest entry missing roi_dir and roi_root.")
        roi_dir = _resolve_dir_from_root(roi_root, str(audio_dir_rel))
    else:
        roi_dir = str(roi_dir)

    return audio_dir, roi_dir

