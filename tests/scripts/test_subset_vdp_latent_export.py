from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "subset_vdp_latent_export.py"
)
SPEC = importlib.util.spec_from_file_location("subset_vdp_latent_export", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_subset_excludes_explicit_bird_and_preserves_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    for bird in ("keep", "exclude"):
        directory = source / bird
        directory.mkdir(parents=True)
        (directory / "clip.npz").write_bytes(f"npz-{bird}".encode())
        (directory / "clip.json").write_text(
            json.dumps({"bird_id": bird}), encoding="utf-8"
        )

    summary = MODULE.materialize_subset(source, destination, {"exclude"})

    assert summary["included_members"] == 1
    assert summary["excluded_members"] == 1
    assert (destination / "keep" / "clip.npz").read_bytes() == b"npz-keep"
    assert not (destination / "exclude").exists()


def test_subset_rejects_absent_exclusion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "clip.npz").write_bytes(b"npz")
    (source / "clip.json").write_text(
        json.dumps({"bird_id": "present"}), encoding="utf-8"
    )
    try:
        MODULE.materialize_subset(source, tmp_path / "out", {"absent"})
    except ValueError as exc:
        assert "absent" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("absent excluded bird must be rejected")
