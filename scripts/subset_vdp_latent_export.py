#!/usr/bin/env python3
"""Materialize a byte-identical latent-export subset by explicit metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(source) != _sha256_file(destination):
            raise ValueError(f"existing subset member differs: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize_subset(
    source_root: Path, destination_root: Path, excluded_birds: set[str]
) -> dict[str, Any]:
    npz_paths = sorted(source_root.rglob("*.npz"))
    if not npz_paths:
        raise ValueError("source export contains no NPZ members")
    included = 0
    excluded = 0
    methods: dict[str, int] = {}
    seen_excluded: set[str] = set()
    for npz_path in npz_paths:
        json_path = npz_path.with_suffix(".json")
        if not json_path.is_file():
            raise ValueError(f"missing JSON sidecar for {npz_path}")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        bird_id = str(metadata.get("bird_id"))
        if bird_id in excluded_birds:
            seen_excluded.add(bird_id)
            excluded += 1
            continue
        rel = npz_path.relative_to(source_root)
        for source in (npz_path, json_path):
            destination = destination_root / rel.with_suffix(source.suffix)
            method = _link_or_copy(source, destination)
            methods[method] = methods.get(method, 0) + 1
        included += 1
    missing = excluded_birds - seen_excluded
    if missing:
        raise ValueError(f"excluded birds were absent from source: {sorted(missing)}")
    return {
        "source_members": len(npz_paths),
        "included_members": included,
        "excluded_members": excluded,
        "excluded_birds": sorted(excluded_birds),
        "materialization": dict(sorted(methods.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--exclude-bird", action="append", required=True)
    parser.add_argument("--summary-out", required=True, type=Path)
    args = parser.parse_args()
    summary = materialize_subset(
        args.source_root, args.destination_root, set(args.exclude_bird)
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
