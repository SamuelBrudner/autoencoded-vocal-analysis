#!/usr/bin/env python3
"""Build a deterministic single-bird birdsong manifest from an existing manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_bird_id(value: object) -> str:
    if value is None:
        return ""
    return "".join(str(value).split()).upper()


def _coerce_dph(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        dph = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(dph):
        return None
    return int(dph)


def _resolve_dir(root: Path, rel: str) -> str:
    if rel in (".", ""):
        return root.as_posix()
    return (root / rel).as_posix()


def _summarize(entries: list[dict]) -> dict:
    by_regime = defaultdict(lambda: {"birds": set(), "directories": 0, "files": 0})
    total_files = 0
    for entry in entries:
        regime = entry.get("regime", "unknown")
        bird = entry.get("bird_id_norm")
        by_regime[regime]["directories"] += 1
        by_regime[regime]["files"] += int(entry.get("num_files", 0))
        total_files += int(entry.get("num_files", 0))
        if bird:
            by_regime[regime]["birds"].add(str(bird))

    by_regime_out = {}
    for regime, stats in by_regime.items():
        by_regime_out[regime] = {
            "birds": len(stats["birds"]),
            "directories": stats["directories"],
            "files": stats["files"],
        }

    return {
        "birds": len(
            {entry.get("bird_id_norm") for entry in entries if entry.get("bird_id_norm")}
        ),
        "directories": len(entries),
        "files": total_files,
        "by_regime": by_regime_out,
    }


def build_single_bird_manifest(
    source_manifest: dict,
    *,
    bird_id: str,
    min_dph: int,
    max_dph: int,
    test_every_n: int,
    test_offset: int,
    audio_root: Optional[Path] = None,
    roi_root: Optional[Path] = None,
) -> dict:
    if test_every_n <= 0:
        raise ValueError("--test-every-n must be positive.")
    if test_offset < 0 or test_offset >= test_every_n:
        raise ValueError("--test-offset must be in [0, --test-every-n).")
    if min_dph > max_dph:
        raise ValueError("--min-dph must be <= --max-dph.")

    target_bird = _normalize_bird_id(bird_id)
    if not target_bird:
        raise ValueError("--bird-id must resolve to a non-empty bird ID.")

    all_entries = list(source_manifest.get("train", [])) + list(source_manifest.get("test", []))
    candidates = []
    for entry in all_entries:
        entry_bird = _normalize_bird_id(
            entry.get("bird_id_norm") or entry.get("bird_id_raw")
        )
        if entry_bird != target_bird:
            continue

        dph = _coerce_dph(entry.get("dph"))
        if dph is None or dph < int(min_dph) or dph > int(max_dph):
            continue

        payload = dict(entry)
        payload["bird_id_norm"] = target_bird
        payload["_dph"] = dph
        candidates.append(payload)

    if not candidates:
        raise ValueError(
            f"No entries found for bird '{target_bird}' in DPH range "
            f"[{min_dph}, {max_dph}]."
        )

    candidates.sort(key=lambda entry: (entry["_dph"], str(entry.get("audio_dir_rel", ""))))

    root = audio_root or Path(str(source_manifest.get("root", ".")))
    roi = roi_root or Path(str(source_manifest.get("roi_root", ".")))

    train_entries: list[dict] = []
    test_entries: list[dict] = []
    for index, entry in enumerate(candidates):
        split = "test" if (index % test_every_n) == test_offset else "train"
        payload = {key: value for key, value in entry.items() if key != "_dph"}
        payload["split"] = split
        rel = str(payload.get("audio_dir_rel", "."))
        payload["audio_dir"] = _resolve_dir(root, rel)
        payload["roi_dir"] = _resolve_dir(roi, rel)
        if split == "train":
            train_entries.append(payload)
        else:
            test_entries.append(payload)

    if not train_entries:
        raise ValueError("The requested selection produced no train entries.")
    if not test_entries:
        raise ValueError("The requested selection produced no test entries.")

    total_entries = len(train_entries) + len(test_entries)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest_path": str(source_manifest.get("source_manifest_path") or ""),
        "metadata_path": source_manifest.get("metadata_path"),
        "root": root.as_posix(),
        "roi_root": roi.as_posix(),
        "seed": source_manifest.get("seed", 0),
        "train_fraction": len(train_entries) / float(total_entries),
        "filters": {
            "bird_id_norm": target_bird,
            "min_dph": int(min_dph),
            "max_dph": int(max_dph),
            "test_every_n": int(test_every_n),
            "test_offset": int(test_offset),
            "split_strategy": "deterministic_directory_stride",
        },
        "summary": {
            "train": _summarize(train_entries),
            "test": _summarize(test_entries),
        },
        "train": train_entries,
        "test": test_entries,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic single-bird birdsong manifest."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--bird-id", type=str, required=True)
    parser.add_argument("--min-dph", type=int, required=True)
    parser.add_argument("--max-dph", type=int, required=True)
    parser.add_argument("--test-every-n", type=int, required=True)
    parser.add_argument("--test-offset", type=int, required=True)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    source_manifest = _load_manifest(args.source_manifest)
    source_manifest["source_manifest_path"] = args.source_manifest.as_posix()
    manifest = build_single_bird_manifest(
        source_manifest,
        bird_id=args.bird_id,
        min_dph=int(args.min_dph),
        max_dph=int(args.max_dph),
        test_every_n=int(args.test_every_n),
        test_offset=int(args.test_offset),
        audio_root=args.audio_root,
        roi_root=args.roi_root,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)

    payload = {
        "bird_id_norm": manifest["filters"]["bird_id_norm"],
        "manifest_path": args.out.as_posix(),
        "train": manifest["summary"]["train"],
        "test": manifest["summary"]["test"],
    }
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
