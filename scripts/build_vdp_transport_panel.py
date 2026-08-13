#!/usr/bin/env python3
"""Build the fixed longitudinal panel for VDP posterior transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _explicit_dph(value: object) -> float | None:
    if value is None:
        return None
    try:
        dph = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(dph) or dph < 0:
        return None
    return dph


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("split", "bird_id", "audio_dir_rel", "filename"):
                if key not in row:
                    raise ValueError(f"Member line {line_number} is missing {key!r}.")
            rows.append(row)
    if not rows:
        raise ValueError("Source member manifest is empty.")
    return rows


def _selection_key(row: dict[str, Any], *, seed: int) -> str:
    payload = "\0".join(
        [
            str(seed),
            "vdp_longitudinal_posterior_transport",
            str(row["bird_id"]),
            repr(row["dph"]),
            str(row["split"]),
            str(row["audio_dir_rel"]),
            str(row["filename"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tutor_start_by_dir(
    metadata_path: Path, directories: set[str]
) -> dict[str, float | None]:
    try:
        import pyarrow.dataset as ds  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Panel construction requires pyarrow.") from exc
    dataset = ds.dataset(metadata_path, format="parquet")
    required = {"audio_dir_rel", "tutor_start_day"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    table = dataset.to_table(
        columns=["audio_dir_rel", "tutor_start_day"],
        filter=ds.field("audio_dir_rel").isin(sorted(directories)),
    )
    values: dict[str, set[float]] = defaultdict(set)
    seen: set[str] = set()
    for row in table.to_pylist():
        rel = str(row["audio_dir_rel"])
        seen.add(rel)
        value = row.get("tutor_start_day")
        if value is not None:
            values[rel].add(float(value))
    missing_dirs = sorted(directories - seen)
    if missing_dirs:
        raise ValueError(
            f"Metadata has no rows for {len(missing_dirs)} selected directories; "
            f"first is {missing_dirs[0]!r}."
        )
    result: dict[str, float | None] = {}
    for rel in directories:
        found = values.get(rel, set())
        if len(found) > 1:
            raise ValueError(f"Directory {rel!r} has conflicting tutor-start values.")
        result[rel] = next(iter(found)) if found else None
    return result


def build_panel(
    *,
    source_manifest_path: Path,
    source_members_path: Path,
    metadata_path: Path,
    min_clips_per_measure: int = 2,
    max_clips_per_measure: int = 2,
    min_measures_per_bird: int = 20,
    seed: int = 20260813,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if min_clips_per_measure < 1:
        raise ValueError("min_clips_per_measure must be positive.")
    if max_clips_per_measure < min_clips_per_measure:
        raise ValueError("max_clips_per_measure must be at least the minimum.")
    if min_measures_per_bird < 2:
        raise ValueError("min_measures_per_bird must be at least two.")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    source_birds: set[str] = set()
    for split in ("train", "test"):
        for entry in source_manifest.get(split, []):
            key = (split, str(entry["audio_dir_rel"]))
            if key in entries:
                raise ValueError(f"Duplicate source manifest entry {key}.")
            entries[key] = entry
            bird = entry.get("bird_id_norm")
            if bird is not None:
                source_birds.add(str(bird))

    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    nonexplicit_members = 0
    bird_regimes: dict[str, set[str]] = defaultdict(set)
    bird_splits: dict[str, set[str]] = defaultdict(set)
    for member in _load_jsonl(source_members_path):
        split = str(member["split"])
        key = (split, str(member["audio_dir_rel"]))
        entry = entries.get(key)
        if entry is None:
            raise ValueError(f"Source member has no matching manifest entry: {key}.")
        bird = str(entry.get("bird_id_norm") or member["bird_id"])
        regime = entry.get("regime")
        if regime is None:
            raise ValueError(f"Member {member['filename']!r} lacks explicit regime.")
        dph = _explicit_dph(entry.get("dph"))
        if dph is None:
            nonexplicit_members += 1
            continue
        bird_regimes[bird].add(str(regime))
        bird_splits[bird].add(split)
        grouped[(bird, dph, str(regime), split)].append(
            dict(member, bird_id=bird, dph=dph, regime=str(regime))
        )

    conflicting_regime = sorted(
        bird for bird, regimes in bird_regimes.items() if len(regimes) != 1
    )
    split_overlap = sorted(
        bird for bird, splits in bird_splits.items() if len(splits) != 1
    )
    if conflicting_regime:
        raise ValueError(f"Bird spans regimes: {conflicting_regime[0]}.")
    if split_overlap:
        raise ValueError(f"Bird spans train/test splits: {split_overlap[0]}.")

    eligible_groups = {
        key: rows for key, rows in grouped.items() if len(rows) >= min_clips_per_measure
    }
    measures_by_bird = Counter(key[0] for key in eligible_groups)
    included_birds = {
        bird
        for bird, count in measures_by_bird.items()
        if count >= min_measures_per_bird
    }
    selected: list[dict[str, Any]] = []
    for key in sorted(eligible_groups):
        bird, _, _, _ = key
        if bird not in included_birds:
            continue
        ordered = sorted(
            eligible_groups[key], key=lambda row: _selection_key(row, seed=seed)
        )
        selected.extend(ordered[:max_clips_per_measure])
    if not selected:
        raise ValueError("No measures pass the transport panel gates.")

    directories = {str(row["audio_dir_rel"]) for row in selected}
    tutor_by_dir = _tutor_start_by_dir(metadata_path, directories)
    for row in selected:
        row["recording_id"] = None
        row["tutor_start_dph"] = tutor_by_dir[str(row["audio_dir_rel"])]
        row["panel"] = "longitudinal_posterior_transport"
    selected.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["bird_id"]),
            float(row["dph"]),
            str(row["audio_dir_rel"]),
            str(row["filename"]),
        )
    )

    counts = Counter((str(row["split"]), str(row["audio_dir_rel"])) for row in selected)
    panel_entries: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for key in sorted(counts):
        entry = dict(entries[key])
        entry.pop("audio_dir", None)
        entry.pop("roi_dir", None)
        entry["num_files"] = counts[key]
        entry["recording_id"] = None
        entry["tutor_start_dph"] = tutor_by_dir[str(entry["audio_dir_rel"])]
        entry["panel"] = "longitudinal_posterior_transport"
        panel_entries[key[0]].append(entry)

    included_measures = {(str(row["bird_id"]), float(row["dph"])) for row in selected}
    included_by_split = Counter(
        next(iter(bird_splits[bird])) for bird in included_birds
    )
    included_by_regime = Counter(
        next(iter(bird_regimes[bird])) for bird in included_birds
    )
    excluded_birds = sorted(source_birds - included_birds)
    manifest = {
        "schema_version": "vdp_longitudinal_transport_panel_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_member_manifest_sha256": _sha256_file(source_members_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "split_semantics": "bird_disjoint",
        "selection": {
            "seed": seed,
            "rule": "stable hash rank independently within bird-by-explicit-dph",
            "min_clips_per_measure": min_clips_per_measure,
            "max_clips_per_measure": max_clips_per_measure,
            "min_measures_per_bird": min_measures_per_bird,
            "missing_values": "excluded_not_inferred",
        },
        "summary": {
            "source_birds": len(source_birds),
            "included_birds": len(included_birds),
            "excluded_birds": excluded_birds,
            "included_birds_by_split": dict(sorted(included_by_split.items())),
            "included_birds_by_regime": dict(sorted(included_by_regime.items())),
            "included_measures": len(included_measures),
            "selected_members": len(selected),
            "source_groups_with_explicit_dph": len(grouped),
            "source_groups_below_clip_minimum": len(grouped) - len(eligible_groups),
            "source_members_without_explicit_dph": nonexplicit_members,
            "bird_split_overlap": split_overlap,
        },
        "train": panel_entries["train"],
        "test": panel_entries["test"],
    }
    return manifest, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-members", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-members", type=Path, required=True)
    parser.add_argument("--min-clips-per-measure", type=int, default=2)
    parser.add_argument("--max-clips-per-measure", type=int, default=2)
    parser.add_argument("--min-measures-per-bird", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    manifest, members = build_panel(
        source_manifest_path=args.source_manifest,
        source_members_path=args.source_members,
        metadata_path=args.metadata,
        min_clips_per_measure=args.min_clips_per_measure,
        max_clips_per_measure=args.max_clips_per_measure,
        min_measures_per_bird=args.min_measures_per_bird,
        seed=args.seed,
    )
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_members.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_members.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in members
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **manifest["summary"],
                "manifest_sha256": _sha256_file(args.out_manifest),
                "members_sha256": _sha256_file(args.out_members),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
