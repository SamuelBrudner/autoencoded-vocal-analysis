#!/usr/bin/env python3
"""Audit a birdsong manifest for frozen-latent transport analyses."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TUTOR_DAY_RE = re.compile(r"\bday\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)


def _load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        for entry in manifest.get(split, []):
            copied = dict(entry)
            copied.setdefault("split", split)
            out.append(copied)
    return out


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_tutor_start_day(entry: Dict[str, Any]) -> Optional[float]:
    direct = _finite_float(entry.get("tutor_start_day"))
    if direct is not None:
        return direct
    for key in ("top_dir", "pre_bird_path", "audio_dir_rel"):
        value = entry.get(key)
        if not value:
            continue
        match = TUTOR_DAY_RE.search(str(value))
        if match:
            return float(match.group(1))
    return None


def _sorted_numbers(values: Iterable[float]) -> List[float]:
    return sorted({float(value) for value in values if math.isfinite(float(value))})


def _bird_summary(entries: List[Dict[str, Any]], min_longitudinal_days: int) -> List[Dict[str, Any]]:
    by_bird: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        bird = entry.get("bird_id_norm")
        if bird:
            by_bird[str(bird)].append(entry)

    rows: List[Dict[str, Any]] = []
    for bird, bird_entries in sorted(by_bird.items()):
        regimes = sorted({str(e.get("regime")) for e in bird_entries if e.get("regime")})
        splits = sorted({str(e.get("split")) for e in bird_entries if e.get("split")})
        dphs = _sorted_numbers(
            dph
            for dph in (_finite_float(e.get("dph")) for e in bird_entries)
            if dph is not None
        )
        tutor_days = _sorted_numbers(
            day
            for day in (_parse_tutor_start_day(e) for e in bird_entries)
            if day is not None
        )
        files = sum(int(e.get("num_files") or 0) for e in bird_entries)
        rows.append(
            {
                "bird_id_norm": bird,
                "regimes": regimes,
                "splits": splits,
                "directories": len(bird_entries),
                "files": files,
                "dph_count": len(dphs),
                "dph_min": dphs[0] if dphs else None,
                "dph_max": dphs[-1] if dphs else None,
                "longitudinal": len(dphs) >= min_longitudinal_days,
                "tutor_start_days": tutor_days,
            }
        )
    return rows


def _summarize(entries: List[Dict[str, Any]], min_longitudinal_days: int) -> Dict[str, Any]:
    bird_rows = _bird_summary(entries, min_longitudinal_days=min_longitudinal_days)
    bird_regimes: Dict[str, set] = defaultdict(set)
    longitudinal_by_regime: Dict[str, set] = defaultdict(set)
    birds_by_split: Dict[str, set] = defaultdict(set)
    tutor_days_by_regime: Dict[str, set] = defaultdict(set)

    dph_missing_dirs = 0
    dph_missing_files = 0
    tutor_day_missing_dirs = 0
    tutor_day_missing_files = 0
    days_since_tutor_dirs = 0

    for entry in entries:
        bird = entry.get("bird_id_norm")
        regime = str(entry.get("regime") or "unknown")
        split = str(entry.get("split") or "unknown")
        num_files = int(entry.get("num_files") or 0)
        dph = _finite_float(entry.get("dph"))
        tutor_day = _parse_tutor_start_day(entry)
        if bird:
            bird_regimes[regime].add(str(bird))
            birds_by_split[split].add(str(bird))
        if dph is None:
            dph_missing_dirs += 1
            dph_missing_files += num_files
        if tutor_day is None:
            tutor_day_missing_dirs += 1
            tutor_day_missing_files += num_files
        else:
            tutor_days_by_regime[regime].add(float(tutor_day))
        if dph is not None and tutor_day is not None:
            days_since_tutor_dirs += 1

    bird_lookup = {row["bird_id_norm"]: row for row in bird_rows}
    for regime, birds in bird_regimes.items():
        for bird in birds:
            if bird_lookup[bird]["longitudinal"]:
                longitudinal_by_regime[regime].add(bird)

    train_birds = birds_by_split.get("train", set())
    test_birds = birds_by_split.get("test", set())
    overlap = sorted(train_birds.intersection(test_birds))
    multi_regime_birds = [
        row["bird_id_norm"] for row in bird_rows if len(row["regimes"]) > 1
    ]
    longitudinal_birds = [
        row["bird_id_norm"] for row in bird_rows if row["longitudinal"]
    ]

    tutor_day_values = {
        regime: sorted(values) for regime, values in sorted(tutor_days_by_regime.items())
    }
    tutor_day_variation = {
        regime: len(values) for regime, values in tutor_day_values.items()
    }
    tutored_tutor_days = set()
    for regime, values in tutor_days_by_regime.items():
        if regime != "isolates":
            tutored_tutor_days.update(values)

    return {
        "total_directories": len(entries),
        "total_files": sum(int(entry.get("num_files") or 0) for entry in entries),
        "total_birds": len(bird_rows),
        "birds_by_regime": {
            regime: len(birds) for regime, birds in sorted(bird_regimes.items())
        },
        "birds_by_split": {
            split: len(birds) for split, birds in sorted(birds_by_split.items())
        },
        "bird_split_overlap": overlap,
        "multi_regime_birds": sorted(multi_regime_birds),
        "longitudinal_threshold_days": int(min_longitudinal_days),
        "longitudinal_birds": len(longitudinal_birds),
        "longitudinal_by_regime": {
            regime: len(birds) for regime, birds in sorted(longitudinal_by_regime.items())
        },
        "dph_missing_dirs": dph_missing_dirs,
        "dph_missing_files": dph_missing_files,
        "tutor_day_missing_dirs": tutor_day_missing_dirs,
        "tutor_day_missing_files": tutor_day_missing_files,
        "days_since_tutor_available_dirs": days_since_tutor_dirs,
        "tutor_start_day_values_by_regime": tutor_day_values,
        "tutor_start_day_value_count_by_regime": tutor_day_variation,
        "readiness": {
            "bird_level_split_ok": not overlap,
            "single_regime_per_bird": not multi_regime_birds,
            "has_longitudinal_multi_bird": len(longitudinal_birds) >= 2,
            "has_tutor_onset_variation": len(tutored_tutor_days) > 1,
            "has_isolate_contrast": "isolates" in bird_regimes
            and bool(set(bird_regimes.keys()) - {"isolates"}),
            "can_compute_days_since_tutor": days_since_tutor_dirs > 0,
        },
        "birds": bird_rows,
    }


def _markdown_report(summary: Dict[str, Any], manifest_path: Path) -> str:
    readiness = summary["readiness"]
    lines = [
        "# Developmental Transport Cohort Readiness",
        "",
        f"- Manifest: `{manifest_path.as_posix()}`",
        f"- Birds: {summary['total_birds']}",
        f"- Directories: {summary['total_directories']}",
        f"- Files: {summary['total_files']}",
        f"- Longitudinal birds: {summary['longitudinal_birds']} "
        f"(threshold {summary['longitudinal_threshold_days']} DPH values)",
        f"- DPH-missing directories: {summary['dph_missing_dirs']}",
        f"- Tutor-day-missing directories: {summary['tutor_day_missing_dirs']}",
        "",
        "## Readiness Gates",
        "",
    ]
    for key, value in readiness.items():
        status = "PASS" if value else "FAIL"
        lines.append(f"- {status}: `{key}`")
    lines.extend(["", "## Birds By Regime", ""])
    for regime, count in summary["birds_by_regime"].items():
        long_count = summary["longitudinal_by_regime"].get(regime, 0)
        tutor_days = summary["tutor_start_day_values_by_regime"].get(regime, [])
        lines.append(
            f"- {regime}: {count} birds, {long_count} longitudinal, "
            f"tutor_start_days={tutor_days}"
        )
    if summary["bird_split_overlap"]:
        lines.extend(["", "## Split Leakage", ""])
        lines.append(
            "- Birds in both train and test: "
            + ", ".join(summary["bird_split_overlap"])
        )
    if summary["multi_regime_birds"]:
        lines.extend(["", "## Multi-Regime Birds", ""])
        lines.append("- " + ", ".join(summary["multi_regime_birds"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a birdsong manifest for developmental transport readiness."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--min-longitudinal-days", type=int, default=20)
    args = parser.parse_args()

    if args.min_longitudinal_days <= 0:
        raise ValueError("--min-longitudinal-days must be positive.")

    manifest = _load_manifest(args.manifest)
    entries = _entries(manifest)
    if not entries:
        raise ValueError("Manifest has no train/test entries.")

    summary = _summarize(
        entries,
        min_longitudinal_days=int(args.min_longitudinal_days),
    )
    payload = {
        "manifest_path": args.manifest.as_posix(),
        "summary": summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            _markdown_report(summary, manifest_path=args.manifest),
            encoding="utf-8",
        )

    readiness = summary["readiness"]
    passed = sum(1 for value in readiness.values() if value)
    print(
        "Readiness gates: "
        f"{passed}/{len(readiness)} passed; "
        f"birds={summary['total_birds']} longitudinal={summary['longitudinal_birds']}"
    )


if __name__ == "__main__":
    main()
