#!/usr/bin/env python3
"""Build file-level metadata for the birdsong dataset."""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REGIME_RE = re.compile(r"\b(bells|simple|samba|isolates)\b", re.IGNORECASE)
DAY_RE = re.compile(r"day[ _-]*(\d+)", re.IGNORECASE)
BIRD_RE = re.compile(r"^[A-Za-z]+\s*\d+$")
SESSION_RE = re.compile(
    r"^(?:\d+)?(?P<mon>[A-Za-z]{3,9})_(?P<day>\d{2})_(?P<hour>\d{2})_(?P<min>\d{2})$"
)
RECORDING_RE = re.compile(
    r"^(?P<pre>.+?)_on_(?P<mon>[A-Za-z]{3,9})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<min>\d{2})(?:_(?P<sec>\d{2}))?$"
)
BIRD_IN_NAME_RE = re.compile(r"\bbird(?P<digits>\d+)\b", re.IGNORECASE)

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def month_to_int(mon):
    if not mon:
        return None
    key = mon.strip().lower()
    if len(key) >= 3:
        key = key[:3]
    return MONTH_MAP.get(key)


def parse_dir_meta(rel_dir):
    if rel_dir == ".":
        parts = []
    else:
        parts = rel_dir.split(os.sep)

    top_dir = parts[0] if parts else None

    regime = None
    tutor_start_day = None
    if top_dir:
        match = REGIME_RE.search(top_dir)
        if match:
            regime = match.group(1).lower()
        match = DAY_RE.search(top_dir)
        if match:
            tutor_start_day = int(match.group(1))

    photoperiod = None
    bird_id_raw = None
    bird_idx = None
    for idx, part in enumerate(parts[1:], start=1):
        if photoperiod is None and "photoperiod" in part.lower():
            photoperiod = part
            continue
        if bird_id_raw is None and BIRD_RE.match(part):
            bird_id_raw = part
            bird_idx = idx
            break

    pre_bird_path = None
    session_label = None
    dph = None
    session_month = None
    session_day = None
    session_hour = None
    session_min = None

    if bird_idx is not None:
        pre_bird_parts = []
        for part in parts[1:bird_idx]:
            if photoperiod and part == photoperiod:
                continue
            pre_bird_parts.append(part)
        if pre_bird_parts:
            pre_bird_path = "/".join(pre_bird_parts)

        tail_parts = parts[bird_idx + 1 :]
        session_parts = []
        for part in tail_parts:
            if part.isdigit():
                dph = int(part)
                break
            session_parts.append(part)

        if session_parts:
            session_label = "/".join(session_parts)
            session_base = session_parts[-1]
            match = SESSION_RE.match(session_base)
            if match:
                session_month = month_to_int(match.group("mon"))
                session_day = int(match.group("day"))
                session_hour = int(match.group("hour"))
                session_min = int(match.group("min"))

    bird_id_norm = None
    bird_id_num = None
    if bird_id_raw:
        bird_id_norm = re.sub(r"\s+", "", bird_id_raw).upper()
        match = re.search(r"(\d+)", bird_id_raw)
        if match:
            bird_id_num = int(match.group(1))

    return {
        "top_dir": top_dir,
        "regime": regime,
        "tutor_start_day": tutor_start_day,
        "photoperiod": photoperiod,
        "pre_bird_path": pre_bird_path,
        "bird_id_raw": bird_id_raw,
        "bird_id_norm": bird_id_norm,
        "bird_id_num": bird_id_num,
        "session_label": session_label,
        "session_month": session_month,
        "session_day": session_day,
        "session_hour": session_hour,
        "session_min": session_min,
        "dph": dph,
    }


def parse_file_meta(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    file_ext = ext[1:] if ext else None
    is_audio = file_ext == "wav"
    is_appledouble = file_name.startswith("._")

    file_index = None
    recording_month = None
    recording_day = None
    recording_hour = None
    recording_min = None
    recording_sec = None

    stem = os.path.splitext(file_name)[0]
    if "_on_" in stem:
        match = RECORDING_RE.match(stem)
        if match:
            pre = match.group("pre")
            for token in reversed(pre.split("_")):
                if token.isdigit():
                    file_index = int(token)
                    break
            recording_month = month_to_int(match.group("mon"))
            recording_day = int(match.group("day"))
            recording_hour = int(match.group("hour"))
            recording_min = int(match.group("min"))
            sec = match.group("sec")
            if sec is not None:
                recording_sec = int(sec)
        else:
            match = BIRD_IN_NAME_RE.search(stem)
            if match:
                file_index = None
    return {
        "file_name": file_name,
        "file_ext": file_ext,
        "file_index": file_index,
        "recording_month": recording_month,
        "recording_day": recording_day,
        "recording_hour": recording_hour,
        "recording_min": recording_min,
        "recording_sec": recording_sec,
        "is_audio": is_audio,
        "is_appledouble": is_appledouble,
    }


def write_chunk(writer, columns):
    if not columns or len(next(iter(columns.values()))) == 0:
        return
    table = pa.Table.from_pydict(columns, schema=writer.schema)
    writer.write_table(table)


def build_metadata(root, out_path, summary_path, chunk_size, report_every):
    root = Path(root).resolve()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("rel_path", pa.string()),
            ("audio_dir_rel", pa.string()),
            ("top_dir", pa.string()),
            ("regime", pa.string()),
            ("tutor_start_day", pa.int32()),
            ("photoperiod", pa.string()),
            ("pre_bird_path", pa.string()),
            ("bird_id_raw", pa.string()),
            ("bird_id_norm", pa.string()),
            ("bird_id_num", pa.int32()),
            ("session_label", pa.string()),
            ("session_month", pa.int32()),
            ("session_day", pa.int32()),
            ("session_hour", pa.int32()),
            ("session_min", pa.int32()),
            ("dph", pa.int32()),
            ("file_name", pa.string()),
            ("file_ext", pa.string()),
            ("file_index", pa.int64()),
            ("recording_month", pa.int32()),
            ("recording_day", pa.int32()),
            ("recording_hour", pa.int32()),
            ("recording_min", pa.int32()),
            ("recording_sec", pa.int32()),
            ("size_bytes", pa.int64()),
            ("is_audio", pa.bool_()),
            ("is_appledouble", pa.bool_()),
        ]
    )

    writer = pq.ParquetWriter(out_path.as_posix(), schema, compression="zstd")

    columns = {name: [] for name in schema.names}
    dir_cache = {}

    total = 0
    audio_count = 0
    ext_counts = Counter()
    top_dir_counts = Counter()
    regime_counts = Counter()

    for dirpath, _, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir not in dir_cache:
            dir_cache[rel_dir] = parse_dir_meta(rel_dir)
        dir_meta = dir_cache[rel_dir]

        audio_dir_rel = "." if rel_dir == "." else rel_dir.replace(os.sep, "/")

        for file_name in filenames:
            rel_path = file_name if rel_dir == "." else os.path.join(rel_dir, file_name)
            rel_path = rel_path.replace(os.sep, "/")

            file_meta = parse_file_meta(file_name)

            try:
                size_bytes = os.stat(os.path.join(dirpath, file_name)).st_size
            except OSError:
                size_bytes = None

            columns["rel_path"].append(rel_path)
            columns["audio_dir_rel"].append(audio_dir_rel)
            columns["top_dir"].append(dir_meta["top_dir"])
            columns["regime"].append(dir_meta["regime"])
            columns["tutor_start_day"].append(dir_meta["tutor_start_day"])
            columns["photoperiod"].append(dir_meta["photoperiod"])
            columns["pre_bird_path"].append(dir_meta["pre_bird_path"])
            columns["bird_id_raw"].append(dir_meta["bird_id_raw"])
            columns["bird_id_norm"].append(dir_meta["bird_id_norm"])
            columns["bird_id_num"].append(dir_meta["bird_id_num"])
            columns["session_label"].append(dir_meta["session_label"])
            columns["session_month"].append(dir_meta["session_month"])
            columns["session_day"].append(dir_meta["session_day"])
            columns["session_hour"].append(dir_meta["session_hour"])
            columns["session_min"].append(dir_meta["session_min"])
            columns["dph"].append(dir_meta["dph"])
            columns["file_name"].append(file_meta["file_name"])
            columns["file_ext"].append(file_meta["file_ext"])
            columns["file_index"].append(file_meta["file_index"])
            columns["recording_month"].append(file_meta["recording_month"])
            columns["recording_day"].append(file_meta["recording_day"])
            columns["recording_hour"].append(file_meta["recording_hour"])
            columns["recording_min"].append(file_meta["recording_min"])
            columns["recording_sec"].append(file_meta["recording_sec"])
            columns["size_bytes"].append(size_bytes)
            columns["is_audio"].append(file_meta["is_audio"])
            columns["is_appledouble"].append(file_meta["is_appledouble"])

            total += 1
            if file_meta["is_audio"]:
                audio_count += 1
            ext_counts[file_meta["file_ext"] or "<none>"] += 1
            if dir_meta["top_dir"]:
                top_dir_counts[dir_meta["top_dir"]] += 1
            if dir_meta["regime"]:
                regime_counts[dir_meta["regime"]] += 1

            if total % report_every == 0:
                print(f"Processed {total} files...", file=sys.stderr)

            if len(columns["rel_path"]) >= chunk_size:
                write_chunk(writer, columns)
                columns = {name: [] for name in schema.names}

    write_chunk(writer, columns)
    writer.close()

    summary = {
        "root": root.as_posix(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": total,
        "audio_rows": audio_count,
        "top_dir_counts": dict(top_dir_counts),
        "regime_counts": dict(regime_counts),
        "ext_counts": dict(ext_counts),
        "output": out_path.as_posix(),
    }

    if summary_path:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/Volumes/samsung_ssd/data/birdsong",
        help="Root birdsong dataset directory",
    )
    parser.add_argument(
        "--out",
        default="data/metadata/birdsong/birdsong_files.parquet",
        help="Output Parquet filename",
    )
    parser.add_argument(
        "--summary",
        default="data/metadata/birdsong/birdsong_files_summary.json",
        help="Output summary JSON filename",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Rows per Parquet write",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=500000,
        help="Progress reporting interval",
    )

    args = parser.parse_args()

    summary = build_metadata(
        args.root,
        args.out,
        args.summary,
        args.chunk_size,
        args.report_every,
    )

    print(
        f"Wrote {summary['rows']} rows to {summary['output']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
