#!/usr/bin/env python3
"""AWS Batch shard runner for birdsong latent export (S3 -> local -> S3).

This script is intended for AWS Batch *array jobs* where
`AWS_BATCH_JOB_ARRAY_INDEX` maps to `--shard-index`, and `--num-shards`
matches the array size.

Workflow (per shard):
1. Download manifest + config + checkpoint from S3.
2. Deterministically select this shard's manifest entries.
3. Download this shard's audio directories (+ ROI bundles when enabled).
4. Run `scripts/export_latent_sequences.py` on the shard subset.
5. Upload latent `.npz/.json` outputs back to S3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.cloud.manifest_sharding import (
    apply_max_dirs,
    iter_manifest_entry_pairs,
    pairs_to_manifest,
    select_shard,
)


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int | None = None) -> int | None:
    val = _env(name)
    if val is None:
        return default
    return int(val)


def _env_float(name: str, default: float | None = None) -> float | None:
    val = _env(name)
    if val is None:
        return default
    return float(val)


def _env_bool(name: str, default: bool = False) -> bool:
    val = _env(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _join_s3_uri(root_uri: str, rel: str) -> str:
    root_uri = str(root_uri).rstrip("/")
    rel = str(rel or ".").lstrip("/")
    if rel in (".", ""):
        return root_uri
    return f"{root_uri}/{rel}"


def _require_aws_cli() -> str:
    aws = shutil.which("aws")
    if not aws:
        raise RuntimeError(
            "aws CLI is required but was not found on PATH. "
            "Install awscli (or AWS CLI v2) and configure credentials."
        )
    return aws


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{output}")
    return proc


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _download_one_dir(
    aws: str,
    s3_src: str,
    local_dst: Path,
    include_patterns: list[str] | None,
    quiet: bool,
) -> dict:
    start = time.time()
    local_dst.mkdir(parents=True, exist_ok=True)
    cmd = [aws, "s3", "sync", str(s3_src), str(local_dst)]
    if include_patterns:
        cmd.extend(["--exclude", "*"])
        for pattern in include_patterns:
            cmd.extend(["--include", pattern])
    if quiet:
        cmd.append("--only-show-errors")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return {
        "s3_src": str(s3_src),
        "local_dst": local_dst.as_posix(),
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - start),
        "output": output.strip() if proc.returncode != 0 else None,
        "cmd": cmd if proc.returncode != 0 else None,
    }


def _count_wavs(audio_dir: Path) -> int:
    if not audio_dir.exists():
        return 0
    try:
        return len(
            [
                name
                for name in os.listdir(audio_dir.as_posix())
                if str(name).lower().endswith(".wav")
            ]
        )
    except FileNotFoundError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one shard of latent export inside an AWS Batch array job."
    )
    parser.add_argument(
        "--manifest-s3-uri",
        type=str,
        default=_env("AVA_MANIFEST_S3_URI"),
        required=False,
        help="S3 URI to manifest JSON, e.g. s3://bucket/path/birdsong_manifest.json",
    )
    parser.add_argument(
        "--config-s3-uri",
        type=str,
        default=_env("AVA_CONFIG_S3_URI"),
        required=False,
        help="S3 URI to fixed-window config YAML.",
    )
    parser.add_argument(
        "--checkpoint-s3-uri",
        type=str,
        default=_env("AVA_CHECKPOINT_S3_URI"),
        required=False,
        help="S3 URI to model checkpoint .tar file.",
    )
    parser.add_argument(
        "--s3-audio-root",
        type=str,
        default=_env("AVA_S3_AUDIO_ROOT"),
        required=False,
        help="S3 URI root for audio directories.",
    )
    parser.add_argument(
        "--s3-roi-root",
        type=str,
        default=_env("AVA_S3_ROI_ROOT"),
        required=False,
        help="S3 URI root for ROI directories (txt or parquet).",
    )
    parser.add_argument(
        "--s3-latent-root",
        type=str,
        default=_env("AVA_S3_LATENT_ROOT"),
        required=False,
        help="S3 URI root for latent outputs (.npz/.json).",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default=_env("AVA_SPLIT", "all"))
    parser.add_argument("--num-shards", type=int, default=_env_int("AVA_NUM_SHARDS"))
    parser.add_argument(
        "--shard-index",
        type=int,
        default=_env_int("AWS_BATCH_JOB_ARRAY_INDEX", 0),
        help="Shard index. Defaults to AWS_BATCH_JOB_ARRAY_INDEX (or 0).",
    )
    parser.add_argument("--max-dirs", type=int, default=_env_int("AVA_MAX_DIRS"))
    parser.add_argument("--max-files-per-dir", type=int, default=_env_int("AVA_MAX_FILES_PER_DIR"))
    parser.add_argument("--max-clips", type=int, default=_env_int("AVA_MAX_CLIPS"))

    parser.add_argument("--download-jobs", type=int, default=_env_int("AVA_DOWNLOAD_JOBS", 8))
    parser.add_argument("--include", action="append", default=None)

    parser.add_argument(
        "--roi-format",
        choices=["txt", "parquet"],
        default=_env("AVA_ROI_FORMAT", "parquet"),
    )
    parser.add_argument("--roi-parquet-name", type=str, default=_env("AVA_ROI_PARQUET_NAME", "roi.parquet"))
    parser.add_argument("--no-rois", action="store_true", default=_env_bool("AVA_NO_ROIS", False))

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=_env("AVA_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=_env_int("AVA_BATCH_SIZE", 64))
    parser.add_argument("--hop-length-sec", type=float, default=_env_float("AVA_HOP_LENGTH_SEC"))
    parser.add_argument("--start-time-sec", type=float, default=_env_float("AVA_START_TIME_SEC", 0.0))
    parser.add_argument("--end-time-sec", type=float, default=_env_float("AVA_END_TIME_SEC"))
    parser.add_argument("--export-energy", action="store_true", default=_env_bool("AVA_EXPORT_ENERGY", True))
    parser.add_argument("--audio-sha256", action="store_true", default=_env_bool("AVA_AUDIO_SHA256", False))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=_env_bool("AVA_SKIP_EXISTING", True),
        help="Skip clips that already exist locally in out-dir (delegates to exporter).",
    )
    parser.add_argument("--continue-on-error", action="store_true", default=_env_bool("AVA_CONTINUE_ON_ERROR", False))
    parser.add_argument("--report-every", type=int, default=_env_int("AVA_REPORT_EVERY", 25))

    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(_env("AVA_WORKDIR", "/tmp/ava_latent_workdir")),
        help="Local scratch directory for downloads and outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--s3-summary-root",
        type=str,
        default=_env("AVA_S3_SUMMARY_ROOT"),
        help="Optional S3 URI root for per-shard summaries.",
    )

    args = parser.parse_args()

    if not args.manifest_s3_uri:
        raise ValueError("--manifest-s3-uri (or AVA_MANIFEST_S3_URI) is required.")
    if not args.config_s3_uri:
        raise ValueError("--config-s3-uri (or AVA_CONFIG_S3_URI) is required.")
    if not args.checkpoint_s3_uri:
        raise ValueError("--checkpoint-s3-uri (or AVA_CHECKPOINT_S3_URI) is required.")
    if not args.s3_audio_root:
        raise ValueError("--s3-audio-root (or AVA_S3_AUDIO_ROOT) is required.")
    if not args.s3_latent_root:
        raise ValueError("--s3-latent-root (or AVA_S3_LATENT_ROOT) is required.")
    if not args.no_rois and not args.s3_roi_root:
        raise ValueError("--s3-roi-root (or AVA_S3_ROI_ROOT) is required unless --no-rois is set.")
    if args.num_shards is None:
        raise ValueError("--num-shards (or AVA_NUM_SHARDS) is required for deterministic array sharding.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards).")
    if args.download_jobs <= 0:
        raise ValueError("--download-jobs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_dirs is not None and int(args.max_dirs) <= 0:
        raise ValueError("--max-dirs must be positive.")
    if args.max_files_per_dir is not None and int(args.max_files_per_dir) <= 0:
        raise ValueError("--max-files-per-dir must be positive.")
    if args.max_clips is not None and int(args.max_clips) <= 0:
        raise ValueError("--max-clips must be positive.")

    audio_include_patterns = args.include

    local_manifest_path = args.workdir / "inputs" / "manifest.json"
    local_config_path = args.workdir / "inputs" / "config.yaml"
    local_checkpoint_path = args.workdir / "inputs" / "checkpoint.tar"
    local_manifest_subset_path = args.workdir / "inputs" / "manifest_subset.json"
    local_audio_root = args.workdir / "audio"
    local_roi_root = args.workdir / "roi"
    local_latent_root = args.workdir / "latent"
    local_export_summary_path = args.workdir / "out" / "latent_export_summary.json"
    local_job_summary_path = args.workdir / "out" / "job_summary.json"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest_s3_uri": str(args.manifest_s3_uri),
                    "config_s3_uri": str(args.config_s3_uri),
                    "checkpoint_s3_uri": str(args.checkpoint_s3_uri),
                    "s3_audio_root": str(args.s3_audio_root),
                    "s3_roi_root": str(args.s3_roi_root) if args.s3_roi_root else None,
                    "s3_latent_root": str(args.s3_latent_root),
                    "split": str(args.split),
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                    "max_dirs": args.max_dirs,
                    "max_files_per_dir": args.max_files_per_dir,
                    "max_clips": args.max_clips,
                    "roi_format": str(args.roi_format),
                    "no_rois": bool(args.no_rois),
                    "workdir": args.workdir.as_posix(),
                    "local_audio_root": local_audio_root.as_posix(),
                    "local_roi_root": local_roi_root.as_posix(),
                    "local_latent_root": local_latent_root.as_posix(),
                },
                indent=2,
            )
        )
        return

    aws = _require_aws_cli()

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Download inputs.
    local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _run([aws, "s3", "cp", str(args.manifest_s3_uri), str(local_manifest_path), "--only-show-errors"])
    _run([aws, "s3", "cp", str(args.config_s3_uri), str(local_config_path), "--only-show-errors"])
    _run([aws, "s3", "cp", str(args.checkpoint_s3_uri), str(local_checkpoint_path), "--only-show-errors"])

    manifest = _load_json(local_manifest_path)
    pairs = iter_manifest_entry_pairs(manifest, split=str(args.split))
    pairs = apply_max_dirs(pairs, max_dirs=args.max_dirs)
    planned_total = len(pairs)
    pairs = select_shard(pairs, num_shards=int(args.num_shards), shard_index=int(args.shard_index))
    shard_total = len(pairs)

    if not pairs:
        summary = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "nothing_to_do",
            "manifest_s3_uri": str(args.manifest_s3_uri),
            "split": str(args.split),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "planned_total": int(planned_total),
            "selected_for_shard": int(shard_total),
        }
        _write_json(local_job_summary_path, summary)
        print(json.dumps(summary, indent=2))
        if args.s3_summary_root:
            _run(
                [
                    aws,
                    "s3",
                    "cp",
                    str(local_job_summary_path),
                    _join_s3_uri(str(args.s3_summary_root), f"job_summary_shard_{args.shard_index}.json"),
                    "--only-show-errors",
                ]
            )
        return

    subset_manifest = pairs_to_manifest(pairs)
    _write_json(local_manifest_subset_path, subset_manifest)
    subset_train_count = int(len(subset_manifest.get("train", []) or []))
    subset_test_count = int(len(subset_manifest.get("test", []) or []))

    # Download audio/ROI directories for this shard.
    quiet = True
    download_failures = 0
    download_results: list[dict] = []
    start_dl = time.time()
    with ThreadPoolExecutor(max_workers=int(args.download_jobs)) as pool:
        futures = []
        for _, entry in pairs:
            rel = entry.get("audio_dir_rel")
            if rel is None:
                raise ValueError("Manifest entry missing audio_dir_rel (required for S3 layout).")
            rel = str(rel)

            s3_audio_src = _join_s3_uri(str(args.s3_audio_root), rel)
            local_audio_dst = local_audio_root / rel
            futures.append(
                pool.submit(
                    _download_one_dir,
                    aws,
                    s3_src=s3_audio_src,
                    local_dst=local_audio_dst,
                    include_patterns=audio_include_patterns,
                    quiet=quiet,
                )
            )

            if not args.no_rois:
                s3_roi_src = _join_s3_uri(str(args.s3_roi_root), rel)
                local_roi_dst = local_roi_root / rel
                roi_patterns = ["*.txt", "*.TXT"] if args.roi_format == "txt" else [str(args.roi_parquet_name)]
                futures.append(
                    pool.submit(
                        _download_one_dir,
                        aws,
                        s3_src=s3_roi_src,
                        local_dst=local_roi_dst,
                        include_patterns=roi_patterns,
                        quiet=quiet,
                    )
                )

        for fut in as_completed(futures):
            res = fut.result()
            download_results.append(res)
            if res.get("status") != "ok":
                download_failures += 1
    elapsed_dl = max(1e-9, time.time() - start_dl)

    if download_failures:
        summary = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "download_failed",
            "download_failures": int(download_failures),
            "download_total": int(len(download_results)),
            "download_elapsed_sec": float(elapsed_dl),
            "download_tasks_per_sec": float(len(download_results) / elapsed_dl),
            "errors": [r for r in download_results if r.get("status") != "ok"],
        }
        _write_json(local_job_summary_path, summary)
        print(json.dumps(summary, indent=2))
        if args.s3_summary_root:
            _run(
                [
                    aws,
                    "s3",
                    "cp",
                    str(local_job_summary_path),
                    _join_s3_uri(str(args.s3_summary_root), f"job_summary_shard_{args.shard_index}.json"),
                    "--only-show-errors",
                ]
            )
        sys.exit(1)

    # Ensure each selected directory has at least one wav before export.
    empty_audio_dirs: list[dict] = []
    audio_wav_counts: list[dict] = []
    for _, entry in pairs:
        rel = str(entry.get("audio_dir_rel") or "")
        local_audio_dir = local_audio_root / rel
        wav_count = _count_wavs(local_audio_dir)
        audio_wav_counts.append(
            {"audio_dir_rel": rel, "wav_count": int(wav_count)}
        )
        if wav_count <= 0:
            empty_audio_dirs.append(
                {"audio_dir_rel": rel, "local_audio_dir": local_audio_dir.as_posix()}
            )

    if empty_audio_dirs:
        summary = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "no_audio_downloaded",
            "download_total": int(len(download_results)),
            "download_elapsed_sec": float(elapsed_dl),
            "empty_audio_dirs": empty_audio_dirs,
            "audio_wav_counts": audio_wav_counts,
        }
        _write_json(local_job_summary_path, summary)
        print(json.dumps(summary, indent=2))
        if args.s3_summary_root:
            _run(
                [
                    aws,
                    "s3",
                    "cp",
                    str(local_job_summary_path),
                    _join_s3_uri(str(args.s3_summary_root), f"job_summary_shard_{args.shard_index}.json"),
                    "--only-show-errors",
                ]
            )
        sys.exit(1)

    export_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_latent_sequences.py"),
        "--manifest",
        str(local_manifest_subset_path),
        "--split",
        "all",
        "--config",
        str(local_config_path),
        "--checkpoint",
        str(local_checkpoint_path),
        "--out-dir",
        str(local_latent_root),
        "--audio-root",
        str(local_audio_root),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-shards",
        "1",
        "--shard-index",
        "0",
        "--report-every",
        str(int(args.report_every)),
        "--summary-out",
        str(local_export_summary_path),
    ]

    if not args.skip_existing:
        export_cmd.append("--no-skip-existing")
    if args.max_files_per_dir is not None:
        export_cmd.extend(["--max-files-per-dir", str(int(args.max_files_per_dir))])
    if args.max_clips is not None:
        export_cmd.extend(["--max-clips", str(int(args.max_clips))])

    if args.hop_length_sec is not None:
        export_cmd.extend(["--hop-length-sec", str(float(args.hop_length_sec))])
    if args.start_time_sec is not None:
        export_cmd.extend(["--start-time-sec", str(float(args.start_time_sec))])
    if args.end_time_sec is not None:
        export_cmd.extend(["--end-time-sec", str(float(args.end_time_sec))])
    if args.export_energy:
        export_cmd.append("--export-energy")
    if args.audio_sha256:
        export_cmd.append("--audio-sha256")
    if args.continue_on_error:
        export_cmd.append("--continue-on-error")

    if args.no_rois:
        export_cmd.append("--no-rois")
    else:
        export_cmd.extend(["--roi-root", str(local_roi_root), "--roi-format", str(args.roi_format)])
        if args.roi_format == "parquet":
            export_cmd.extend(["--roi-parquet-name", str(args.roi_parquet_name)])

    _run(export_cmd, check=True)

    # Upload latent artifacts.
    sync_cmd = [
        aws,
        "s3",
        "sync",
        str(local_latent_root),
        str(args.s3_latent_root),
        "--exclude",
        "*",
        "--include",
        "*.npz",
        "--include",
        "*.json",
        "--only-show-errors",
    ]
    _run(sync_cmd, check=True)

    export_summary = _load_json(local_export_summary_path) if local_export_summary_path.exists() else {}

    job_summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "manifest_s3_uri": str(args.manifest_s3_uri),
        "config_s3_uri": str(args.config_s3_uri),
        "checkpoint_s3_uri": str(args.checkpoint_s3_uri),
        "s3_audio_root": str(args.s3_audio_root),
        "s3_roi_root": str(args.s3_roi_root) if args.s3_roi_root else None,
        "s3_latent_root": str(args.s3_latent_root),
        "split": str(args.split),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "max_dirs": args.max_dirs,
        "max_files_per_dir": args.max_files_per_dir,
        "max_clips": args.max_clips,
        "subset_manifest_counts": {
            "train": int(subset_train_count),
            "test": int(subset_test_count),
        },
        "export_split": "all",
        "download_jobs": int(args.download_jobs),
        "download_total": int(len(download_results)),
        "download_elapsed_sec": float(elapsed_dl),
        "audio_wav_counts": audio_wav_counts,
        "roi_format": str(args.roi_format),
        "no_rois": bool(args.no_rois),
        "planned_total": int(planned_total),
        "selected_for_shard": int(shard_total),
        "latent_summary": export_summary,
    }
    _write_json(local_job_summary_path, job_summary)
    print(json.dumps(job_summary, indent=2))

    if args.s3_summary_root:
        _run(
            [
                aws,
                "s3",
                "cp",
                str(local_job_summary_path),
                _join_s3_uri(str(args.s3_summary_root), f"job_summary_shard_{args.shard_index}.json"),
                "--only-show-errors",
            ]
        )
        if local_export_summary_path.exists():
            _run(
                [
                    aws,
                    "s3",
                    "cp",
                    str(local_export_summary_path),
                    _join_s3_uri(
                        str(args.s3_summary_root),
                        f"latent_export_summary_shard_{args.shard_index}.json",
                    ),
                    "--only-show-errors",
                ]
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
