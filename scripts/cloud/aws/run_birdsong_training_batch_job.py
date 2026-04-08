#!/usr/bin/env python3
"""AWS Batch runner for fixed-window birdsong training."""

from __future__ import annotations

import argparse
from collections import deque
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

from ava.models.disk_telemetry import (
    append_disk_telemetry_snapshot,
    collect_disk_telemetry,
    format_disk_telemetry_summary,
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
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{output}"
        )
    return proc


def _stream_to_log(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail: deque[str] = deque(maxlen=200)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
            tail.append(line)
        returncode = proc.wait()
    output = "".join(tail)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={returncode}): {' '.join(cmd)}\n{output}"
        )
    return subprocess.CompletedProcess(cmd, returncode, output, None)


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
    include_patterns: list[str],
    quiet: bool,
) -> dict:
    start = time.time()
    local_dst.mkdir(parents=True, exist_ok=True)
    cmd = [aws, "s3", "sync", str(s3_src), str(local_dst), "--exclude", "*"]
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


def _sync_path_to_s3(aws: str, local_path: Path, s3_uri: str) -> None:
    if not local_path.exists():
        return
    if local_path.is_dir():
        _run([aws, "s3", "sync", local_path.as_posix(), str(s3_uri), "--only-show-errors"])
    else:
        _run([aws, "s3", "cp", local_path.as_posix(), str(s3_uri), "--only-show-errors"])


def _unique_entries(manifest: dict) -> list[dict]:
    ordered: list[dict] = []
    seen: set[str] = set()
    for split in ("train", "test"):
        for entry in manifest.get(split, []):
            rel = entry.get("audio_dir_rel")
            if rel is None:
                raise ValueError("Manifest entry missing audio_dir_rel.")
            key = str(rel)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(entry)
    return ordered


def _sum_manifest_files(manifest: dict) -> int:
    total = 0
    for split in ("train", "test"):
        for entry in manifest.get(split, []):
            total += int(entry.get("num_files") or 0)
    return int(total)


def _record_disk_telemetry(
    output_dir: Path,
    stage: str,
    roots: list[Path],
    *,
    extra: dict | None = None,
) -> dict:
    payload = collect_disk_telemetry(roots)
    payload["stage"] = stage
    if extra:
        payload.update(extra)
    print(format_disk_telemetry_summary(payload), flush=True)
    try:
        append_disk_telemetry_snapshot(output_dir, payload)
    except Exception as exc:  # pragma: no cover - best effort during disk failures
        print(
            f"Warning: failed to persist disk telemetry snapshot for stage={stage}: {exc}",
            file=sys.stderr,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a birdsong training job inside AWS Batch.")
    parser.add_argument("--manifest-s3-uri", type=str, default=_env("AVA_MANIFEST_S3_URI"), required=False)
    parser.add_argument("--config-s3-uri", type=str, default=_env("AVA_CONFIG_S3_URI"), required=False)
    parser.add_argument("--s3-audio-root", type=str, default=_env("AVA_S3_AUDIO_ROOT"), required=False)
    parser.add_argument("--s3-roi-root", type=str, default=_env("AVA_S3_ROI_ROOT"), required=False)
    parser.add_argument("--s3-run-root", type=str, default=_env("AVA_S3_RUN_ROOT"), required=False)
    parser.add_argument("--run-name", type=str, default=_env("AVA_RUN_NAME"))
    parser.add_argument("--roi-format", choices=["txt", "parquet"], default=_env("AVA_ROI_FORMAT", "parquet"))
    parser.add_argument("--roi-parquet-name", type=str, default=_env("AVA_ROI_PARQUET_NAME", "roi.parquet"))
    parser.add_argument("--download-jobs", type=int, default=_env_int("AVA_DOWNLOAD_JOBS", 8))
    parser.add_argument("--batch-size", type=int, default=_env_int("AVA_BATCH_SIZE"))
    parser.add_argument("--num-workers", type=int, default=_env_int("AVA_NUM_WORKERS"))
    parser.add_argument("--epochs", type=int, default=_env_int("AVA_EPOCHS"))
    parser.add_argument("--train-dataset-length", type=int, default=_env_int("AVA_TRAIN_DATASET_LENGTH"))
    parser.add_argument("--test-dataset-length", type=int, default=_env_int("AVA_TEST_DATASET_LENGTH"))
    parser.add_argument("--spec-cache-dir", type=Path, default=Path(_env("AVA_SPEC_CACHE_DIR", "/mnt/ava_cache/spec_cache")))
    parser.add_argument("--trainer-kwargs-json", type=str, default=_env("AVA_TRAINER_KWARGS_JSON"))
    parser.add_argument("--preflight-sample-dirs", type=int, default=_env_int("AVA_PREFLIGHT_SAMPLE_DIRS", 25))
    parser.add_argument("--preflight-sample-segments", type=int, default=_env_int("AVA_PREFLIGHT_SAMPLE_SEGMENTS", 5000))
    parser.add_argument("--preflight-seed", type=int, default=_env_int("AVA_PREFLIGHT_SEED", 0))
    parser.add_argument("--max-empty-fraction", type=float, default=_env_float("AVA_MAX_EMPTY_FRACTION", 0.01))
    parser.add_argument(
        "--disk-telemetry-every-n-epochs",
        type=int,
        default=_env_int("AVA_DISK_TELEMETRY_EVERY_N_EPOCHS", 5),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(_env("AVA_WORKDIR", "/mnt/ava_cache/ava_train_workdir")),
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.manifest_s3_uri:
        raise ValueError("--manifest-s3-uri (or AVA_MANIFEST_S3_URI) is required.")
    if not args.config_s3_uri:
        raise ValueError("--config-s3-uri (or AVA_CONFIG_S3_URI) is required.")
    if not args.s3_audio_root:
        raise ValueError("--s3-audio-root (or AVA_S3_AUDIO_ROOT) is required.")
    if not args.s3_roi_root:
        raise ValueError("--s3-roi-root (or AVA_S3_ROI_ROOT) is required.")
    if not args.s3_run_root:
        raise ValueError("--s3-run-root (or AVA_S3_RUN_ROOT) is required.")
    if args.download_jobs <= 0:
        raise ValueError("--download-jobs must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.train_dataset_length is not None and args.train_dataset_length <= 0:
        raise ValueError("--train-dataset-length must be positive.")
    if args.test_dataset_length is not None and args.test_dataset_length <= 0:
        raise ValueError("--test-dataset-length must be positive.")
    if args.preflight_sample_dirs <= 0 or args.preflight_sample_segments <= 0:
        raise ValueError("Preflight sample sizes must be positive.")
    if not (0.0 <= float(args.max_empty_fraction) <= 1.0):
        raise ValueError("--max-empty-fraction must be in [0, 1].")
    if args.disk_telemetry_every_n_epochs is not None and args.disk_telemetry_every_n_epochs <= 0:
        raise ValueError("--disk-telemetry-every-n-epochs must be positive.")

    run_name = str(
        args.run_name
        or os.environ.get("AWS_BATCH_JOB_NAME")
        or f"birdsong-train-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    )
    s3_run_root = _join_s3_uri(str(args.s3_run_root), run_name)

    local_inputs_dir = args.workdir / "inputs"
    local_logs_dir = args.workdir / "logs"
    local_coverage_dir = args.workdir / "coverage"
    local_audio_root = args.workdir / "audio"
    local_roi_root = args.workdir / "roi"
    local_run_dir = args.workdir / "training_run"
    local_tmp_dir = args.workdir / "tmp"
    local_summary_path = args.workdir / "job_summary.json"
    local_manifest_path = local_inputs_dir / "manifest.json"
    local_config_path = local_inputs_dir / "config.yaml"
    local_coverage_log_path = local_logs_dir / "coverage_stdout.log"
    local_training_log_path = local_logs_dir / "training_stdout.log"
    local_download_summary_path = local_logs_dir / "download_summary.json"
    local_disk_telemetry_dir = local_logs_dir / "disk_telemetry"
    disk_telemetry_roots = [
        Path("/tmp"),
        Path("/mnt/ava_cache"),
        args.workdir,
        local_tmp_dir,
        args.spec_cache_dir,
        local_audio_root,
        local_roi_root,
        local_run_dir,
    ]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest_s3_uri": str(args.manifest_s3_uri),
                    "config_s3_uri": str(args.config_s3_uri),
                    "s3_audio_root": str(args.s3_audio_root),
                    "s3_roi_root": str(args.s3_roi_root),
                    "s3_run_root": str(s3_run_root),
                    "workdir": args.workdir.as_posix(),
                    "local_audio_root": local_audio_root.as_posix(),
                    "local_roi_root": local_roi_root.as_posix(),
                    "local_run_dir": local_run_dir.as_posix(),
                    "local_tmp_dir": local_tmp_dir.as_posix(),
                    "disk_telemetry_roots": [path.as_posix() for path in disk_telemetry_roots],
                    "disk_telemetry_every_n_epochs": args.disk_telemetry_every_n_epochs,
                    "run_name": run_name,
                    "roi_format": str(args.roi_format),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "epochs": args.epochs,
                    "train_dataset_length": args.train_dataset_length,
                    "test_dataset_length": args.test_dataset_length,
                },
                indent=2,
            )
        )
        return

    aws = _require_aws_cli()
    if args.workdir.exists():
        shutil.rmtree(args.workdir, ignore_errors=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    local_inputs_dir.mkdir(parents=True, exist_ok=True)
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    args.spec_cache_dir.mkdir(parents=True, exist_ok=True)
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[env_name] = local_tmp_dir.as_posix()
    _record_disk_telemetry(
        local_disk_telemetry_dir,
        "after_setup",
        disk_telemetry_roots,
        extra={"run_name": run_name},
    )

    status = "failed"
    summary: dict = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "s3_run_root": s3_run_root,
        "manifest_s3_uri": str(args.manifest_s3_uri),
        "config_s3_uri": str(args.config_s3_uri),
        "s3_audio_root": str(args.s3_audio_root),
        "s3_roi_root": str(args.s3_roi_root),
        "roi_format": str(args.roi_format),
        "disk_telemetry_roots": [path.as_posix() for path in disk_telemetry_roots],
        "disk_telemetry_every_n_epochs": int(args.disk_telemetry_every_n_epochs),
    }

    try:
        _run([aws, "s3", "cp", str(args.manifest_s3_uri), local_manifest_path.as_posix(), "--only-show-errors"])
        _run([aws, "s3", "cp", str(args.config_s3_uri), local_config_path.as_posix(), "--only-show-errors"])
        manifest = _load_json(local_manifest_path)
        entries = _unique_entries(manifest)
        total_files = _sum_manifest_files(manifest)
        summary["directories_total"] = int(len(entries))
        summary["manifest_files_total"] = int(total_files)

        audio_patterns = ["*.wav", "*.WAV"]
        roi_patterns = [str(args.roi_parquet_name)] if str(args.roi_format) == "parquet" else ["*.txt"]
        download_results: list[dict] = []
        start_download = time.time()
        with ThreadPoolExecutor(max_workers=int(args.download_jobs)) as pool:
            futures = []
            for entry in entries:
                rel = str(entry["audio_dir_rel"])
                futures.append(
                    pool.submit(
                        _download_one_dir,
                        aws,
                        s3_src=_join_s3_uri(str(args.s3_audio_root), rel),
                        local_dst=local_audio_root / rel,
                        include_patterns=audio_patterns,
                        quiet=True,
                    )
                )
                futures.append(
                    pool.submit(
                        _download_one_dir,
                        aws,
                        s3_src=_join_s3_uri(str(args.s3_roi_root), rel),
                        local_dst=local_roi_root / rel,
                        include_patterns=roi_patterns,
                        quiet=True,
                    )
                )
            for fut in as_completed(futures):
                download_results.append(fut.result())

        audio_download_failures = [
            item for item in download_results
            if item["status"] != "ok" and "/audio/" in item["local_dst"]
        ]
        roi_download_failures = [
            item for item in download_results
            if item["status"] != "ok" and "/roi/" in item["local_dst"]
        ]
        summary["download_elapsed_sec"] = float(time.time() - start_download)
        summary["audio_download_failures"] = int(len(audio_download_failures))
        summary["roi_download_failures"] = int(len(roi_download_failures))
        _write_json(local_download_summary_path, {"results": download_results})
        _record_disk_telemetry(
            local_disk_telemetry_dir,
            "after_downloads",
            disk_telemetry_roots,
            extra={
                "audio_download_failures": int(len(audio_download_failures)),
                "roi_download_failures": int(len(roi_download_failures)),
            },
        )
        if audio_download_failures or roi_download_failures:
            raise RuntimeError("Audio/ROI directory download failed for one or more entries.")

        coverage_cmd = [
            sys.executable,
            "scripts/report_birdsong_roi_coverage.py",
            "--manifest",
            local_manifest_path.as_posix(),
            "--split",
            "all",
            "--audio-root",
            local_audio_root.as_posix(),
            "--roi-root",
            local_roi_root.as_posix(),
            "--roi-format",
            str(args.roi_format),
            "--roi-parquet-name",
            str(args.roi_parquet_name),
            "--out-dir",
            local_coverage_dir.as_posix(),
        ]
        _stream_to_log(coverage_cmd, local_coverage_log_path)
        coverage_summary = _load_json(local_coverage_dir / "summary.json")
        per_directory = _load_json(local_coverage_dir / "per_directory.json")
        wav_files_checked = int(sum(int(item.get("wav_files_checked") or 0) for item in per_directory))
        empty_files = int(coverage_summary.get("empty_roi_files") or 0)
        empty_fraction = float(empty_files / wav_files_checked) if wav_files_checked else 0.0
        summary["coverage_summary"] = coverage_summary
        summary["coverage_wav_files_checked"] = wav_files_checked
        summary["coverage_empty_fraction"] = empty_fraction
        _record_disk_telemetry(
            local_disk_telemetry_dir,
            "after_coverage",
            disk_telemetry_roots,
            extra={"coverage_empty_fraction": empty_fraction},
        )
        if (
            int(coverage_summary.get("missing_roi_dirs") or 0)
            or int(coverage_summary.get("missing_roi_files") or 0)
            or int(coverage_summary.get("roi_parse_errors") or 0)
        ):
            raise RuntimeError("Coverage report found missing or invalid ROI outputs.")
        if empty_fraction > float(args.max_empty_fraction):
            raise RuntimeError(
                f"Coverage empty ROI fraction {empty_fraction:.4f} exceeds threshold {float(args.max_empty_fraction):.4f}."
            )

        train_cmd = [
            sys.executable,
            "scripts/launch_birdsong_training.py",
            "--manifest",
            local_manifest_path.as_posix(),
            "--config",
            local_config_path.as_posix(),
            "--save-dir",
            local_run_dir.as_posix(),
            "--audio-root",
            local_audio_root.as_posix(),
            "--roi-root",
            local_roi_root.as_posix(),
            "--streaming",
            "--roi-format",
            str(args.roi_format),
            "--roi-parquet-name",
            str(args.roi_parquet_name),
            "--spec-cache-dir",
            args.spec_cache_dir.as_posix(),
            "--preflight-sample-dirs",
            str(args.preflight_sample_dirs),
            "--preflight-sample-segments",
            str(args.preflight_sample_segments),
            "--preflight-seed",
            str(args.preflight_seed),
        ]
        if args.batch_size is not None:
            train_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.num_workers is not None:
            train_cmd.extend(["--num-workers", str(args.num_workers)])
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.train_dataset_length is not None:
            train_cmd.extend(["--train-dataset-length", str(args.train_dataset_length)])
        if args.test_dataset_length is not None:
            train_cmd.extend(["--test-dataset-length", str(args.test_dataset_length)])
        if args.trainer_kwargs_json:
            train_cmd.extend(["--trainer-kwargs-json", str(args.trainer_kwargs_json)])
        train_cmd.extend([
            "--disk-telemetry-every-n-epochs",
            str(args.disk_telemetry_every_n_epochs),
        ])
        for root in disk_telemetry_roots:
            train_cmd.extend(["--disk-telemetry-root", root.as_posix()])

        _record_disk_telemetry(
            local_disk_telemetry_dir,
            "before_training",
            disk_telemetry_roots,
        )
        _stream_to_log(train_cmd, local_training_log_path)
        status = "completed"
        summary["training_run_dir"] = local_run_dir.as_posix()
    except Exception as exc:
        summary["error"] = str(exc)
        _record_disk_telemetry(
            local_disk_telemetry_dir,
            "exception",
            disk_telemetry_roots,
            extra={"error": str(exc)},
        )
        raise
    finally:
        summary["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["status"] = status
        _record_disk_telemetry(
            local_disk_telemetry_dir,
            f"final_{status}",
            disk_telemetry_roots,
        )
        _write_json(local_summary_path, summary)
        try:
            _sync_path_to_s3(aws, local_summary_path, _join_s3_uri(s3_run_root, "job_summary.json"))
            _sync_path_to_s3(aws, local_manifest_path, _join_s3_uri(s3_run_root, "inputs/manifest.json"))
            _sync_path_to_s3(aws, local_config_path, _join_s3_uri(s3_run_root, "inputs/config.yaml"))
            _sync_path_to_s3(aws, local_download_summary_path, _join_s3_uri(s3_run_root, "logs/download_summary.json"))
            _sync_path_to_s3(aws, local_coverage_log_path, _join_s3_uri(s3_run_root, "logs/coverage_stdout.log"))
            _sync_path_to_s3(aws, local_training_log_path, _join_s3_uri(s3_run_root, "logs/training_stdout.log"))
            _sync_path_to_s3(aws, local_disk_telemetry_dir, _join_s3_uri(s3_run_root, "logs/disk_telemetry"))
            _sync_path_to_s3(aws, local_coverage_dir, _join_s3_uri(s3_run_root, "coverage"))
            _sync_path_to_s3(aws, local_run_dir, _join_s3_uri(s3_run_root, "training_run"))
        except Exception as upload_exc:
            print(f"Warning: failed to upload one or more outputs to S3: {upload_exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
