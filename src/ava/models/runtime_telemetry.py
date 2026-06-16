"""Runtime telemetry callbacks for long-running training jobs."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - optional outside training environments
    pl = None

if pl is not None:
    _LightningCallbackBase = pl.Callback
else:  # pragma: no cover - only used when Lightning is unavailable
    class _LightningCallbackBase:
        pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_optional(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _read_proc_stat() -> Optional[tuple[int, int]]:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            first = handle.readline().strip().split()
    except OSError:
        return None
    if not first or first[0] != "cpu":
        return None
    try:
        values = [int(part) for part in first[1:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def _read_meminfo() -> dict[str, Optional[float]]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            rows = handle.readlines()
    except OSError:
        return {"host_mem_total_mib": None, "host_mem_used_percent": None}
    parsed: dict[str, float] = {}
    for row in rows:
        parts = row.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(":")
        try:
            parsed[key] = float(parts[1])
        except ValueError:
            continue
    total_kib = parsed.get("MemTotal")
    available_kib = parsed.get("MemAvailable", parsed.get("MemFree"))
    if not total_kib or available_kib is None:
        return {"host_mem_total_mib": None, "host_mem_used_percent": None}
    used_percent = max(0.0, min(100.0, 100.0 * (1.0 - available_kib / total_kib)))
    return {
        "host_mem_total_mib": total_kib / 1024.0,
        "host_mem_used_percent": used_percent,
    }


def _read_process_rss_mib() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for row in handle:
                if row.startswith("VmRSS:"):
                    parts = row.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _read_process_io() -> Optional[dict[str, int]]:
    try:
        with open("/proc/self/io", "r", encoding="utf-8") as handle:
            rows = handle.readlines()
    except OSError:
        return None
    parsed: dict[str, int] = {}
    for row in rows:
        key, _, value = row.partition(":")
        if not key or not value:
            continue
        try:
            parsed[key.strip()] = int(value.strip())
        except ValueError:
            continue
    if "read_bytes" not in parsed and "write_bytes" not in parsed:
        return None
    return {
        "read_bytes": int(parsed.get("read_bytes", 0)),
        "write_bytes": int(parsed.get("write_bytes", 0)),
    }


def _read_network_bytes() -> Optional[dict[str, int]]:
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as handle:
            rows = handle.readlines()[2:]
    except OSError:
        return None
    rx = 0
    tx = 0
    found = False
    for row in rows:
        iface, _, payload = row.partition(":")
        if not payload:
            continue
        if iface.strip() == "lo":
            continue
        parts = payload.split()
        if len(parts) < 16:
            continue
        try:
            rx += int(parts[0])
            tx += int(parts[8])
            found = True
        except ValueError:
            continue
    if not found:
        return None
    return {"rx_bytes": rx, "tx_bytes": tx}


def _read_gpu_snapshot() -> dict[str, Optional[float] | str]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        util_values = []
        mem_used = 0.0
        mem_total = 0.0
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_values.append(float(utilization.gpu))
            mem_used += float(memory.used) / (1024.0 * 1024.0)
            mem_total += float(memory.total) / (1024.0 * 1024.0)
        return {
            "gpu_backend": "nvml",
            "gpu_backend_reason": "",
            "gpu_count": float(count),
            "gpu_util_percent_mean": (
                sum(util_values) / len(util_values) if util_values else None
            ),
            "gpu_mem_used_mib_sum": mem_used if count else None,
            "gpu_mem_total_mib_sum": mem_total if count else None,
        }
    except Exception as exc:
        nvml_reason = str(exc)

    proc = _run_optional(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc["returncode"] != 0:
        return {
            "gpu_backend": "none",
            "gpu_backend_reason": nvml_reason or (proc["stderr"] or proc["stdout"]),
            "gpu_count": 0.0,
            "gpu_util_percent_mean": None,
            "gpu_mem_used_mib_sum": None,
            "gpu_mem_total_mib_sum": None,
        }
    rows = [row.strip() for row in str(proc["stdout"]).splitlines() if row.strip()]
    util_values = []
    mem_used = []
    mem_total = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 3:
            continue
        util = _parse_float(parts[0])
        used = _parse_float(parts[1])
        total = _parse_float(parts[2])
        if util is not None:
            util_values.append(util)
        if used is not None:
            mem_used.append(used)
        if total is not None:
            mem_total.append(total)
    return {
        "gpu_backend": "nvidia-smi",
        "gpu_backend_reason": nvml_reason,
        "gpu_count": float(len(rows)),
        "gpu_util_percent_mean": (
            sum(util_values) / len(util_values) if util_values else None
        ),
        "gpu_mem_used_mib_sum": sum(mem_used) if mem_used else None,
        "gpu_mem_total_mib_sum": sum(mem_total) if mem_total else None,
    }


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _infer_batch_size(batch: Any) -> Optional[int]:
    shape = getattr(batch, "shape", None)
    if shape is not None and len(shape) > 0:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    if isinstance(batch, dict):
        for value in batch.values():
            inferred = _infer_batch_size(value)
            if inferred is not None:
                return inferred
    if isinstance(batch, (list, tuple)):
        for value in batch:
            inferred = _infer_batch_size(value)
            if inferred is not None:
                return inferred
    return None


def _format_optional(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


class RuntimeTelemetryCallback(_LightningCallbackBase):
    """Record resource samples and batch timing during Lightning training."""

    _RESOURCE_FIELDS = [
        "timestamp_utc",
        "stage",
        "epoch",
        "global_step",
        "dt_sec",
        "host_cpu_util_percent",
        "host_mem_used_percent",
        "host_mem_total_mib",
        "process_rss_mib",
        "process_read_mib_per_sec",
        "process_write_mib_per_sec",
        "network_rx_mib_per_sec",
        "network_tx_mib_per_sec",
        "gpu_backend",
        "gpu_backend_reason",
        "gpu_count",
        "gpu_util_percent_mean",
        "gpu_mem_used_mib_sum",
        "gpu_mem_total_mib_sum",
    ]
    _BATCH_FIELDS = [
        "timestamp_utc",
        "phase",
        "epoch",
        "batch_idx",
        "global_step",
        "batch_size",
        "data_wait_sec",
        "batch_compute_sec",
        "samples_per_sec",
    ]

    def __init__(
        self,
        save_dir: str,
        resource_interval_sec: float = 30.0,
        batch_log_every_n_batches: int = 50,
    ):
        if float(resource_interval_sec) <= 0:
            raise ValueError("resource_interval_sec must be positive.")
        if int(batch_log_every_n_batches) <= 0:
            raise ValueError("batch_log_every_n_batches must be positive.")
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.output_dir = self.save_dir / "runtime_telemetry"
        self.resource_interval_sec = float(resource_interval_sec)
        self.batch_log_every_n_batches = int(batch_log_every_n_batches)
        self.resource_path = self.output_dir / "resource_telemetry.csv"
        self.batch_path = self.output_dir / "batch_timings.csv"
        self.summary_path = self.output_dir / "summary.json"
        self._prev_resource_time: Optional[float] = None
        self._prev_cpu: Optional[tuple[int, int]] = None
        self._prev_io: Optional[dict[str, int]] = None
        self._prev_net: Optional[dict[str, int]] = None
        self._last_sample_time: Optional[float] = None
        self._batch_start: dict[str, float] = {}
        self._last_batch_end: dict[str, float] = {}
        self._current_data_wait: dict[str, Optional[float]] = {}
        self._batch_counts: dict[str, int] = {}
        self._batch_sums: dict[str, dict[str, float]] = {}
        self._resource_count = 0
        self._resource_sums: dict[str, float] = {}

    def _should_capture(self, trainer: Any) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _resource_snapshot(self, trainer: Any, stage: str) -> dict[str, Any]:
        now = time.monotonic()
        current_cpu = _read_proc_stat()
        current_io = _read_process_io()
        current_net = _read_network_bytes()
        dt = None if self._prev_resource_time is None else now - self._prev_resource_time
        row: dict[str, Any] = {
            "timestamp_utc": _utc_now_iso(),
            "stage": stage,
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "dt_sec": dt,
            "host_cpu_util_percent": None,
            "process_read_mib_per_sec": None,
            "process_write_mib_per_sec": None,
            "network_rx_mib_per_sec": None,
            "network_tx_mib_per_sec": None,
        }
        if current_cpu is not None and self._prev_cpu is not None:
            idle_delta = current_cpu[0] - self._prev_cpu[0]
            total_delta = current_cpu[1] - self._prev_cpu[1]
            if total_delta > 0:
                row["host_cpu_util_percent"] = 100.0 * (1.0 - idle_delta / total_delta)
        if current_io is not None and self._prev_io is not None and dt and dt > 0:
            row["process_read_mib_per_sec"] = (
                (current_io["read_bytes"] - self._prev_io["read_bytes"])
                / (1024.0 * 1024.0)
                / dt
            )
            row["process_write_mib_per_sec"] = (
                (current_io["write_bytes"] - self._prev_io["write_bytes"])
                / (1024.0 * 1024.0)
                / dt
            )
        if current_net is not None and self._prev_net is not None and dt and dt > 0:
            row["network_rx_mib_per_sec"] = (
                (current_net["rx_bytes"] - self._prev_net["rx_bytes"])
                / (1024.0 * 1024.0)
                / dt
            )
            row["network_tx_mib_per_sec"] = (
                (current_net["tx_bytes"] - self._prev_net["tx_bytes"])
                / (1024.0 * 1024.0)
                / dt
            )
        row.update(_read_meminfo())
        row["process_rss_mib"] = _read_process_rss_mib()
        row.update(_read_gpu_snapshot())
        self._prev_resource_time = now
        self._prev_cpu = current_cpu
        self._prev_io = current_io
        self._prev_net = current_net
        return row

    def _record_resource(self, trainer: Any, stage: str, *, force: bool = False) -> None:
        if not self._should_capture(trainer):
            return
        now = time.monotonic()
        if (
            not force
            and self._last_sample_time is not None
            and now - self._last_sample_time < self.resource_interval_sec
        ):
            return
        row = self._resource_snapshot(trainer, stage)
        self._last_sample_time = now
        _append_csv(self.resource_path, self._RESOURCE_FIELDS, row)
        self._resource_count += 1
        for key in (
            "host_cpu_util_percent",
            "host_mem_used_percent",
            "process_rss_mib",
            "gpu_util_percent_mean",
            "gpu_mem_used_mib_sum",
        ):
            value = row.get(key)
            if value is not None and value != "":
                self._resource_sums[key] = self._resource_sums.get(key, 0.0) + float(value)
        print(
            "[runtime-telemetry] "
            f"stage={stage} step={row.get('global_step')} "
            f"cpu={_format_optional(row.get('host_cpu_util_percent'), 1)}% "
            f"gpu={_format_optional(row.get('gpu_util_percent_mean'), 1)}% "
            f"gpu_mem={_format_optional(row.get('gpu_mem_used_mib_sum'), 1)}MiB "
            f"rss={_format_optional(row.get('process_rss_mib'), 1)}MiB",
            flush=True,
        )

    def _record_batch(
        self,
        trainer: Any,
        phase: str,
        batch: Any,
        batch_idx: int,
        ended_at: float,
    ) -> None:
        if not self._should_capture(trainer):
            return
        started_at = self._batch_start.get(phase)
        if started_at is None:
            return
        compute_sec = max(0.0, ended_at - started_at)
        batch_size = _infer_batch_size(batch)
        samples_per_sec = (
            float(batch_size) / compute_sec
            if batch_size is not None and compute_sec > 0
            else None
        )
        row = {
            "timestamp_utc": _utc_now_iso(),
            "phase": phase,
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "batch_idx": int(batch_idx),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "batch_size": batch_size,
            "data_wait_sec": self._current_data_wait.get(phase),
            "batch_compute_sec": compute_sec,
            "samples_per_sec": samples_per_sec,
        }
        _append_csv(self.batch_path, self._BATCH_FIELDS, row)
        counts = self._batch_counts
        counts[phase] = counts.get(phase, 0) + 1
        sums = self._batch_sums.setdefault(
            phase,
            {"data_wait_sec": 0.0, "batch_compute_sec": 0.0, "samples_per_sec": 0.0},
        )
        for key in ("data_wait_sec", "batch_compute_sec", "samples_per_sec"):
            value = row.get(key)
            if value is not None and value != "":
                sums[key] = sums.get(key, 0.0) + float(value)
        if counts[phase] % self.batch_log_every_n_batches == 0:
            print(
                "[batch-telemetry] "
                f"phase={phase} epoch={row['epoch']} batch={batch_idx} "
                f"step={row['global_step']} "
                f"data_wait={_format_optional(row.get('data_wait_sec'))}s "
                f"compute={_format_optional(compute_sec)}s "
                f"samples_per_sec={_format_optional(samples_per_sec)}",
                flush=True,
            )
        self._last_batch_end[phase] = ended_at

    def _start_batch(self, trainer: Any, phase: str) -> None:
        if not self._should_capture(trainer):
            return
        if getattr(trainer, "sanity_checking", False):
            return
        now = time.monotonic()
        previous_end = self._last_batch_end.get(phase)
        self._current_data_wait[phase] = (
            None if previous_end is None else max(0.0, now - previous_end)
        )
        self._batch_start[phase] = now

    def _write_summary(self) -> None:
        batch_summary = {}
        for phase, count in self._batch_counts.items():
            sums = self._batch_sums.get(phase, {})
            batch_summary[phase] = {
                "batch_count": int(count),
                "mean_data_wait_sec": (
                    sums.get("data_wait_sec", 0.0) / count if count else None
                ),
                "mean_batch_compute_sec": (
                    sums.get("batch_compute_sec", 0.0) / count if count else None
                ),
                "mean_samples_per_sec": (
                    sums.get("samples_per_sec", 0.0) / count if count else None
                ),
            }
        resource_summary = {
            "sample_count": int(self._resource_count),
        }
        if self._resource_count:
            for key, value in self._resource_sums.items():
                resource_summary[f"mean_{key}"] = value / self._resource_count
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(
                {
                    "batch_summary": batch_summary,
                    "resource_summary": resource_summary,
                    "batch_timings_csv": self.batch_path.as_posix(),
                    "resource_telemetry_csv": self.resource_path.as_posix(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        self._record_resource(trainer, "fit_start", force=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._start_batch(trainer, "train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        ended_at = time.monotonic()
        self._record_batch(trainer, "train", batch, int(batch_idx), ended_at)
        self._record_resource(trainer, "train_batch_end")

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self._start_batch(trainer, "validation")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        ended_at = time.monotonic()
        self._record_batch(trainer, "validation", batch, int(batch_idx), ended_at)
        self._record_resource(trainer, "validation_batch_end")

    def on_exception(self, trainer, pl_module, exception) -> None:  # pragma: no cover - live-training guardrail
        self._record_resource(trainer, "exception", force=True)
        self._write_summary()

    def on_fit_end(self, trainer, pl_module) -> None:
        self._record_resource(trainer, "fit_end", force=True)
        self._write_summary()
