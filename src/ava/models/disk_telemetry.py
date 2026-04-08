"""Disk telemetry helpers for long-running training jobs."""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

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


def _humanize_bytes(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PiB"


def _nearest_existing_path(path: Path) -> Optional[Path]:
    current = path
    while True:
        if current.exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _run_optional(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _normalize_roots(roots: Iterable[str | Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        path = Path(root)
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _collect_root_usage(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": path.as_posix(),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
    }
    probe = _nearest_existing_path(path)
    if probe is not None:
        total, used, free = shutil.disk_usage(probe)
        payload["disk_usage"] = {
            "probe_path": probe.as_posix(),
            "total_bytes": int(total),
            "used_bytes": int(used),
            "free_bytes": int(free),
            "total_human": _humanize_bytes(int(total)),
            "used_human": _humanize_bytes(int(used)),
            "free_human": _humanize_bytes(int(free)),
        }
    else:
        payload["disk_usage"] = None

    if not path.exists():
        payload["du_human"] = None
        payload["du_kib"] = None
        payload["du_error"] = "missing"
        return payload

    du_h = _run_optional(["du", "-sh", path.as_posix()])
    du_k = _run_optional(["du", "-sk", path.as_posix()])
    if du_h["returncode"] == 0 and du_h["stdout"].strip():
        payload["du_human"] = du_h["stdout"].strip().split()[0]
    else:
        payload["du_human"] = None
    if du_k["returncode"] == 0 and du_k["stdout"].strip():
        try:
            payload["du_kib"] = int(du_k["stdout"].strip().split()[0])
        except ValueError:
            payload["du_kib"] = None
    else:
        payload["du_kib"] = None
    error_parts = []
    if du_h["returncode"] not in (None, 0):
        error_parts.append((du_h["stderr"] or du_h["stdout"] or "").strip())
    if du_k["returncode"] not in (None, 0):
        error_parts.append((du_k["stderr"] or du_k["stdout"] or "").strip())
    payload["du_error"] = "; ".join(part for part in error_parts if part) or None
    return payload


def collect_disk_telemetry(roots: Iterable[str | Path]) -> dict[str, Any]:
    normalized = _normalize_roots(roots)
    df_h = _run_optional(["df", "-h"])
    snapshot = {
        "captured_at_utc": _utc_now_iso(),
        "roots": [_collect_root_usage(path) for path in normalized],
        "df_h": df_h["stdout"],
        "df_h_error": None if df_h["returncode"] == 0 else (df_h["stderr"] or df_h["stdout"] or "").strip(),
    }
    return snapshot


def append_disk_telemetry_snapshot(output_dir: str | Path, payload: dict[str, Any]) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "snapshots.jsonl"
    latest_path = output_path / "latest.json"
    sequence = 1
    if jsonl_path.exists():
        try:
            sequence = len(jsonl_path.read_text(encoding="utf-8").splitlines()) + 1
        except OSError:
            sequence = 1
    snapshot = dict(payload)
    snapshot.setdefault("snapshot_index", sequence)
    line = json.dumps(snapshot, sort_keys=True)
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    latest_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return {"jsonl": jsonl_path, "latest": latest_path}


def format_disk_telemetry_summary(payload: dict[str, Any]) -> str:
    stage = str(payload.get("stage") or "snapshot")
    epoch = payload.get("epoch")
    prefix = f"[disk-telemetry] stage={stage}"
    if epoch is not None:
        prefix += f" epoch={epoch}"
    pieces = []
    for root in payload.get("roots", []):
        path = str(root.get("path"))
        label = path.rstrip("/").split("/")[-1] or path
        disk_usage = root.get("disk_usage") or {}
        free_human = disk_usage.get("free_human")
        du_human = root.get("du_human")
        piece = f"{label}:"
        details = []
        if du_human is not None:
            details.append(f"du={du_human}")
        if free_human is not None:
            details.append(f"free={free_human}")
        if root.get("du_error"):
            details.append(f"du_error={root['du_error']}")
        piece += ",".join(details) if details else "n/a"
        pieces.append(piece)
    return prefix + " " + " | ".join(pieces)


class DiskTelemetryCallback(_LightningCallbackBase):
    """Record disk telemetry snapshots at fit start and every N epochs."""

    def __init__(self, save_dir: str, roots: Iterable[str | Path], every_n_epochs: int = 5):
        if int(every_n_epochs) <= 0:
            raise ValueError("every_n_epochs must be positive.")
        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.output_dir = self.save_dir / "disk_telemetry"
        self.roots = _normalize_roots(roots)
        self.every_n_epochs = int(every_n_epochs)

    def _should_capture(self, trainer: Any) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _capture(self, trainer: Any, stage: str, epoch: Optional[int] = None, extra: Optional[dict[str, Any]] = None) -> None:
        if not self._should_capture(trainer):
            return
        payload = collect_disk_telemetry(self.roots)
        payload["stage"] = stage
        if epoch is not None:
            payload["epoch"] = int(epoch)
        global_step = getattr(trainer, "global_step", None)
        if global_step is not None:
            payload["global_step"] = int(global_step)
        if extra:
            payload.update(extra)
        append_disk_telemetry_snapshot(self.output_dir, payload)
        print(format_disk_telemetry_summary(payload), flush=True)

    def on_fit_start(self, trainer, pl_module) -> None:
        self._capture(trainer, "fit_start", epoch=int(getattr(trainer, "current_epoch", 0)))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        if epoch % self.every_n_epochs == 0:
            self._capture(trainer, "train_epoch_end", epoch=epoch)

    def on_exception(self, trainer, pl_module, exception) -> None:  # pragma: no cover - exercised in live training
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        self._capture(trainer, "exception", epoch=epoch, extra={"error": str(exception)})

    def on_fit_end(self, trainer, pl_module) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        self._capture(trainer, "fit_end", epoch=epoch)
