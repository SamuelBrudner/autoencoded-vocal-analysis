import csv
import json
import time
import types
from pathlib import Path

import numpy as np

from ava.models.runtime_telemetry import RuntimeTelemetryCallback


def test_runtime_telemetry_callback_writes_batch_and_resource_files(tmp_path: Path):
    callback = RuntimeTelemetryCallback(
        save_dir=tmp_path.as_posix(),
        resource_interval_sec=0.01,
        batch_log_every_n_batches=1,
    )
    trainer = types.SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        global_step=0,
        sanity_checking=False,
    )
    batch = np.zeros((4, 8), dtype=np.float32)

    callback.on_fit_start(trainer, object())
    callback.on_train_batch_start(trainer, object(), batch, 0)
    time.sleep(0.02)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, object(), None, batch, 0)
    callback.on_fit_end(trainer, object())

    batch_path = tmp_path / "runtime_telemetry" / "batch_timings.csv"
    resource_path = tmp_path / "runtime_telemetry" / "resource_telemetry.csv"
    summary_path = tmp_path / "runtime_telemetry" / "summary.json"

    assert batch_path.exists()
    assert resource_path.exists()
    assert summary_path.exists()

    with batch_path.open("r", encoding="utf-8", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))
    assert len(batch_rows) == 1
    assert batch_rows[0]["phase"] == "train"
    assert batch_rows[0]["batch_size"] == "4"
    assert float(batch_rows[0]["batch_compute_sec"]) > 0

    with resource_path.open("r", encoding="utf-8", newline="") as handle:
        resource_rows = list(csv.DictReader(handle))
    assert resource_rows
    assert "gpu_backend" in resource_rows[0]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["batch_summary"]["train"]["batch_count"] == 1
