import json
import types
from pathlib import Path

from ava.models.disk_telemetry import DiskTelemetryCallback, collect_disk_telemetry


def test_collect_disk_telemetry_reports_existing_root(tmp_path: Path):
    root = tmp_path / "sample"
    root.mkdir()
    (root / "clip.txt").write_text("abc", encoding="utf-8")

    payload = collect_disk_telemetry([root])

    assert payload["df_h"]
    assert len(payload["roots"]) == 1
    entry = payload["roots"][0]
    assert entry["path"] == root.as_posix()
    assert entry["exists"] is True
    assert entry["du_human"] is not None
    assert entry["du_kib"] is not None
    assert entry["disk_usage"]["free_bytes"] >= 0


def test_disk_telemetry_callback_writes_snapshots(tmp_path: Path):
    root = tmp_path / "sample"
    root.mkdir()
    (root / "clip.txt").write_text("abc", encoding="utf-8")
    callback = DiskTelemetryCallback(
        save_dir=tmp_path.as_posix(),
        roots=[root],
        every_n_epochs=2,
    )
    trainer = types.SimpleNamespace(is_global_zero=True, current_epoch=0, global_step=0)

    callback.on_fit_start(trainer, object())
    trainer.current_epoch = 1
    trainer.global_step = 25
    callback.on_train_epoch_end(trainer, object())

    snapshot_path = tmp_path / "disk_telemetry" / "snapshots.jsonl"
    assert snapshot_path.exists()
    entries = [
        json.loads(line)
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["stage"] for entry in entries] == ["fit_start", "train_epoch_end"]
    assert entries[1]["epoch"] == 2
