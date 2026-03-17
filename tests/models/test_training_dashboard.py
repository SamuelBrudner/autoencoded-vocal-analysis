import json

import numpy as np
import pytest

from ava.models.torch_onnx_compat import patch_torch_onnx_exporter


torch = pytest.importorskip("torch")
patch_torch_onnx_exporter()

from torch.utils.data import DataLoader, Dataset

from ava.models.training_dashboard import (
    backfill_training_dashboard,
    build_dashboard_payload,
    write_training_dashboard,
)


class _TensorDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


def _make_loaders(data, batch_size=2):
    dataset = _TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return {"train": loader, "test": loader}


def test_write_training_dashboard_outputs_html_and_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"git_commit": "abc123", "dataset_root": "/tmp/data"}),
        encoding="utf-8",
    )
    payload = build_dashboard_payload(
        save_dir=run_dir.as_posix(),
        history={
            "train_loss": [{"epoch": 1, "step": 8, "value": 10.0}],
            "val_loss": [{"epoch": 1, "step": 8, "value": 12.5}],
        },
        status="running",
        started_at="2026-03-17T10:00:00+00:00",
        max_epochs=10,
        current_epoch=1,
        latest_step=8,
        device="cpu",
    )

    json_path, html_path = write_training_dashboard(run_dir.as_posix(), payload)

    assert run_dir.joinpath("training_dashboard.json").as_posix() == json_path
    assert run_dir.joinpath("training_dashboard.html").as_posix() == html_path
    loaded = json.loads(run_dir.joinpath("training_dashboard.json").read_text(encoding="utf-8"))
    assert loaded["status"] == "running"
    assert loaded["summary"]["latest_metrics"]["train_loss"] == pytest.approx(10.0)
    html_text = run_dir.joinpath("training_dashboard.html").read_text(encoding="utf-8")
    assert "AVA Training Dashboard" in html_text
    assert "Latest Val Loss" in html_text


def test_backfill_training_dashboard_from_tensorboard_logs(tmp_path):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    run_dir = tmp_path / "backfill_run"
    log_dir = run_dir / "lightning_logs" / "version_0"
    log_dir.mkdir(parents=True)
    writer = SummaryWriter(log_dir.as_posix())
    writer.add_scalar("epoch", 0, 4)
    writer.add_scalar("train_loss", 11.0, 4)
    writer.add_scalar("val_loss", 13.0, 4)
    writer.add_scalar("epoch", 1, 8)
    writer.add_scalar("train_loss", 7.5, 8)
    writer.add_scalar("val_loss", 9.0, 8)
    writer.flush()
    writer.close()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"git_commit": "def456", "config_path": "config.yaml"}),
        encoding="utf-8",
    )

    payload, json_path, html_path = backfill_training_dashboard(
        run_dir.as_posix(),
        status="completed",
    )

    assert payload["status"] == "completed"
    assert payload["history"]["train_loss"][-1]["value"] == pytest.approx(7.5)
    assert payload["history"]["train_loss"][-1]["epoch"] == 2
    assert payload["summary"]["best_metrics"]["val_loss"]["value"] == pytest.approx(9.0)
    assert json_path.endswith("training_dashboard.json")
    assert html_path.endswith("training_dashboard.html")


def test_train_vae_writes_training_dashboard(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from ava.models.lightning_vae import train_vae

    data = torch.from_numpy(np.random.randn(4, 128, 128).astype(np.float32))
    loaders = _make_loaders(data)
    save_dir = tmp_path / "training_run"

    train_vae(
        loaders,
        save_dir=save_dir.as_posix(),
        epochs=1,
        test_freq=1,
        save_freq=None,
        vis_freq=None,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
    )

    dashboard_path = save_dir / "training_dashboard.json"
    assert dashboard_path.exists()
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["summary"]["latest_metrics"]["train_loss"] > 0
    assert "val_loss" in payload["summary"]["latest_metrics"]
