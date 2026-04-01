import json

import numpy as np
import pytest

from ava.models.torch_onnx_compat import patch_torch_onnx_exporter

torch = pytest.importorskip("torch")
patch_torch_onnx_exporter()

from torch.utils.data import DataLoader, Dataset

from ava.models.fixed_window_config import FixedWindowTrainConfig
from ava.models.vrnn import VRNN


class _SequenceDataset(Dataset):
    def __init__(self, x, mask, start_times=None):
        self.x = x
        self.mask = mask
        self.start_times = start_times

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.x[idx], "mask": self.mask[idx]}
        if self.start_times is not None:
            item["start_times"] = self.start_times[idx]
        return item


def _make_loaders(x, mask, batch_size=2, start_times=None):
    dataset = _SequenceDataset(x, mask, start_times=start_times)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return {"train": loader, "test": loader}


def test_vrnn_forward_and_loss_respect_mask():
    x = torch.rand(2, 5, 4, 3)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )
    model = VRNN(
        save_dir="",
        device_name="cpu",
        input_shape=(4, 3),
        z_dim=5,
        hidden_dim=16,
        x_feature_dim=12,
        z_feature_dim=8,
    )

    outputs = model(x, mask=mask, sample_posterior=False)
    loss, stats = model.compute_loss(x, mask=mask, kl_beta=0.25)

    assert outputs["recon"].shape == x.shape
    assert outputs["posterior_mu"].shape == (2, 5, 5)
    assert torch.isfinite(loss).item()
    assert torch.isfinite(stats["recon_mse"]).item()
    assert torch.isfinite(stats["kl"]).item()
    assert stats["sequence_length_mean"].item() == pytest.approx(3.5)
    assert stats["oscillator_loss"].item() == pytest.approx(0.0)


def test_sequence_train_config_exposes_opt_in_oscillator_kwargs():
    config = FixedWindowTrainConfig(
        oscillator_weight=0.25,
        oscillator_frequency_hz=8.0,
        oscillator_start_dim=2,
        oscillator_radius_weight=0.1,
        oscillator_radius_target=0.75,
        oscillator_center_weight=0.05,
    )

    kwargs = config.to_sequence_train_kwargs()

    assert kwargs["oscillator_weight"] == pytest.approx(0.25)
    assert kwargs["oscillator_frequency_hz"] == pytest.approx(8.0)
    assert kwargs["oscillator_start_dim"] == 2
    assert kwargs["oscillator_radius_weight"] == pytest.approx(0.1)
    assert kwargs["oscillator_radius_target"] == pytest.approx(0.75)
    assert kwargs["oscillator_center_weight"] == pytest.approx(0.05)


def test_vrnn_oscillator_regularizer_is_opt_in():
    x = torch.rand(2, 5, 4, 3)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )
    start_times = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.0, 0.1, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = VRNN(
        save_dir="",
        device_name="cpu",
        input_shape=(4, 3),
        z_dim=5,
        hidden_dim=16,
        x_feature_dim=12,
        z_feature_dim=8,
        oscillator_weight=0.0,
        oscillator_frequency_hz=5.0,
        sequence_hop_length_sec=0.1,
    )

    torch.manual_seed(0)
    loss_off, stats_off = model.compute_loss(
        x,
        mask=mask,
        start_times=start_times,
        kl_beta=0.25,
    )

    model.oscillator_weight = 0.5
    torch.manual_seed(0)
    loss_on, stats_on = model.compute_loss(
        x,
        mask=mask,
        start_times=start_times,
        kl_beta=0.25,
    )

    assert stats_off["oscillator_loss"].item() == pytest.approx(0.0)
    assert stats_on["oscillator_loss"].item() > 0.0
    assert loss_on.item() > loss_off.item()


def test_vrnn_checkpoint_roundtrip(tmp_path):
    model = VRNN(
        save_dir=tmp_path.as_posix(),
        device_name="cpu",
        input_shape=(4, 3),
        z_dim=4,
        hidden_dim=12,
        x_feature_dim=10,
        z_feature_dim=6,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)
    model.loss["train"][0] = 1.23
    model.epoch = 2
    model.save_state("checkpoint_002.tar")

    loaded = VRNN(
        save_dir=tmp_path.as_posix(),
        device_name="cpu",
        input_shape=(4, 3),
        z_dim=4,
        hidden_dim=12,
        x_feature_dim=10,
        z_feature_dim=6,
    )
    loaded.load_state((tmp_path / "checkpoint_002.tar").as_posix())

    for name, value in model.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[name])
    assert loaded.loss["train"][0] == pytest.approx(1.23)
    assert loaded.epoch == 2


def test_train_sequence_vae_writes_dashboard_and_checkpoint(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from ava.models.lightning_sequence_vae import train_sequence_vae

    torch.manual_seed(0)
    x = torch.rand(4, 3, 6, 5)
    mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, True, True],
            [True, False, False],
        ]
    )
    start_times = torch.tensor(
        [
            [0.0, 0.05, 0.10],
            [0.0, 0.05, 0.10],
            [0.0, 0.05, 0.10],
            [0.0, 0.05, 0.10],
        ],
        dtype=torch.float32,
    )
    loaders = _make_loaders(x, mask, start_times=start_times)
    save_dir = tmp_path / "sequence_training"

    train_sequence_vae(
        loaders,
        save_dir=save_dir.as_posix(),
        input_shape=(6, 5),
        epochs=1,
        test_freq=1,
        save_freq=1,
        z_dim=4,
        hidden_dim=16,
        x_feature_dim=12,
        z_feature_dim=8,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
    )

    dashboard_path = save_dir / "training_dashboard.json"
    checkpoint_path = save_dir / "checkpoint_001.tar"
    assert dashboard_path.exists()
    assert checkpoint_path.exists()
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["summary"]["latest_metrics"]["train_loss"] > 0
