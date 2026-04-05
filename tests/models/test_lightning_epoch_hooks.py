import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset


def _ensure_pytorch_lightning_importable() -> None:
    try:
        import pytorch_lightning  # noqa: F401
        return
    except ImportError:
        pass

    pl_module = types.ModuleType("pytorch_lightning")

    class _LightningModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def save_hyperparameters(self, *args, **kwargs):
            return None

    class _Trainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _Callback:
        pass

    loggers_module = types.ModuleType("pytorch_lightning.loggers")

    class _TensorBoardLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    pl_module.LightningModule = _LightningModule
    pl_module.Trainer = _Trainer
    pl_module.Callback = _Callback
    loggers_module.TensorBoardLogger = _TensorBoardLogger
    sys.modules["pytorch_lightning"] = pl_module
    sys.modules["pytorch_lightning.loggers"] = loggers_module


def _load_lightning_vae_module():
    _ensure_pytorch_lightning_importable()
    sys.modules.pop("ava.models.lightning_vae", None)
    return importlib.import_module("ava.models.lightning_vae")


class _EpochTrackingDataset(Dataset):
    def __init__(self, data):
        self._data = data
        self.epochs = []

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]

    def set_epoch(self, epoch):
        self.epochs.append(int(epoch))


def test_lightning_train_epoch_hook_updates_loader_dataset_epoch():
    lightning_vae = _load_lightning_vae_module()
    dataset = _EpochTrackingDataset(torch.randn(4, 64, 64))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    fake_module = types.SimpleNamespace(
        trainer=types.SimpleNamespace(train_dataloader=loader),
        current_epoch=3,
    )

    lightning_vae.VAELightningModule.on_train_epoch_start(fake_module)

    assert dataset.epochs == [3]


def test_lightning_validation_epoch_hook_updates_all_val_datasets():
    lightning_vae = _load_lightning_vae_module()
    dataset_a = _EpochTrackingDataset(torch.randn(4, 64, 64))
    dataset_b = _EpochTrackingDataset(torch.randn(4, 64, 64))
    loaders = [
        DataLoader(dataset_a, batch_size=2, shuffle=False, num_workers=0),
        DataLoader(dataset_b, batch_size=2, shuffle=False, num_workers=0),
    ]
    fake_module = types.SimpleNamespace(
        trainer=types.SimpleNamespace(val_dataloaders=loaders),
        current_epoch=5,
    )

    lightning_vae.VAELightningModule.on_validation_epoch_start(fake_module)

    assert dataset_a.epochs == [5]
    assert dataset_b.epochs == [5]
