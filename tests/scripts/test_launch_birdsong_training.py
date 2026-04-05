import importlib.util
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")


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


def _load_launch_module():
    _ensure_pytorch_lightning_importable()
    sys.modules.pop("ava.models.lightning_vae", None)
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "launch_birdsong_training.py"
    )
    spec = importlib.util.spec_from_file_location(
        "launch_birdsong_training",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_streaming_dataset_lengths_defaults_to_shared_length():
    module = _load_launch_module()
    assert module._resolve_streaming_dataset_lengths(2048, None, None) == (2048, 2048)


def test_resolve_streaming_dataset_lengths_accepts_split_overrides():
    module = _load_launch_module()
    assert module._resolve_streaming_dataset_lengths(2048, 131072, 512) == (
        131072,
        512,
    )


def test_resolve_streaming_dataset_lengths_rejects_non_positive_values():
    module = _load_launch_module()
    with pytest.raises(ValueError):
        module._resolve_streaming_dataset_lengths(2048, 0, 256)
    with pytest.raises(ValueError):
        module._resolve_streaming_dataset_lengths(2048, 1024, -1)
