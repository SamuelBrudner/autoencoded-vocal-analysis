import importlib.util
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "cloud"
        / "aws"
        / "run_birdsong_training_batch_job.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_birdsong_training_batch_job",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extend_spec_cache_args_can_disable_cache():
    module = _load_module()
    cmd = ["python", "scripts/launch_birdsong_training.py"]

    module._extend_spec_cache_args(
        cmd,
        disable_spec_cache=True,
        spec_cache_dir=Path("/mnt/ava_cache/spec_cache"),
    )

    assert cmd[-1] == "--disable-spec-cache"
    assert "--spec-cache-dir" not in cmd


def test_extend_spec_cache_args_can_set_cache_dir():
    module = _load_module()
    cmd = ["python", "scripts/launch_birdsong_training.py"]

    module._extend_spec_cache_args(
        cmd,
        disable_spec_cache=False,
        spec_cache_dir=Path("/mnt/ava_cache/spec_cache"),
    )

    assert cmd[-2:] == ["--spec-cache-dir", "/mnt/ava_cache/spec_cache"]


def test_env_flag_parses_truthy_and_falsey_values(monkeypatch):
    module = _load_module()

    monkeypatch.setenv("TEST_SPEC_CACHE_FLAG", "true")
    assert module._env_flag("TEST_SPEC_CACHE_FLAG", default=False) is True

    monkeypatch.setenv("TEST_SPEC_CACHE_FLAG", "0")
    assert module._env_flag("TEST_SPEC_CACHE_FLAG", default=True) is False

    monkeypatch.delenv("TEST_SPEC_CACHE_FLAG", raising=False)
    assert module._env_flag("UNSET_FLAG", default=True) is True
    assert module._env_flag("UNSET_FLAG", default=False) is False
