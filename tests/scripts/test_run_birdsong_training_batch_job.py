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


def test_sync_stable_checkpoints_requires_two_observations(monkeypatch, tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "checkpoint_005.tar"
    checkpoint.write_bytes(b"checkpoint")
    sync_calls = []

    monkeypatch.setattr(
        module,
        "_sync_path_to_s3",
        lambda aws, local_path, s3_uri: sync_calls.append((local_path, s3_uri)),
    )

    class TwoPollStop:
        def __init__(self):
            self.polls = 0

        def wait(self, _interval):
            self.polls += 1
            return self.polls > 2

    module._sync_stable_checkpoints(
        "aws",
        tmp_path,
        "s3://bucket/run/training_run",
        TwoPollStop(),
        1.0,
    )

    assert sync_calls == [
        (checkpoint, "s3://bucket/run/training_run/checkpoint_005.tar")
    ]
