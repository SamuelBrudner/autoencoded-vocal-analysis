import argparse
import importlib.util
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "cloud"
        / "aws"
        / "submit_birdsong_training_job.py"
    )
    spec = importlib.util.spec_from_file_location(
        "submit_birdsong_training_job",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_payload_includes_dependency_timeout_and_env():
    module = _load_module()
    args = argparse.Namespace(
        job_name="pk249-train",
        job_queue="ava-gpu-queue-4x",
        job_definition="ava-train-gpu-4x",
        manifest_s3_uri="s3://bucket/pk249/manifest.json",
        config_s3_uri="s3://bucket/pk249/config.yaml",
        s3_audio_root="s3://bucket/pk249/audio",
        s3_roi_root="s3://bucket/pk249/roi",
        s3_run_root="s3://bucket/pk249/runs",
        run_name="run-001",
        roi_format="parquet",
        roi_parquet_name="roi.parquet",
        download_jobs=8,
        batch_size=32,
        num_workers=4,
        epochs=51,
        train_dataset_length=262144,
        test_dataset_length=16384,
        spec_cache_dir="/mnt/ava_cache/spec_cache",
        trainer_kwargs_json='{"accelerator":"gpu","devices":4,"strategy":"ddp"}',
        preflight_sample_dirs=25,
        preflight_sample_segments=5000,
        preflight_seed=0,
        max_empty_fraction=0.01,
        disk_telemetry_every_n_epochs=5,
        workdir="/tmp/ava_train_workdir",
        timeout_seconds=172800,
        depends_on_job_id=["roi-job-123"],
        override_command=False,
        emit_json=None,
        submit=False,
    )

    payload = module.build_payload(args)

    assert payload["jobName"] == "pk249-train"
    assert payload["jobQueue"] == "ava-gpu-queue-4x"
    assert payload["jobDefinition"] == "ava-train-gpu-4x"
    assert payload["dependsOn"] == [{"jobId": "roi-job-123"}]
    assert payload["timeout"] == {"attemptDurationSeconds": 172800}
    env = {
        item["name"]: item["value"]
        for item in payload["containerOverrides"]["environment"]
    }
    assert env["AVA_MANIFEST_S3_URI"] == "s3://bucket/pk249/manifest.json"
    assert env["AVA_CONFIG_S3_URI"] == "s3://bucket/pk249/config.yaml"
    assert env["AVA_S3_AUDIO_ROOT"] == "s3://bucket/pk249/audio"
    assert env["AVA_S3_ROI_ROOT"] == "s3://bucket/pk249/roi"
    assert env["AVA_S3_RUN_ROOT"] == "s3://bucket/pk249/runs"
    assert env["AVA_RUN_NAME"] == "run-001"
    assert env["AVA_BATCH_SIZE"] == "32"
    assert env["AVA_NUM_WORKERS"] == "4"
    assert env["AVA_EPOCHS"] == "51"
    assert env["AVA_TRAIN_DATASET_LENGTH"] == "262144"
    assert env["AVA_TEST_DATASET_LENGTH"] == "16384"
    assert env["AVA_DISK_TELEMETRY_EVERY_N_EPOCHS"] == "5"
