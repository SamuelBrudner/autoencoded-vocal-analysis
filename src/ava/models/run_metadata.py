"""Utilities for capturing run metadata for training runs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional


def _file_sha256(path: Optional[str]) -> Optional[str]:
	if not path:
		return None
	try:
		digest = hashlib.sha256()
		with open(path, "rb") as handle:
			for chunk in iter(lambda: handle.read(1024 * 1024), b""):
				digest.update(chunk)
		return digest.hexdigest()
	except OSError:
		return None


def _find_git_root(start: Optional[Path] = None) -> Optional[Path]:
	current = (start or Path(__file__).resolve()).parent
	for candidate in (current, *current.parents):
		if (candidate / ".git").exists():
			return candidate
	return None


def _git_commit() -> Optional[str]:
	git_root = _find_git_root()
	if git_root is None:
		return None
	try:
		result = subprocess.run(
			["git", "-C", git_root.as_posix(), "rev-parse", "HEAD"],
			check=True,
			capture_output=True,
			text=True,
		)
	except (OSError, subprocess.CalledProcessError):
		return None
	commit = result.stdout.strip()
	return commit or None


def _manifest_root(path: Optional[str]) -> Optional[str]:
	if not path:
		return None
	try:
		with open(path, "r", encoding="utf-8") as handle:
			manifest = json.load(handle)
	except (OSError, json.JSONDecodeError):
		return None
	return manifest.get("root") or manifest.get("dataset_root")


def build_run_metadata(
	config_path: Optional[str] = None,
	manifest_path: Optional[str] = None,
	dataset_root: Optional[str] = None,
) -> dict:
	resolved_dataset_root = dataset_root or _manifest_root(manifest_path)
	return {
		"config_path": config_path,
		"config_sha256": _file_sha256(config_path),
		"manifest_path": manifest_path,
		"git_commit": _git_commit(),
		"dataset_root": resolved_dataset_root,
	}


def write_run_metadata(
	save_dir: str,
	config_path: Optional[str] = None,
	manifest_path: Optional[str] = None,
	dataset_root: Optional[str] = None,
	output_name: str = "run_metadata.json",
) -> Optional[str]:
	if not save_dir:
		return None
	try:
		Path(save_dir).mkdir(parents=True, exist_ok=True)
		metadata = build_run_metadata(
			config_path=config_path,
			manifest_path=manifest_path,
			dataset_root=dataset_root,
		)
		out_path = Path(save_dir) / output_name
		with open(out_path, "w", encoding="utf-8") as handle:
			json.dump(metadata, handle, indent=2, sort_keys=False)
		return out_path.as_posix()
	except OSError:
		return None
