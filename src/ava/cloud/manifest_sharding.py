"""Deterministic sharding for birdsong manifest entries.

This mirrors the ordering used by scripts/run_birdsong_roi.py:
- split=train: iterate manifest["train"] in order
- split=test: iterate manifest["test"] in order
- split=all: iterate train then test, preserving order within each list

Sharding is by *entry index* with modulo arithmetic, so reruns remain stable as
long as the manifest ordering is stable.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


ManifestEntry = Mapping[str, Any]
EntryPair = tuple[str, ManifestEntry]  # (split_name, entry)


def iter_manifest_entry_pairs(manifest: Mapping[str, Any], split: str) -> list[EntryPair]:
    """Return ordered (split_name, entry) pairs for `split`."""
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all")

    pairs: list[EntryPair] = []
    if split in {"train", "all"}:
        for entry in manifest.get("train", []) or []:
            pairs.append(("train", entry))
    if split in {"test", "all"}:
        for entry in manifest.get("test", []) or []:
            pairs.append(("test", entry))
    return pairs


def apply_max_dirs(pairs: list[EntryPair], max_dirs: Optional[int]) -> list[EntryPair]:
    if max_dirs is None:
        return list(pairs)
    if int(max_dirs) < 0:
        raise ValueError("max_dirs must be >= 0")
    return list(pairs[: int(max_dirs)])


def select_shard(pairs: list[EntryPair], num_shards: int, shard_index: int) -> list[EntryPair]:
    """Select the modulo shard of `pairs`."""
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    if int(num_shards) <= 1:
        return list(pairs)
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")
    return [pair for idx, pair in enumerate(pairs) if (idx % int(num_shards)) == int(shard_index)]


def pairs_to_manifest(pairs: list[EntryPair]) -> dict:
    """Build a manifest-like dict with `train` and `test` keys from pairs."""
    out = {"train": [], "test": []}
    for split_name, entry in pairs:
        if split_name not in {"train", "test"}:
            raise ValueError(f"Unexpected split name: {split_name!r}")
        out[split_name].append(dict(entry))
    return out


def count_manifest_entries(manifest: Mapping[str, Any], split: str) -> int:
    return len(iter_manifest_entry_pairs(manifest, split=split))

