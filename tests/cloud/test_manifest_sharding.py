from __future__ import annotations

import pytest

from ava.cloud.manifest_sharding import (
    apply_max_dirs,
    count_manifest_entries,
    iter_manifest_entry_pairs,
    pairs_to_manifest,
    select_shard,
)


def _manifest() -> dict:
    return {
        "train": [{"audio_dir_rel": "t0"}, {"audio_dir_rel": "t1"}, {"audio_dir_rel": "t2"}],
        "test": [{"audio_dir_rel": "s0"}, {"audio_dir_rel": "s1"}],
    }


def test_iter_order_all_is_train_then_test() -> None:
    pairs = iter_manifest_entry_pairs(_manifest(), split="all")
    rels = [p[1]["audio_dir_rel"] for p in pairs]
    assert rels == ["t0", "t1", "t2", "s0", "s1"]


def test_count_manifest_entries() -> None:
    m = _manifest()
    assert count_manifest_entries(m, split="train") == 3
    assert count_manifest_entries(m, split="test") == 2
    assert count_manifest_entries(m, split="all") == 5


def test_apply_max_dirs() -> None:
    pairs = iter_manifest_entry_pairs(_manifest(), split="all")
    out = apply_max_dirs(pairs, max_dirs=2)
    assert [p[1]["audio_dir_rel"] for p in out] == ["t0", "t1"]


def test_select_shard_modulo() -> None:
    pairs = iter_manifest_entry_pairs(_manifest(), split="all")
    # idx: 0=t0, 1=t1, 2=t2, 3=s0, 4=s1
    shard0 = select_shard(pairs, num_shards=2, shard_index=0)
    shard1 = select_shard(pairs, num_shards=2, shard_index=1)
    assert [p[1]["audio_dir_rel"] for p in shard0] == ["t0", "t2", "s1"]
    assert [p[1]["audio_dir_rel"] for p in shard1] == ["t1", "s0"]


def test_pairs_to_manifest_preserves_splits() -> None:
    pairs = [
        ("train", {"audio_dir_rel": "t0"}),
        ("test", {"audio_dir_rel": "s0"}),
        ("train", {"audio_dir_rel": "t1"}),
    ]
    out = pairs_to_manifest(pairs)
    assert [e["audio_dir_rel"] for e in out["train"]] == ["t0", "t1"]
    assert [e["audio_dir_rel"] for e in out["test"]] == ["s0"]


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        iter_manifest_entry_pairs(_manifest(), split="nope")
    with pytest.raises(ValueError):
        apply_max_dirs(iter_manifest_entry_pairs(_manifest(), split="all"), max_dirs=-1)
    with pytest.raises(ValueError):
        select_shard(iter_manifest_entry_pairs(_manifest(), split="all"), num_shards=0, shard_index=0)
    with pytest.raises(ValueError):
        select_shard(iter_manifest_entry_pairs(_manifest(), split="all"), num_shards=2, shard_index=2)

