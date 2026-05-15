"""Manifest-aware S3 audio coverage checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

from ava.cloud.manifest_sharding import apply_max_dirs, iter_manifest_entry_pairs


_S3_LS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(\d+)\s+(.+)$")


@dataclass(frozen=True)
class S3Uri:
	bucket: str
	key: str


@dataclass(frozen=True)
class ManifestAudioPrefix:
	split: str
	manifest_index: int
	audio_dir_rel: str
	bird_id_norm: str | None
	regime: str | None
	dph: float | None
	expected_files: int
	s3_uri: str
	s3_key_prefix: str


def parse_s3_uri(uri: str) -> S3Uri:
	"""Parse an s3:// URI into bucket and key components."""
	value = str(uri).strip()
	if not value.startswith("s3://"):
		raise ValueError(f"Expected s3:// URI, got: {uri!r}")
	remainder = value[len("s3://") :]
	bucket, sep, key = remainder.partition("/")
	if not bucket or not sep:
		raise ValueError(f"Expected s3://bucket/key URI, got: {uri!r}")
	return S3Uri(bucket=bucket, key=key.strip("/"))


def join_s3_uri(root_uri: str, rel: str) -> str:
	root = str(root_uri).rstrip("/")
	rel = str(rel or "").strip("/")
	return root if not rel else f"{root}/{rel}"


def join_s3_key(*parts: str) -> str:
	return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def build_manifest_audio_prefixes(
	manifest: Mapping[str, Any],
	*,
	split: str,
	s3_audio_root: str,
	max_dirs: Optional[int] = None,
) -> list[ManifestAudioPrefix]:
	"""Return manifest audio prefixes expected under an S3 audio root."""
	root = parse_s3_uri(s3_audio_root)
	pairs = iter_manifest_entry_pairs(manifest, split=split)
	pairs = apply_max_dirs(pairs, max_dirs=max_dirs)
	prefixes: list[ManifestAudioPrefix] = []
	for index, (split_name, entry) in enumerate(pairs):
		rel = str(entry.get("audio_dir_rel") or "").strip("/")
		if not rel:
			raise ValueError("Manifest entry missing audio_dir_rel.")
		try:
			dph = float(entry["dph"]) if entry.get("dph") is not None else None
		except (TypeError, ValueError):
			dph = None
		prefixes.append(
			ManifestAudioPrefix(
				split=split_name,
				manifest_index=index,
				audio_dir_rel=rel,
				bird_id_norm=(
					None if entry.get("bird_id_norm") is None else str(entry.get("bird_id_norm"))
				),
				regime=None if entry.get("regime") is None else str(entry.get("regime")),
				dph=dph,
				expected_files=int(entry.get("num_files") or 0),
				s3_uri=join_s3_uri(s3_audio_root, rel),
				s3_key_prefix=join_s3_key(root.key, rel),
			)
		)
	return prefixes


def parse_aws_s3_ls_key(line: str) -> str | None:
	"""Extract the object key from one `aws s3 ls --recursive` output line."""
	match = _S3_LS_RE.match(str(line).rstrip("\n"))
	if not match:
		return None
	return match.group(2)


def is_audio_wav_key(key: str) -> bool:
	return str(key).lower().endswith(".wav")


def count_wav_keys_by_manifest_prefix(
	prefixes: Iterable[ManifestAudioPrefix],
	keys: Iterable[str],
) -> tuple[dict[str, int], int]:
	"""Count WAV keys assigned to manifest prefixes.

	The lookup walks object-key parent directories until it finds an exact
	manifest `s3_key_prefix`, which keeps the check robust to spaces in path
	components and to the few cohort directories with four path components.
	"""
	prefix_list = list(prefixes)
	by_key_prefix = {item.s3_key_prefix: item.audio_dir_rel for item in prefix_list}
	counts = {item.audio_dir_rel: 0 for item in prefix_list}
	unmatched = 0
	for key in keys:
		if not is_audio_wav_key(key):
			continue
		parts = str(key).strip("/").split("/")
		matched_rel = None
		for end in range(len(parts) - 1, 0, -1):
			candidate = "/".join(parts[:end])
			matched_rel = by_key_prefix.get(candidate)
			if matched_rel is not None:
				break
		if matched_rel is None:
			unmatched += 1
		else:
			counts[matched_rel] += 1
	return counts, unmatched


def summarize_manifest_audio_coverage(
	prefixes: Iterable[ManifestAudioPrefix],
	keys: Iterable[str],
) -> dict[str, Any]:
	"""Build a JSON-serializable coverage summary for manifest audio in S3."""
	prefix_list = list(prefixes)
	counts, unmatched = count_wav_keys_by_manifest_prefix(prefix_list, keys)
	rows = []
	for item in prefix_list:
		observed = int(counts.get(item.audio_dir_rel, 0))
		expected = int(item.expected_files)
		rows.append(
			{
				**asdict(item),
				"observed_wav_files": observed,
				"has_wavs": observed > 0,
				"count_match": expected == observed,
				"count_delta": int(observed - expected),
			}
		)
	missing = [row for row in rows if not row["has_wavs"]]
	count_mismatches = [row for row in rows if not row["count_match"]]
	return {
		"status": "ok" if not missing and not count_mismatches else "incomplete",
		"manifest_entries": int(len(rows)),
		"observed_dirs": int(sum(1 for row in rows if row["has_wavs"])),
		"missing_dirs": int(len(missing)),
		"count_mismatch_dirs": int(len(count_mismatches)),
		"expected_wav_files": int(sum(row["expected_files"] for row in rows)),
		"observed_wav_files": int(sum(row["observed_wav_files"] for row in rows)),
		"unmatched_wav_keys": int(unmatched),
		"missing": missing,
		"count_mismatches": count_mismatches,
		"rows": rows,
	}


__all__ = [
	"ManifestAudioPrefix",
	"S3Uri",
	"build_manifest_audio_prefixes",
	"count_wav_keys_by_manifest_prefix",
	"is_audio_wav_key",
	"join_s3_key",
	"join_s3_uri",
	"parse_aws_s3_ls_key",
	"parse_s3_uri",
	"summarize_manifest_audio_coverage",
]
