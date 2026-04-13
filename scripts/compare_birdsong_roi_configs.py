#!/usr/bin/env python3
"""Compare ROI segmentation configs on a directory of birdsong wavs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.segmenting.utils import get_spec


ALGORITHMS = {
    "amplitude": get_onsets_offsets,
    "amplitude_segmentation": get_onsets_offsets,
}


@dataclass(frozen=True)
class ConfigSpec:
    label: str
    path: Path
    params: dict


@dataclass(frozen=True)
class ClipMetrics:
    clip_stem: str
    config_label: str
    n_segments: int
    total_roi_sec: float
    max_segment_sec: float
    median_segment_sec: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio_dir",
        type=Path,
        help="Directory containing wav files to compare.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Repeatable ROI config yaml path.",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=[],
        help="Optional label for each --config (defaults to stem).",
    )
    parser.add_argument(
        "--reference-roi-parquet",
        type=Path,
        default=None,
        help="Optional parquet bundle to compare against the deployed ROI output.",
    )
    parser.add_argument(
        "--clip-stem",
        type=str,
        action="append",
        default=[],
        help="Repeatable clip stem to analyze. Defaults to a random sample from audio_dir.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=12,
        help="When --clip-stem is omitted, analyze at most this many randomly sampled clips.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for clip sampling.",
    )
    parser.add_argument(
        "--plot-clips",
        type=int,
        default=6,
        help="How many analyzed clips to render as comparison figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSON/CSV summaries and plots.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = payload.get("segment", payload)
    if not isinstance(params, dict):
        raise ValueError(f"segment config must be a mapping: {path}")
    params = dict(params)
    algo_key = str(params.get("algorithm", "amplitude")).lower()
    if algo_key not in ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algo_key}' in {path}")
    params["algorithm"] = ALGORITHMS[algo_key]
    return params


def _load_reference_index(path: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyarrow is required for --reference-roi-parquet") from exc

    table = pq.read_table(path, columns=["clip_stem", "onsets_sec", "offsets_sec"])
    payload = table.to_pydict()
    stems = payload.get("clip_stem") or []
    onsets = payload.get("onsets_sec") or []
    offsets = payload.get("offsets_sec") or []
    if not (len(stems) == len(onsets) == len(offsets)):
        raise ValueError(f"malformed ROI parquet bundle: {path}")

    index: dict[str, np.ndarray] = {}
    for stem, onset_list, offset_list in zip(stems, onsets, offsets):
        onset_arr = np.asarray(onset_list or [], dtype=np.float64).reshape(-1)
        offset_arr = np.asarray(offset_list or [], dtype=np.float64).reshape(-1)
        if onset_arr.shape != offset_arr.shape:
            raise ValueError(f"mismatched onset/offset lengths for stem '{stem}' in {path}")
        index[str(stem)] = np.column_stack([onset_arr, offset_arr]) if onset_arr.size else np.zeros((0, 2), dtype=np.float64)
    return index


def _select_clips(audio_dir: Path, stems: list[str], *, max_clips: int, seed: int) -> list[Path]:
    if stems:
        paths = []
        for stem in stems:
            stem_path = Path(stem)
            if stem_path.suffix.lower() == ".wav":
                candidate = audio_dir / stem_path.name
            else:
                candidate = audio_dir / f"{stem}.wav"
            if not candidate.exists():
                raise FileNotFoundError(candidate)
            paths.append(candidate)
        return paths

    wavs = sorted(audio_dir.glob("*.wav"))
    if max_clips <= 0 or len(wavs) <= max_clips:
        return wavs
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(wavs), size=max_clips, replace=False))
    return [wavs[int(i)] for i in idx]


def _segments_to_metrics(clip_stem: str, config_label: str, segments: np.ndarray) -> ClipMetrics:
    durations = segments[:, 1] - segments[:, 0] if segments.size else np.zeros((0,), dtype=np.float64)
    return ClipMetrics(
        clip_stem=clip_stem,
        config_label=config_label,
        n_segments=int(durations.size),
        total_roi_sec=float(durations.sum()) if durations.size else 0.0,
        max_segment_sec=float(durations.max()) if durations.size else 0.0,
        median_segment_sec=float(np.median(durations)) if durations.size else 0.0,
    )


def _compute_segments(audio: np.ndarray, params: dict) -> np.ndarray:
    local_params = copy.deepcopy(params)
    onsets, offsets = local_params["algorithm"](audio, local_params)
    onset_arr = np.asarray(onsets, dtype=np.float64).reshape(-1)
    offset_arr = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if onset_arr.shape != offset_arr.shape:
        raise ValueError("segmenter returned mismatched onsets/offsets")
    if onset_arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack([onset_arr, offset_arr])


def _write_metrics_csv(path: Path, rows: list[ClipMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_stem",
                "config_label",
                "n_segments",
                "total_roi_sec",
                "max_segment_sec",
                "median_segment_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_stem": row.clip_stem,
                    "config_label": row.config_label,
                    "n_segments": row.n_segments,
                    "total_roi_sec": f"{row.total_roi_sec:.9f}",
                    "max_segment_sec": f"{row.max_segment_sec:.9f}",
                    "median_segment_sec": f"{row.median_segment_sec:.9f}",
                }
            )


def _summarize_metrics(rows: list[ClipMetrics]) -> dict[str, dict[str, float | int]]:
    by_label: dict[str, list[ClipMetrics]] = {}
    for row in rows:
        by_label.setdefault(row.config_label, []).append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for label, entries in sorted(by_label.items()):
        n_segments = np.asarray([row.n_segments for row in entries], dtype=np.int64)
        total_roi = np.asarray([row.total_roi_sec for row in entries], dtype=np.float64)
        max_seg = np.asarray([row.max_segment_sec for row in entries], dtype=np.float64)
        summary[label] = {
            "n_clips": int(len(entries)),
            "empty_clips": int(np.sum(n_segments == 0)),
            "segments_per_clip_median": float(np.median(n_segments)),
            "segments_per_clip_mean": float(np.mean(n_segments)),
            "total_roi_sec_median": float(np.median(total_roi)),
            "total_roi_sec_mean": float(np.mean(total_roi)),
            "max_segment_sec_median": float(np.median(max_seg)),
            "max_segment_sec_p90": float(np.percentile(max_seg, 90)),
            "max_segment_sec_p99": float(np.percentile(max_seg, 99)),
            "clips_with_max_seg_ge_0_5": int(np.sum(max_seg >= 0.5)),
            "clips_with_max_seg_ge_1_0": int(np.sum(max_seg >= 1.0)),
            "clips_with_max_seg_ge_2_0": int(np.sum(max_seg >= 2.0)),
        }
    return summary


def _plot_clip(
    *,
    wav_path: Path,
    spec_params: dict,
    segments_by_label: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fs, audio = wavfile.read(wav_path)
    if fs != int(spec_params["fs"]):
        local_spec_params = dict(spec_params)
        local_spec_params["fs"] = fs
    else:
        local_spec_params = spec_params
    spec, dt, freqs = get_spec(audio, local_spec_params)
    duration_sec = float(len(audio) / fs)

    labels = list(segments_by_label.keys())
    fig, axes = plt.subplots(
        1 + len(labels),
        1,
        figsize=(12, 2.4 + 1.2 * len(labels)),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * len(labels)},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    ax0 = axes[0]
    ax0.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration_sec, float(freqs[0]) / 1e3, float(freqs[-1]) / 1e3],
    )
    ax0.set_ylabel("kHz")
    ax0.set_title(wav_path.name)

    for ax, label in zip(axes[1:], labels):
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        segments = segments_by_label[label]
        for onset, offset in segments:
            ax.plot([onset, offset], [0.5, 0.5], linewidth=6, solid_capstyle="butt")
        ax.set_xlim(0.0, duration_sec)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.label and len(args.label) != len(args.config):
        raise ValueError("--label must be provided zero times or once per --config")

    labels = list(args.label) if args.label else [path.stem for path in args.config]
    configs = [
        ConfigSpec(label=label, path=path.resolve(), params=_load_config(path.resolve()))
        for label, path in zip(labels, args.config)
    ]

    reference_index = (
        _load_reference_index(args.reference_roi_parquet.resolve())
        if args.reference_roi_parquet is not None
        else None
    )

    wav_paths = _select_clips(
        args.audio_dir.resolve(),
        args.clip_stem,
        max_clips=int(args.max_clips),
        seed=int(args.seed),
    )
    if not wav_paths:
        raise ValueError("no wav files selected")

    metrics: list[ClipMetrics] = []
    plot_payloads: list[tuple[Path, dict[str, np.ndarray], dict]] = []

    for wav_path in wav_paths:
        fs, audio = wavfile.read(wav_path)
        clip_stem = wav_path.stem
        segments_by_label: dict[str, np.ndarray] = {}

        if reference_index is not None:
            reference_segments = reference_index.get(clip_stem, np.zeros((0, 2), dtype=np.float64))
            segments_by_label["reference"] = reference_segments
            metrics.append(_segments_to_metrics(clip_stem, "reference", reference_segments))

        spec_params = None
        for config in configs:
            params = copy.deepcopy(config.params)
            if fs != int(params["fs"]):
                params["fs"] = fs
            segments = _compute_segments(audio, params)
            segments_by_label[config.label] = segments
            metrics.append(_segments_to_metrics(clip_stem, config.label, segments))
            if spec_params is None:
                spec_params = params

        assert spec_params is not None
        plot_payloads.append((wav_path, segments_by_label, spec_params))

    metrics_path = args.output_dir / "per_clip_metrics.csv"
    _write_metrics_csv(metrics_path, metrics)

    summary = {
        "audio_dir": args.audio_dir.resolve().as_posix(),
        "reference_roi_parquet": (
            None if args.reference_roi_parquet is None else args.reference_roi_parquet.resolve().as_posix()
        ),
        "configs": [
            {
                "label": config.label,
                "path": config.path.as_posix(),
            }
            for config in configs
        ],
        "n_clips": int(len(wav_paths)),
        "clip_stems": [path.stem for path in wav_paths],
        "summary_by_label": _summarize_metrics(metrics),
        "per_clip_metrics_csv": metrics_path.as_posix(),
    }

    plots_dir = args.output_dir / "plots"
    for wav_path, segments_by_label, spec_params in plot_payloads[: max(0, int(args.plot_clips))]:
        _plot_clip(
            wav_path=wav_path,
            spec_params=spec_params,
            segments_by_label=segments_by_label,
            output_path=plots_dir / f"{wav_path.stem}.png",
        )
    summary["plots_dir"] = plots_dir.as_posix()

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
