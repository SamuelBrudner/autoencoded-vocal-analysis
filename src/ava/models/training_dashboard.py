"""Utilities for writing lightweight HTML dashboards for training runs."""
from __future__ import annotations

import html
import json
import math
import os
import socket
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - optional for backfill/render utilities
    pl = None

if pl is not None:
    _LightningCallbackBase = pl.Callback
else:  # pragma: no cover - only used when Lightning is unavailable
    class _LightningCallbackBase:
        pass


DASHBOARD_JSON_NAME = "training_dashboard.json"
DASHBOARD_HTML_NAME = "training_dashboard.html"
MAX_VISIBLE_POINTS = 200
REFRESH_SECONDS = 30

SERIES_COLORS = {
    "train": "#0f766e",
    "val": "#b45309",
    "default": "#1d4ed8",
}

PREFERRED_GROUP_ORDER = [
    "loss",
    "recon_mse",
    "recon_nll",
    "kl",
    "kl_weight",
    "latent_mean_abs",
    "latent_var_mean",
    "model_precision",
    "log_precision",
    "invariance_loss",
    "invariance_weight",
    "invariance_mu_l2",
    "invariance_mu_cosine",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numel") and callable(value.numel):
        if value.numel() != 1:
            return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _dashboard_dir(save_dir: str) -> Path:
    if save_dir:
        return Path(save_dir)
    return Path.cwd()


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _relative_path(base_dir: Path, target: Path) -> Optional[str]:
    if not target.exists():
        return None
    try:
        return target.relative_to(base_dir).as_posix()
    except ValueError:
        return target.as_posix()


def _checkpoint_entries(save_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for checkpoint in sorted(save_dir.glob("checkpoint_*.tar")):
        try:
            epoch = int(checkpoint.stem.split("_")[-1])
        except ValueError:
            epoch = None
        entries.append(
            {
                "name": checkpoint.name,
                "path": _relative_path(save_dir, checkpoint),
                "epoch": epoch,
                "size_bytes": checkpoint.stat().st_size,
            }
        )
    return entries


def _lightning_log_dir(save_dir: Path) -> Optional[str]:
    for candidate in sorted(save_dir.glob("lightning_logs/version_*")):
        if candidate.is_dir():
            return _relative_path(save_dir, candidate)
    return None


def _artifact_paths(save_dir: Path) -> dict[str, Any]:
    artifacts = {
        "dashboard_json": DASHBOARD_JSON_NAME,
        "dashboard_html": DASHBOARD_HTML_NAME,
        "run_metadata": None,
        "reconstruction_pdf": None,
        "lightning_log_dir": _lightning_log_dir(save_dir),
        "checkpoints": _checkpoint_entries(save_dir),
    }
    run_metadata = save_dir / "run_metadata.json"
    if run_metadata.exists():
        artifacts["run_metadata"] = _relative_path(save_dir, run_metadata)
    reconstruction = save_dir / "reconstruction.pdf"
    if reconstruction.exists():
        artifacts["reconstruction_pdf"] = _relative_path(save_dir, reconstruction)
    return artifacts


def _prettify_metric_name(name: str) -> str:
    words = str(name).replace("_", " ").split()
    if not words:
        return str(name)
    replacements = {
        "kl": "KL",
        "mse": "MSE",
        "nll": "NLL",
        "mu": "mu",
        "abs": "|.|",
    }
    pretty = []
    for word in words:
        lower = word.lower()
        pretty.append(replacements.get(lower, lower.capitalize()))
    return " ".join(pretty)


def _series_color(metric_name: str) -> str:
    if str(metric_name).startswith("train_"):
        return SERIES_COLORS["train"]
    if str(metric_name).startswith("val_"):
        return SERIES_COLORS["val"]
    return SERIES_COLORS["default"]


def _metric_groups(history: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    standalone: dict[str, list[dict[str, Any]]] = {}
    for metric_name, points in history.items():
        if not points:
            continue
        if metric_name.startswith("train_"):
            grouped.setdefault(metric_name[len("train_") :], {})["train"] = points
            continue
        if metric_name.startswith("val_"):
            grouped.setdefault(metric_name[len("val_") :], {})["val"] = points
            continue
        standalone[metric_name] = points

    order_index = {name: idx for idx, name in enumerate(PREFERRED_GROUP_ORDER)}
    groups: list[dict[str, Any]] = []
    for suffix in sorted(grouped, key=lambda item: (order_index.get(item, 999), item)):
        groups.append(
            {
                "title": _prettify_metric_name(suffix),
                "series": [
                    {
                        "name": f"train_{suffix}",
                        "label": f"Train {_prettify_metric_name(suffix)}",
                        "points": grouped[suffix]["train"],
                    }
                    for key in ("train",)
                    if key in grouped[suffix]
                ]
                + [
                    {
                        "name": f"val_{suffix}",
                        "label": f"Val {_prettify_metric_name(suffix)}",
                        "points": grouped[suffix]["val"],
                    }
                    for key in ("val",)
                    if key in grouped[suffix]
                ],
            }
        )
    for metric_name in sorted(standalone):
        groups.append(
            {
                "title": _prettify_metric_name(metric_name),
                "series": [
                    {
                        "name": metric_name,
                        "label": _prettify_metric_name(metric_name),
                        "points": standalone[metric_name],
                    }
                ],
            }
        )
    return groups


def _trim_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= MAX_VISIBLE_POINTS:
        return list(points)
    step = max(1, math.ceil(len(points) / MAX_VISIBLE_POINTS))
    trimmed = list(points[::step])
    if trimmed[-1] != points[-1]:
        trimmed.append(points[-1])
    return trimmed


def _render_svg_chart(series_list: list[dict[str, Any]]) -> str:
    width = 760
    height = 260
    padding_left = 56
    padding_right = 16
    padding_top = 20
    padding_bottom = 36
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    normalized = []
    all_x = []
    all_y = []
    x_key = "epoch"
    for series in series_list:
        trimmed = _trim_points(series["points"])
        if not all("epoch" in point for point in trimmed):
            x_key = "step"
        normalized.append((series, trimmed))

    for _, points in normalized:
        for point in points:
            x_value = point.get(x_key)
            if x_value is None:
                x_value = point.get("step")
            all_x.append(float(x_value))
            all_y.append(float(point["value"]))

    if not all_x or not all_y:
        return "<p class=\"chart-empty\">No metrics recorded yet.</p>"

    min_x = min(all_x)
    max_x = max(all_x)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    min_y = min(all_y)
    max_y = max(all_y)
    if math.isclose(min_y, max_y):
        padding = abs(min_y) * 0.05 if not math.isclose(min_y, 0.0) else 1.0
        min_y -= padding
        max_y += padding
    else:
        padding = (max_y - min_y) * 0.08
        min_y -= padding
        max_y += padding

    def x_to_px(value: float) -> float:
        return padding_left + ((value - min_x) / (max_x - min_x)) * plot_width

    def y_to_px(value: float) -> float:
        return padding_top + (1.0 - ((value - min_y) / (max_y - min_y))) * plot_height

    x_ticks = [min_x + (max_x - min_x) * idx / 4 for idx in range(5)]
    y_ticks = [min_y + (max_y - min_y) * idx / 4 for idx in range(5)]

    grid_lines = []
    for tick in x_ticks:
        x_px = x_to_px(tick)
        grid_lines.append(
            f"<line x1=\"{x_px:.2f}\" y1=\"{padding_top}\" "
            f"x2=\"{x_px:.2f}\" y2=\"{padding_top + plot_height:.2f}\" "
            "stroke=\"#d8e2dc\" stroke-width=\"1\" />"
        )
    for tick in y_ticks:
        y_px = y_to_px(tick)
        grid_lines.append(
            f"<line x1=\"{padding_left}\" y1=\"{y_px:.2f}\" "
            f"x2=\"{padding_left + plot_width:.2f}\" y2=\"{y_px:.2f}\" "
            "stroke=\"#d8e2dc\" stroke-width=\"1\" />"
        )

    label_lines = []
    for tick in x_ticks:
        x_px = x_to_px(tick)
        label = f"{tick:.0f}" if x_key == "epoch" else f"{tick:.0f}"
        label_lines.append(
            f"<text x=\"{x_px:.2f}\" y=\"{height - 10}\" text-anchor=\"middle\">"
            f"{html.escape(label)}</text>"
        )
    for tick in y_ticks:
        y_px = y_to_px(tick)
        label_lines.append(
            f"<text x=\"{padding_left - 8}\" y=\"{y_px + 4:.2f}\" text-anchor=\"end\">"
            f"{html.escape(f'{tick:.3g}')}</text>"
        )

    series_paths = []
    for series, points in normalized:
        path = " ".join(
            (
                f"{x_to_px(float(point.get(x_key, point.get('step')))):.2f},"
                f"{y_to_px(float(point['value'])):.2f}"
            )
            for point in points
        )
        color = _series_color(series["name"])
        series_paths.append(
            f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"3\" "
            f"stroke-linecap=\"round\" stroke-linejoin=\"round\" points=\"{path}\" />"
        )
        last_point = points[-1]
        series_paths.append(
            f"<circle cx=\"{x_to_px(float(last_point.get(x_key, last_point.get('step')))):.2f}\" "
            f"cy=\"{y_to_px(float(last_point['value'])):.2f}\" r=\"4.5\" fill=\"{color}\" />"
        )

    axis_label = "Epoch" if x_key == "epoch" else "Global Step"
    return (
        "<svg viewBox=\"0 0 760 260\" class=\"chart-svg\" role=\"img\">"
        + "".join(grid_lines)
        + (
            f"<rect x=\"{padding_left}\" y=\"{padding_top}\" width=\"{plot_width:.2f}\" "
            f"height=\"{plot_height:.2f}\" fill=\"transparent\" stroke=\"#7d8f87\" "
            "stroke-width=\"1.25\" />"
        )
        + "".join(series_paths)
        + "".join(label_lines)
        + (
            f"<text x=\"{padding_left + plot_width / 2:.2f}\" y=\"{height - 2}\" "
            "text-anchor=\"middle\">"
            f"{html.escape(axis_label)}</text>"
        )
        + "</svg>"
    )


def _render_artifact_links(payload: dict[str, Any]) -> str:
    artifacts = payload.get("artifacts", {})
    links = []
    if artifacts.get("run_metadata"):
        links.append(
            f"<a href=\"{html.escape(artifacts['run_metadata'])}\">Run metadata</a>"
        )
    if artifacts.get("reconstruction_pdf"):
        links.append(
            f"<a href=\"{html.escape(artifacts['reconstruction_pdf'])}\">Reconstruction PDF</a>"
        )
    if artifacts.get("lightning_log_dir"):
        links.append(
            f"<a href=\"{html.escape(artifacts['lightning_log_dir'])}\">Lightning logs</a>"
        )
    for checkpoint in artifacts.get("checkpoints", [])[-3:]:
        if checkpoint.get("path"):
            label = checkpoint["name"]
            if checkpoint.get("epoch") is not None:
                label = f"{label} (epoch {checkpoint['epoch']})"
            links.append(f"<a href=\"{html.escape(checkpoint['path'])}\">{html.escape(label)}</a>")
    if not links:
        return "<p class=\"artifact-note\">No artifact links available yet.</p>"
    return "<div class=\"artifact-links\">" + "".join(links) + "</div>"


def _render_summary_cards(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    cards = [
        ("Status", payload.get("status", "unknown")),
        (
            "Epoch",
            (
                f"{summary['current_epoch']} / {summary['max_epochs']}"
                if summary.get("current_epoch") is not None and summary.get("max_epochs")
                else "n/a"
            ),
        ),
        (
            "Latest Train Loss",
            _format_metric(summary.get("latest_metrics", {}).get("train_loss")),
        ),
        (
            "Latest Val Loss",
            _format_metric(summary.get("latest_metrics", {}).get("val_loss")),
        ),
        (
            "Best Val Loss",
            _format_metric(summary.get("best_metrics", {}).get("val_loss", {}).get("value")),
        ),
        (
            "Checkpoints",
            str(summary.get("checkpoint_count", 0)),
        ),
        ("Host", summary.get("host") or "n/a"),
        ("Device", summary.get("device") or "n/a"),
    ]
    html_cards = []
    for title, value in cards:
        html_cards.append(
            "<article class=\"summary-card\">"
            f"<h3>{html.escape(str(title))}</h3>"
            f"<p>{html.escape(str(value))}</p>"
            "</article>"
        )
    return "<section class=\"summary-grid\">" + "".join(html_cards) + "</section>"


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def render_training_dashboard_html(
    payload: dict[str, Any],
    refresh_seconds: int = REFRESH_SECONDS,
) -> str:
    history = payload.get("history", {})
    groups = _metric_groups(history)
    summary = payload.get("summary", {})
    metadata = payload.get("run_metadata") or {}
    latest_update = payload.get("updated_at") or "unknown"
    refresh_tag = ""
    if payload.get("status") == "running" and refresh_seconds > 0:
        refresh_tag = (
            f"<meta http-equiv=\"refresh\" content=\"{int(refresh_seconds)}\" />"
        )

    chart_sections = []
    for group in groups:
        legend = "".join(
            (
                "<li>"
                f"<span class=\"legend-swatch\" style=\"background:{_series_color(series['name'])}\"></span>"
                f"{html.escape(series['label'])}"
                "</li>"
            )
            for series in group["series"]
        )
        chart_sections.append(
            "<section class=\"chart-card\">"
            f"<div class=\"chart-header\"><h2>{html.escape(group['title'])}</h2>"
            f"<ul class=\"legend\">{legend}</ul></div>"
            f"{_render_svg_chart(group['series'])}"
            "</section>"
        )
    if not chart_sections:
        chart_sections.append(
            "<section class=\"chart-card\"><h2>Metrics</h2>"
            "<p class=\"chart-empty\">Metrics will appear after the first logged epoch.</p>"
            "</section>"
        )

    details = [
        ("Updated", latest_update),
        ("Started", payload.get("started_at") or "unknown"),
        ("Latest Step", summary.get("latest_step") if summary.get("latest_step") is not None else "n/a"),
        ("AWS Batch Job", summary.get("aws_batch_job_id") or "n/a"),
        ("Git Commit", metadata.get("git_commit") or "n/a"),
        ("Dataset Root", metadata.get("dataset_root") or "n/a"),
        ("Config", metadata.get("config_path") or "n/a"),
        ("Manifest", metadata.get("manifest_path") or "n/a"),
    ]
    detail_rows = "".join(
        (
            "<div class=\"detail-row\">"
            f"<dt>{html.escape(str(label))}</dt>"
            f"<dd>{html.escape(str(value))}</dd>"
            "</div>"
        )
        for label, value in details
    )

    failure_note = ""
    if payload.get("status") == "failed" and payload.get("error"):
        failure_note = (
            "<section class=\"failure-card\">"
            "<h2>Failure</h2>"
            f"<pre>{html.escape(str(payload['error']))}</pre>"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {refresh_tag}
  <title>AVA Training Dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5b6b66;
      --line: #d8e2dc;
      --accent: #0f766e;
      --accent-2: #b45309;
      --shadow: 0 20px 60px rgba(31, 41, 51, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.15), transparent 30rem),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 28rem),
        var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 3vw, 3.2rem);
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      max-width: 54rem;
      color: var(--muted);
      font-size: 1.02rem;
    }}
    .status-pill {{
      display: inline-flex;
      width: fit-content;
      align-items: center;
      gap: 0.5rem;
      padding: 0.45rem 0.85rem;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.01em;
      text-transform: uppercase;
      font-size: 0.78rem;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 14px;
      margin: 24px 0 30px;
    }}
    .summary-card,
    .chart-card,
    .detail-card,
    .failure-card {{
      background: var(--panel);
      border: 1px solid rgba(91, 107, 102, 0.14);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .summary-card {{
      padding: 16px 18px;
      min-height: 112px;
    }}
    .summary-card h3 {{
      margin: 0 0 0.6rem;
      color: var(--muted);
      font-size: 0.84rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .summary-card p {{
      margin: 0;
      font-size: 1.35rem;
      line-height: 1.1;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .dashboard-grid {{
      display: grid;
      gap: 16px;
    }}
    .chart-card {{
      padding: 18px 18px 12px;
    }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .chart-card h2,
    .detail-card h2,
    .failure-card h2 {{
      margin: 0;
      font-size: 1.18rem;
    }}
    .legend {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .legend li {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }}
    .legend-swatch {{
      width: 0.9rem;
      height: 0.9rem;
      border-radius: 999px;
      display: inline-block;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .chart-empty {{
      margin: 0;
      color: var(--muted);
    }}
    .detail-card {{
      padding: 18px;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px 24px;
      margin-top: 14px;
    }}
    .detail-row {{
      display: grid;
      gap: 0.25rem;
    }}
    .detail-row dt {{
      color: var(--muted);
      font-size: 0.86rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .detail-row dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .artifact-links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .artifact-links a {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid rgba(15, 118, 110, 0.25);
      padding-bottom: 1px;
    }}
    .artifact-note {{
      color: var(--muted);
    }}
    .failure-card {{
      padding: 18px;
    }}
    .failure-card pre {{
      white-space: pre-wrap;
      font-family: Menlo, Consolas, monospace;
      margin: 12px 0 0;
      color: #7f1d1d;
    }}
    @media (max-width: 700px) {{
      main {{
        padding: 24px 14px 40px;
      }}
      .summary-card p {{
        font-size: 1.15rem;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <span class="status-pill">{html.escape(str(payload.get("status", "unknown")))}</span>
      <h1>AVA Training Dashboard</h1>
      <p>
        Lightweight progress view for local or remote runs. This page is self-contained,
        refreshes automatically while the job is running, and can be served from any
        static host or copied out of an AWS run directory.
      </p>
    </section>
    {_render_summary_cards(payload)}
    <section class="dashboard-grid">
      {''.join(chart_sections)}
      <section class="detail-card">
        <h2>Run Details</h2>
        <dl class="detail-grid">
          {detail_rows}
        </dl>
        {_render_artifact_links(payload)}
      </section>
      {failure_note}
    </section>
  </main>
</body>
</html>
"""


def write_training_dashboard(
    save_dir: str,
    payload: dict[str, Any],
    refresh_seconds: int = REFRESH_SECONDS,
) -> tuple[str, str]:
    dashboard_dir = _dashboard_dir(save_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["artifacts"] = _artifact_paths(dashboard_dir)
    json_path = dashboard_dir / DASHBOARD_JSON_NAME
    html_path = dashboard_dir / DASHBOARD_HTML_NAME
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_training_dashboard_html(payload, refresh_seconds=refresh_seconds))
    return json_path.as_posix(), html_path.as_posix()


def load_dashboard_payload(save_dir: str) -> Optional[dict[str, Any]]:
    return _load_json(_dashboard_dir(save_dir) / DASHBOARD_JSON_NAME)


def _load_history_from_lightning_logs(save_dir: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}

    candidates = sorted(save_dir.glob("lightning_logs/version_*"))
    if not candidates:
        fallback = save_dir / "lightning_logs"
        if fallback.exists():
            candidates = [fallback]
    for candidate in reversed(candidates):
        try:
            accumulator = EventAccumulator(candidate.as_posix())
            accumulator.Reload()
            scalar_tags = list(accumulator.Tags().get("scalars", []))
        except Exception:
            continue
        if scalar_tags:
            break
    else:
        return {}

    epoch_steps: list[int] = []
    epoch_values: list[int] = []
    if "epoch" in scalar_tags:
        for event in accumulator.Scalars("epoch"):
            epoch_steps.append(int(event.step))
            epoch_values.append(int(round(float(event.value))) + 1)

    def lookup_epoch(step: int) -> Optional[int]:
        if not epoch_steps:
            return None
        index = bisect_right(epoch_steps, step) - 1
        if index < 0:
            return None
        return epoch_values[index]

    history: dict[str, list[dict[str, Any]]] = {}
    for tag in scalar_tags:
        if tag in ("hp_metric", "epoch"):
            continue
        points = []
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            point = {"step": step, "value": float(event.value)}
            epoch = lookup_epoch(step)
            if epoch is not None:
                point["epoch"] = epoch
            points.append(point)
        if points:
            history[tag] = points
    return history


def backfill_training_dashboard(
    save_dir: str,
    status: str = "unknown",
    refresh_seconds: int = REFRESH_SECONDS,
) -> tuple[dict[str, Any], str, str]:
    dashboard_dir = _dashboard_dir(save_dir)
    existing = load_dashboard_payload(dashboard_dir.as_posix())
    if existing is not None:
        json_path, html_path = write_training_dashboard(
            save_dir=dashboard_dir.as_posix(),
            payload=existing,
            refresh_seconds=refresh_seconds,
        )
        return existing, json_path, html_path

    history = _load_history_from_lightning_logs(dashboard_dir)
    payload = build_dashboard_payload(
        save_dir=dashboard_dir.as_posix(),
        history=history,
        status=status,
        started_at=_utc_now_iso(),
    )
    json_path, html_path = write_training_dashboard(
        save_dir=dashboard_dir.as_posix(),
        payload=payload,
        refresh_seconds=refresh_seconds,
    )
    return payload, json_path, html_path


def _latest_metric_entry(history: dict[str, list[dict[str, Any]]], name: str) -> Optional[dict[str, Any]]:
    points = history.get(name) or []
    if not points:
        return None
    return dict(points[-1])


def _best_metric_entry(
    history: dict[str, list[dict[str, Any]]],
    name: str,
    lower_is_better: bool = True,
) -> Optional[dict[str, Any]]:
    points = history.get(name) or []
    if not points:
        return None
    key_fn = min if lower_is_better else max
    return dict(key_fn(points, key=lambda point: point["value"]))


def _build_summary(
    history: dict[str, list[dict[str, Any]]],
    save_dir: Path,
    max_epochs: Optional[int] = None,
    current_epoch: Optional[int] = None,
    latest_step: Optional[int] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
    checkpoints = _checkpoint_entries(save_dir)
    if current_epoch is None:
        epochs = [
            int(point["epoch"])
            for points in history.values()
            for point in points
            if point.get("epoch") is not None
        ]
        if epochs:
            current_epoch = max(epochs)
    if latest_step is None:
        steps = [
            int(point["step"])
            for points in history.values()
            for point in points
            if point.get("step") is not None
        ]
        if steps:
            latest_step = max(steps)
    latest_metrics = {}
    for metric_name in sorted(history):
        latest_entry = _latest_metric_entry(history, metric_name)
        if latest_entry is not None:
            latest_metrics[metric_name] = latest_entry["value"]
    best_metrics = {}
    best_val_loss = _best_metric_entry(history, "val_loss", lower_is_better=True)
    if best_val_loss is not None:
        best_metrics["val_loss"] = best_val_loss
    return {
        "current_epoch": current_epoch,
        "max_epochs": max_epochs,
        "latest_step": latest_step,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": device,
        "aws_batch_job_id": os.environ.get("AWS_BATCH_JOB_ID"),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "latest_metrics": latest_metrics,
        "best_metrics": best_metrics,
    }


def build_dashboard_payload(
    save_dir: str,
    history: dict[str, list[dict[str, Any]]],
    status: str,
    started_at: str,
    max_epochs: Optional[int] = None,
    current_epoch: Optional[int] = None,
    latest_step: Optional[int] = None,
    device: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    dashboard_dir = _dashboard_dir(save_dir)
    return {
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now_iso(),
        "error": error,
        "run_metadata": _load_json(dashboard_dir / "run_metadata.json") or {},
        "history": history,
        "summary": _build_summary(
            history=history,
            save_dir=dashboard_dir,
            max_epochs=max_epochs,
            current_epoch=current_epoch,
            latest_step=latest_step,
            device=device,
        ),
    }


class TrainingDashboardCallback(_LightningCallbackBase):
    """Persist a self-contained dashboard during training."""

    def __init__(
        self,
        save_dir: str,
        refresh_seconds: int = REFRESH_SECONDS,
    ):
        if pl is None:
            raise ImportError(
                "PyTorch Lightning is required for TrainingDashboardCallback. "
                "Install with `pip install pytorch-lightning`."
            )
        self.save_dir = save_dir
        self.refresh_seconds = refresh_seconds
        self.started_at = _utc_now_iso()
        self.history: dict[str, list[dict[str, Any]]] = {}
        self.error: Optional[str] = None

    def _append_metric(
        self,
        metric_name: str,
        value: Any,
        epoch: int,
        step: int,
    ) -> None:
        metric_value = _to_float(value)
        if metric_value is None:
            return
        bucket = self.history.setdefault(metric_name, [])
        point = {"epoch": int(epoch), "step": int(step), "value": metric_value}
        if bucket and bucket[-1]["epoch"] == point["epoch"]:
            bucket[-1] = point
            return
        bucket.append(point)

    def _write(self, trainer, status: str) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        device = None
        if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "root_device"):
            device = str(trainer.strategy.root_device)
        recorded_epochs = [
            int(points[-1]["epoch"])
            for points in self.history.values()
            if points and points[-1].get("epoch") is not None
        ]
        if recorded_epochs:
            current_epoch = max(recorded_epochs)
        elif status == "running":
            current_epoch = 0
        else:
            current_epoch = None
        max_epochs = getattr(trainer, "max_epochs", None)
        payload = build_dashboard_payload(
            save_dir=self.save_dir,
            history=self.history,
            status=status,
            started_at=self.started_at,
            max_epochs=int(max_epochs) if max_epochs is not None else None,
            current_epoch=current_epoch,
            latest_step=int(getattr(trainer, "global_step", 0)),
            device=device,
            error=self.error,
        )
        write_training_dashboard(
            save_dir=self.save_dir,
            payload=payload,
            refresh_seconds=self.refresh_seconds,
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        self._write(trainer, status="running")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        metrics = dict(getattr(trainer, "callback_metrics", {}))
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        step = int(getattr(trainer, "global_step", 0))
        for metric_name, value in metrics.items():
            if not str(metric_name).startswith("train_"):
                continue
            self._append_metric(str(metric_name), value, epoch=epoch, step=step)
        self._write(trainer, status="running")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = dict(getattr(trainer, "callback_metrics", {}))
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        step = int(getattr(trainer, "global_step", 0))
        for metric_name, value in metrics.items():
            if not str(metric_name).startswith("val_"):
                continue
            self._append_metric(str(metric_name), value, epoch=epoch, step=step)
        self._write(trainer, status="running")

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self.error = str(exception)
        self._write(trainer, status="failed")

    def on_fit_end(self, trainer, pl_module) -> None:
        status = "failed" if self.error else "completed"
        self._write(trainer, status=status)
