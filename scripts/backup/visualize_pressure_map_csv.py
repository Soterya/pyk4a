#!/usr/bin/env python3
"""
Visualize pressure maps from a session CSV like pressure_map_save_*.csv.

Self-contained: no imports from other project scripts.

Each row: timestamp_ns + pressure_data (JSON list of 1152 floats, node order 1..32 × 36 cells).
Rendering: log-pressure per cell (option 4), per-frame normalize to [0,1], 2× upscale, inferno.

Usage:
  python visualize_pressure_map_csv.py path/to/pressure_map_save_*.csv

Requires: numpy, matplotlib, scipy. For US Eastern timestamps, the ``tzdata`` package is
needed on some Windows Python builds (``pip install tzdata``).
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom

# --- pressure grid / heatmap (same math as visualize_sync_frames) ---
GRID_SHAPE = (8, 4)
MAX_VOLTAGE = 32767
EPSILON = 1e-6
VIEWER_PLAYBACK_FPS = 5
# US Eastern: EST/EDT via IANA (lazy: needs ``tzdata`` on some Windows installs).
_eastern_tz_cached: ZoneInfo | None = None


def _eastern_tz() -> ZoneInfo | None:
    global _eastern_tz_cached
    if _eastern_tz_cached is not None:
        return _eastern_tz_cached
    try:
        _eastern_tz_cached = ZoneInfo("America/New_York")
    except Exception:
        return None
    return _eastern_tz_cached


def _flatten_frame(frame_data: dict[int, list[float]]) -> list[float]:
    values: list[float] = []
    for node_id in range(1, 33):
        values.extend(frame_data.get(node_id, [32767] * 36))
    return values


def convert_voltage_to_pressure(voltage_values: list[float]) -> tuple[list[float], list[float]]:
    max_voltage = MAX_VOLTAGE
    pressures = [max_voltage - v for v in voltage_values]
    pressures_normalized = [p / max_voltage for p in pressures]
    return pressures, pressures_normalized


def convert_voltage_to_pressure_v2(voltage_values: list[float]) -> list[float]:
    max_voltage = MAX_VOLTAGE
    return [np.log((max_voltage + EPSILON) / (float(v) + EPSILON)) for v in voltage_values]


def create_heatmap_grid_from_flat(
    values_array: list[float], grid_shape: tuple[int, int] = (8, 4)
) -> np.ndarray:
    node_rows, node_cols = grid_shape
    sensor_grid_size = 6
    total_rows = node_rows * sensor_grid_size
    total_cols = node_cols * sensor_grid_size
    grid = np.zeros((total_rows, total_cols))
    node_id = 1
    for node_row in range(node_rows):
        for node_col in range(node_cols):
            if node_id <= 32:
                start_idx = (node_id - 1) * 36
                pressure_data = values_array[start_idx : start_idx + 36]
                if len(pressure_data) == 36:
                    arr = np.array(pressure_data, dtype=float)
                    arr = np.where(np.isfinite(arr), arr, np.nan)
                    sensor_data = arr.reshape((6, 6))
                else:
                    sensor_data = np.full((6, 6), np.nan)
                start_r = node_row * sensor_grid_size
                start_c = node_col * sensor_grid_size
                grid[
                    start_r : start_r + sensor_grid_size,
                    start_c : start_c + sensor_grid_size,
                ] = sensor_data
            node_id += 1
    return grid


def build_grid_for_frame_with_option(
    frame_data: dict[int, list[float]], option: int, grid_shape: tuple[int, int] = (8, 4)
) -> np.ndarray:
    flat = _flatten_frame(frame_data)
    if option == 2:
        _, p_norm = convert_voltage_to_pressure(flat)
        return create_heatmap_grid_from_flat(p_norm, grid_shape)
    if option in (3, 4):
        p2 = convert_voltage_to_pressure_v2(flat)
        return create_heatmap_grid_from_flat(p2, grid_shape)
    raise ValueError("only options 2–4 used for this viewer; use 4")


def build_raw_normalized_grid(frame_data: dict[int, list[float]]) -> np.ndarray:
    grid = build_grid_for_frame_with_option(frame_data, 4, GRID_SHAPE)
    grid = np.nan_to_num(grid, nan=0.0)
    g_min, g_max = np.min(grid), np.max(grid)
    if g_max > g_min:
        grid = (grid - g_min) / (g_max - g_min)
    return grid


def build_log_map_heatmap_2x(frame_data: dict[int, list[float]]) -> np.ndarray:
    grid_raw = build_raw_normalized_grid(frame_data)
    up = zoom(grid_raw, (2, 2), order=1)
    return np.asarray(up, dtype=np.float32, order="C")


def _timestamp_ns_to_iso_eastern(ns: int) -> str:
    """Convert Unix-epoch nanoseconds to ISO-8601 in US Eastern (EST/EDT from America/New_York)."""
    if ns <= 0:
        return "—"
    tz = _eastern_tz()
    if tz is None:
        return "— (install tzdata: pip install tzdata)"
    try:
        dt = datetime.fromtimestamp(ns / 1e9, tz=tz)
        abbr = dt.tzname() or "ET"
        return f"{dt.isoformat(timespec='milliseconds')} {abbr}"
    except (OSError, OverflowError, ValueError):
        return "—"


def flat1152_to_frame_data(values: list[float]) -> dict[int, list[float]]:
    if len(values) != 1152:
        raise ValueError(f"expected 1152 pressure samples, got {len(values)}")
    frame_data: dict[int, list[float]] = {}
    for node_id in range(1, 33):
        start = (node_id - 1) * 36
        frame_data[node_id] = values[start : start + 36]
    return frame_data


def load_frames(csv_path: Path) -> tuple[list[int], list[dict[int, list[float]]]]:
    timestamps_ns: list[int] = []
    frames: list[dict[int, list[float]]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pressure_data" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: expected columns including 'pressure_data'")
        ts_key = "timestamp_ns" if "timestamp_ns" in reader.fieldnames else None
        for row in reader:
            raw = row.get("pressure_data")
            if raw is None or raw == "":
                continue
            try:
                arr = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"{csv_path}: bad pressure_data JSON: {e}") from e
            if not isinstance(arr, list):
                raise ValueError("pressure_data must be a JSON list")
            frames.append(flat1152_to_frame_data([float(x) for x in arr]))
            if ts_key:
                ts_raw = row.get(ts_key, "")
                try:
                    timestamps_ns.append(int(ts_raw) if ts_raw != "" else 0)
                except ValueError:
                    timestamps_ns.append(0)
            else:
                timestamps_ns.append(len(timestamps_ns))
    return timestamps_ns, frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Pressure-only heatmap viewer (standalone).")
    parser.add_argument("csv", type=Path, help="pressure_map_save_*.csv path")
    args = parser.parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"Error: not a file: {csv_path}", file=sys.stderr)
        return 1

    print(f"Loading {csv_path.name}...", file=sys.stderr)
    timestamps_ns, frames = load_frames(csv_path)
    if not frames:
        print("Error: no data rows", file=sys.stderr)
        return 1
    n = len(frames)
    print(f"  {n} frames", file=sys.stderr)

    _bg = "black"
    _fg = "0.88"
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(_bg)
    fig.patch.set_edgecolor(_bg)
    ax.set_facecolor(_bg)
    fig.canvas.manager.set_window_title(f"Pressure map — {csv_path.name}")

    ix = 0
    up0 = build_log_map_heatmap_2x(frames[ix])
    im = ax.imshow(
        up0,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_anchor("C")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.12)
    cax.set_facecolor(_bg)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_tick_params(colors=_fg)
    cbar.outline.set_edgecolor("0.35")
    cbar.ax.tick_params(axis="y", which="minor", colors=_fg)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.35")

    def title_for(i: int) -> str:
        ts = timestamps_ns[i] if i < len(timestamps_ns) else 0
        iso_et = _timestamp_ns_to_iso_eastern(int(ts))
        return (
            f"Frame {i + 1}/{n} | timestamp_ns = {ts}\n"
            f"ISO (US Eastern): {iso_et}"
        )

    title_text = fig.suptitle(title_for(ix), fontsize=10, y=0.97, color=_fg)

    _slider_w = 0.62
    _slider_left = (1.0 - _slider_w) / 2.0
    ax_slider = fig.add_axes([_slider_left, 0.072, _slider_w, 0.028], facecolor=_bg)
    ax_slider.tick_params(colors=_fg, labelcolor=_fg)
    for spine in ax_slider.spines.values():
        spine.set_color("0.35")
    slider = Slider(
        ax_slider,
        "Frame",
        0,
        max(0, n - 1),
        valinit=0,
        valstep=1,
        color="steelblue",
    )
    slider.label.set_color(_fg)
    if hasattr(slider, "track"):
        slider.track.set_facecolor("0.22")
        slider.track.set_edgecolor("0.45")
    _play_w = 0.10
    _play_left = 0.5 - _play_w / 2.0
    play_ax = fig.add_axes((_play_left, 0.022, _play_w, 0.032), facecolor=_bg)
    play_ax.set_frame_on(True)
    for spine in play_ax.spines.values():
        spine.set_color("0.35")
    btn_play = Button(play_ax, "Play", color="0.22", hovercolor="0.32")
    btn_play.label.set_color(_fg)
    playback_ms = max(5.0, 1000.0 / float(VIEWER_PLAYBACK_FPS))
    fig.text(
        0.5,
        0.006,
        f"play: {VIEWER_PLAYBACK_FPS:g} fps",
        ha="center",
        fontsize=8,
        color="0.55",
        transform=fig.transFigure,
    )

    last_drawn: list[int] = [-1]

    def update(frame_index: float) -> None:
        i = int(frame_index)
        if i == last_drawn[0]:
            return
        last_drawn[0] = i
        im.set_data(build_log_map_heatmap_2x(frames[i]))
        im.set_clim(0.0, 1.0)
        title_text.set_text(title_for(i))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    playing = [False]

    def on_play_step(_frame: int) -> None:
        if not playing[0] or n <= 1:
            return
        cur = int(slider.val)
        nxt = (cur + 1) % n
        slider.set_val(nxt)

    playback_anim = FuncAnimation(
        fig,
        on_play_step,
        interval=playback_ms,
        repeat=True,
        cache_frame_data=False,
        blit=False,
        save_count=0,
    )
    playback_anim.event_source.stop()

    def on_play_click(_event) -> None:
        if n <= 1:
            return
        playing[0] = not playing[0]
        btn_play.label.set_text("Pause" if playing[0] else "Play")
        if playing[0]:
            playback_anim.event_source.start()
        else:
            playback_anim.event_source.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(on_play_click)

    def on_key(event) -> None:
        if event.key in ("left", "right"):
            current = int(slider.val)
            mx = n - 1
            if event.key == "left":
                new_val = max(0, current - 1)
            else:
                new_val = min(mx, current + 1)
            if new_val != current:
                slider.set_val(new_val)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.subplots_adjust(bottom=0.18, top=0.86, left=0.08, right=0.92)
    plt.show()
    try:
        playback_anim.event_source.stop()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
