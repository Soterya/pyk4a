import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PANEL_WIDTH = 480
PANEL_HEIGHT = 320
GRID_COLS = 4
GRID_ROWS = 3

NODE_ROWS = 8
NODE_COLS = 4
NODE_H = 6
NODE_W = 6
SENSORS_PER_NODE = NODE_H * NODE_W
NODES_TOTAL = NODE_ROWS * NODE_COLS
TOTAL_VALUES = NODES_TOTAL * SENSORS_PER_NODE
MAX_VOLTAGE = 32767.0
EPSILON = 1e-6


@dataclass
class Row:
    pressure_save_timestamp_ns_ref: int
    kinect_save_timestamp_ns_master: int
    orbbec_save_timestamp_ns_master: int
    fused_save_timestamp_ns_master: int
    pressure_frame_idx_ref: int
    kinect_frame_idx_master: int
    orbbec_frame_idx_master: int
    fused_frame_idx_master: int
    delta_pressure_to_kinect_ms: float
    delta_pressure_to_orbbec_ms: float
    delta_pressure_to_fused_ms: float
    pressure_data_path: str
    kinect_rgb_path_master: str
    kinect_rgb_path_sub1: str
    kinect_rgb_path_sub2: str
    kinect_rgb_path_sub3: str
    kinect_rgb_path_sub4: str
    kinect_depth_path_master: str
    orbbec_rgb_path_master: str
    orbbec_depth_path_master: str
    orbbec_rgb_path_subordinate: str
    orbbec_depth_path_subordinate: str
    fused_depth_path: str
    combined_plot_filename: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from person_x_session_x_pressure_rgb_depth.csv (trimmed synced data)."
    )
    parser.add_argument(
        "--trimmed-csv",
        type=Path,
        required=True,
        help="Path to person_x_session_x_pressure_rgb_depth.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Default: <session>/synced_data_from_pressure_kinect_orbbec/plots_trimmed",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of rows to plot.")
    parser.add_argument("--cell-size", type=int, default=16, help="Scale factor per pressure sensor cell.")
    parser.add_argument("--depth-min-mm", type=float, default=0.0, help="Depth colormap minimum in mm.")
    parser.add_argument("--depth-max-mm", type=float, default=4000.0, help="Depth colormap maximum in mm.")
    return parser.parse_args()


def read_trimmed_csv(path: Path):
    rows: list[Row] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    pressure_save_timestamp_ns_ref=int(r["pressure_save_timestamp_ns_ref"]),
                    kinect_save_timestamp_ns_master=int(r["kinect_save_timestamp_ns_master"]),
                    orbbec_save_timestamp_ns_master=int(r["orbbec_save_timestamp_ns_master"]),
                    fused_save_timestamp_ns_master=int(r["fused_save_timestamp_ns_master"]),
                    pressure_frame_idx_ref=int(r["pressure_frame_idx_ref"]),
                    kinect_frame_idx_master=int(r["kinect_frame_idx_master"]),
                    orbbec_frame_idx_master=int(r["orbbec_frame_idx_master"]),
                    fused_frame_idx_master=int(r["fused_frame_idx_master"]),
                    delta_pressure_to_kinect_ms=float(r["delta_pressure_to_kinect_ms"]),
                    delta_pressure_to_orbbec_ms=float(r["delta_pressure_to_orbbec_ms"]),
                    delta_pressure_to_fused_ms=float(r["delta_pressure_to_fused_ms"]),
                    pressure_data_path=r["pressure_data_path"],
                    kinect_rgb_path_master=r["kinect_rgb_path_master"],
                    kinect_rgb_path_sub1=r["kinect_rgb_path_sub1"],
                    kinect_rgb_path_sub2=r["kinect_rgb_path_sub2"],
                    kinect_rgb_path_sub3=r["kinect_rgb_path_sub3"],
                    kinect_rgb_path_sub4=r["kinect_rgb_path_sub4"],
                    kinect_depth_path_master=r["kinect_depth_path_master"],
                    orbbec_rgb_path_master=r["orbbec_rgb_path_master"],
                    orbbec_depth_path_master=r["orbbec_depth_path_master"],
                    orbbec_rgb_path_subordinate=r["orbbec_rgb_path_subordinate"],
                    orbbec_depth_path_subordinate=r["orbbec_depth_path_subordinate"],
                    fused_depth_path=r["fused_depth_path"],
                    combined_plot_filename=r.get("combined_plot_filename", ""),
                )
            )
    return rows


def colorize_depth_mm(depth_mm: np.ndarray, depth_min_mm: float, depth_max_mm: float) -> np.ndarray:
    if depth_max_mm <= depth_min_mm:
        raise ValueError("depth_max_mm must be greater than depth_min_mm")
    depth = depth_mm.astype(np.float32)
    depth = np.clip(depth, depth_min_mm, depth_max_mm)
    depth_u8 = ((depth - depth_min_mm) * (255.0 / (depth_max_mm - depth_min_mm))).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def load_depth_visual(npz_path: Path, depth_min_mm: float, depth_max_mm: float):
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as data:
            if "depth" not in data:
                return None
            depth = np.asarray(data["depth"])
        if depth.ndim != 2:
            return None
        return colorize_depth_mm(depth, depth_min_mm, depth_max_mm)
    except Exception:
        return None


def build_pressure_grid(values: np.ndarray) -> np.ndarray:
    grid = np.zeros((NODE_ROWS * NODE_H, NODE_COLS * NODE_W), dtype=np.float32)
    for node_idx in range(NODES_TOTAL):
        start = node_idx * SENSORS_PER_NODE
        node_vals = values[start : start + SENSORS_PER_NODE].reshape(NODE_H, NODE_W)
        node_r = node_idx // NODE_COLS
        node_c = node_idx % NODE_COLS
        r0 = node_r * NODE_H
        c0 = node_c * NODE_W
        grid[r0 : r0 + NODE_H, c0 : c0 + NODE_W] = node_vals
    return grid


def convert_voltage_grid_to_log_pressure(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.nan)
    return np.log((MAX_VOLTAGE + EPSILON) / (values + EPSILON))


def normalize_grid_01(grid: np.ndarray) -> np.ndarray:
    g = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    g_min = float(np.min(g))
    g_max = float(np.max(g))
    if g_max > g_min:
        g = (g - g_min) / (g_max - g_min)
    return g


def load_pressure_grid(npz_path: Path):
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as data:
            if "grid" in data:
                return data["grid"].astype(np.float32)
            if "values" in data:
                values = data["values"].astype(np.float32).reshape(-1)
                if values.size >= TOTAL_VALUES:
                    return build_pressure_grid(values[:TOTAL_VALUES])
    except Exception:
        return None
    return None


def colorize_pressure_grid(grid: np.ndarray, cell_size: int):
    # Match visualize_pressure_map_csv.py heatmap style:
    # log-voltage pressure map -> per-frame [0,1] normalize -> 2x upscale -> inferno.
    log_pressure = convert_voltage_grid_to_log_pressure(grid)
    normalized = normalize_grid_01(log_pressure)
    upsampled = cv2.resize(
        normalized,
        (normalized.shape[1] * 2, normalized.shape[0] * 2),
        interpolation=cv2.INTER_LINEAR,
    )
    img8 = np.clip(upsampled * 255.0, 0, 255).astype(np.uint8)
    frame = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    frame = cv2.resize(
        frame,
        (frame.shape[1] * cell_size, frame.shape[0] * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return frame


def fit_to_panel_keep_aspect(image, width: int, height: int):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if image is None:
        return panel
    ih, iw = image.shape[:2]
    if ih <= 0 or iw <= 0:
        return panel
    scale = min(width / iw, height / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = resized
    return panel


def overlay_text(panel, title: str, frame_idx: int, ts_ns: int):
    h, w = panel.shape[:2]
    y0 = h - 64
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    panel[:] = cv2.addWeighted(overlay, 0.60, panel, 0.40, 0)
    cv2.putText(panel, title, (10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"idx={frame_idx} ts={ts_ns}", (10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 120), 1, cv2.LINE_AA)
    return panel


def build_canvas(panel_specs: list[tuple[str, np.ndarray, int, int]], dk_ms: float, do_ms: float, df_ms: float, p_ts: int):
    canvas_h = GRID_ROWS * PANEL_HEIGHT + 40
    canvas_w = GRID_COLS * PANEL_WIDTH
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for i, (title, img, frame_idx, ts) in enumerate(panel_specs):
        row = i // GRID_COLS
        col = i % GRID_COLS
        if row >= GRID_ROWS:
            break
        panel = fit_to_panel_keep_aspect(img, PANEL_WIDTH, PANEL_HEIGHT)
        panel = overlay_text(panel, title, frame_idx, ts)
        y0 = row * PANEL_HEIGHT
        x0 = col * PANEL_WIDTH
        canvas[y0:y0 + PANEL_HEIGHT, x0:x0 + PANEL_WIDTH] = panel

    cv2.rectangle(canvas, (0, GRID_ROWS * PANEL_HEIGHT), (canvas_w, canvas_h), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        (
            f"ref=pressure | dK={dk_ms:.3f}ms | dO={do_ms:.3f}ms | dF={df_ms:.3f}ms | "
            f"pressure_ts={p_ts}"
        ),
        (12, canvas_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    if args.cell_size < 1:
        raise ValueError("--cell-size must be >= 1")
    if args.depth_max_mm <= args.depth_min_mm:
        raise ValueError("--depth-max-mm must be greater than --depth-min-mm")

    trimmed_csv = args.trimmed_csv.resolve()
    if not trimmed_csv.exists():
        raise FileNotFoundError(f"Missing trimmed CSV: {trimmed_csv}")

    session_output_dir = trimmed_csv.parent
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (session_output_dir / "synced_data_from_pressure_kinect_orbbec" / "plots_trimmed")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_trimmed_csv(trimmed_csv)
    saved = 0
    for row in rows:
        k_root = session_output_dir / "kinect"
        o_root = session_output_dir / "orbbec"
        p_root = session_output_dir / "pressure"

        k_rgb_master = cv2.imread(str(k_root / row.kinect_rgb_path_master), cv2.IMREAD_COLOR)
        k_rgb_sub1 = cv2.imread(str(k_root / row.kinect_rgb_path_sub1), cv2.IMREAD_COLOR)
        k_rgb_sub2 = cv2.imread(str(k_root / row.kinect_rgb_path_sub2), cv2.IMREAD_COLOR)
        k_rgb_sub3 = cv2.imread(str(k_root / row.kinect_rgb_path_sub3), cv2.IMREAD_COLOR)
        k_rgb_sub4 = cv2.imread(str(k_root / row.kinect_rgb_path_sub4), cv2.IMREAD_COLOR)
        k_depth = load_depth_visual(k_root / row.kinect_depth_path_master, args.depth_min_mm, args.depth_max_mm)

        o_rgb_master = cv2.imread(str(o_root / row.orbbec_rgb_path_master), cv2.IMREAD_COLOR)
        o_depth_master = load_depth_visual(o_root / row.orbbec_depth_path_master, args.depth_min_mm, args.depth_max_mm)
        o_rgb_sub = cv2.imread(str(o_root / row.orbbec_rgb_path_subordinate), cv2.IMREAD_COLOR)
        o_depth_sub = load_depth_visual(o_root / row.orbbec_depth_path_subordinate, args.depth_min_mm, args.depth_max_mm)

        fused_img = load_depth_visual(session_output_dir / row.fused_depth_path, args.depth_min_mm, args.depth_max_mm)

        p_grid = load_pressure_grid(p_root / row.pressure_data_path)
        p_img = colorize_pressure_grid(p_grid, args.cell_size) if p_grid is not None else None

        required = [k_rgb_master, k_rgb_sub1, k_rgb_sub2, k_rgb_sub3, k_rgb_sub4, k_depth, o_rgb_master, o_depth_master, o_rgb_sub, o_depth_sub, fused_img, p_img]
        if any(img is None for img in required):
            continue

        panel_specs = [
            ("Kinect RGB master", k_rgb_master, row.kinect_frame_idx_master, row.kinect_save_timestamp_ns_master),
            ("Kinect RGB sub1", k_rgb_sub1, row.kinect_frame_idx_master, row.kinect_save_timestamp_ns_master),
            ("Kinect RGB sub2", k_rgb_sub2, row.kinect_frame_idx_master, row.kinect_save_timestamp_ns_master),
            ("Kinect RGB sub3", k_rgb_sub3, row.kinect_frame_idx_master, row.kinect_save_timestamp_ns_master),
            ("Kinect RGB sub4", k_rgb_sub4, row.kinect_frame_idx_master, row.kinect_save_timestamp_ns_master),
            ("Kinect Depth master", k_depth, row.kinect_frame_idx_master, row.kinect_save_timestamp_ns_master),
            ("Orbbec RGB master", o_rgb_master, row.orbbec_frame_idx_master, row.orbbec_save_timestamp_ns_master),
            ("Orbbec Depth master", o_depth_master, row.orbbec_frame_idx_master, row.orbbec_save_timestamp_ns_master),
            ("Orbbec RGB subordinate", o_rgb_sub, row.orbbec_frame_idx_master, row.orbbec_save_timestamp_ns_master),
            ("Orbbec Depth subordinate", o_depth_sub, row.orbbec_frame_idx_master, row.orbbec_save_timestamp_ns_master),
            ("Fused Depth", fused_img, row.fused_frame_idx_master, row.fused_save_timestamp_ns_master),
            ("Pressure", p_img, row.pressure_frame_idx_ref, row.pressure_save_timestamp_ns_ref),
        ]

        canvas = build_canvas(
            panel_specs,
            row.delta_pressure_to_kinect_ms,
            row.delta_pressure_to_orbbec_ms,
            row.delta_pressure_to_fused_ms,
            row.pressure_save_timestamp_ns_ref,
        )

        out_name = row.combined_plot_filename or (
            f"trimmed_pressure_ref_{row.pressure_frame_idx_ref}_{row.pressure_save_timestamp_ns_ref}__"
            f"{row.kinect_frame_idx_master}_{row.kinect_save_timestamp_ns_master}__"
            f"{row.orbbec_frame_idx_master}_{row.orbbec_save_timestamp_ns_master}__"
            f"{row.fused_frame_idx_master}_{row.fused_save_timestamp_ns_master}.png"
        )
        cv2.imwrite(str(output_dir / out_name), canvas)
        saved += 1
        if args.max_pairs is not None and saved >= args.max_pairs:
            break

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Saved plots: {saved}")
    print(f"[INFO] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
