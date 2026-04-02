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


@dataclass
class Row:
    kinect_save_timestamp_ns_master: int
    orbbec_save_timestamp_ns_master: int
    fused_save_timestamp_ns_master: int
    kinect_frame_idx_master: int
    orbbec_frame_idx_master: int
    fused_frame_idx_master: int
    delta_ms: float
    delta_to_fused_ms: float
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
        description="Generate plots from person_x_session_x_rgb_depth.csv (trimmed synced data)."
    )
    parser.add_argument(
        "--trimmed-csv",
        type=Path,
        required=True,
        help="Path to person_x_session_x_rgb_depth.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Default: <session>/synced_data_from_kinect_orbbec/plots_trimmed",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of rows to plot.")
    parser.add_argument("--depth-min-mm", type=float, default=0.0, help="Depth colormap minimum in mm.")
    parser.add_argument("--depth-max-mm", type=float, default=4000.0, help="Depth colormap maximum in mm.")
    return parser.parse_args()


def _as_int(s: str) -> int:
    return int(s)


def _as_float(s: str) -> float:
    return float(s)


def read_trimmed_csv(path: Path):
    rows: list[Row] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        h2i = {h: i for i, h in enumerate(header)}
        for raw in reader:
            if len(raw) >= 20:
                # Handle known shifted schema from upstream CSV generation.
                row = Row(
                    kinect_save_timestamp_ns_master=_as_int(raw[0]),
                    orbbec_save_timestamp_ns_master=_as_int(raw[1]),
                    fused_save_timestamp_ns_master=_as_int(raw[2]),
                    kinect_frame_idx_master=_as_int(raw[3]),
                    orbbec_frame_idx_master=_as_int(raw[4]),
                    fused_frame_idx_master=_as_int(raw[5]),
                    delta_ms=_as_float(raw[6]),
                    delta_to_fused_ms=_as_float(raw[7]),
                    kinect_rgb_path_master=raw[8],
                    kinect_rgb_path_sub1=raw[9],
                    kinect_rgb_path_sub2=raw[10],
                    kinect_rgb_path_sub3=raw[11],
                    kinect_rgb_path_sub4=raw[12],
                    kinect_depth_path_master=raw[13],
                    orbbec_rgb_path_master=raw[14],
                    orbbec_depth_path_master=raw[15],
                    orbbec_rgb_path_subordinate=raw[16],
                    orbbec_depth_path_subordinate=raw[17],
                    fused_depth_path=raw[18],
                    combined_plot_filename=raw[19],
                )
            else:
                # Fallback for proper CSV shape.
                def gv(name: str):
                    idx = h2i.get(name)
                    return raw[idx] if idx is not None and idx < len(raw) else ""

                row = Row(
                    kinect_save_timestamp_ns_master=_as_int(gv("kinect_save_timestamp_ns_master")),
                    orbbec_save_timestamp_ns_master=_as_int(gv("orbbec_save_timestamp_ns_master")),
                    fused_save_timestamp_ns_master=_as_int(gv("fused_save_timestamp_ns_master") or gv("orbbec_save_timestamp_ns_master")),
                    kinect_frame_idx_master=_as_int(gv("kinect_frame_idx_master")),
                    orbbec_frame_idx_master=_as_int(gv("orbbec_frame_idx_master")),
                    fused_frame_idx_master=_as_int(gv("fused_frame_idx_master")),
                    delta_ms=_as_float(gv("delta_ms")),
                    delta_to_fused_ms=_as_float(gv("delta_to_fused_ms")),
                    kinect_rgb_path_master=gv("kinect_rgb_path_master"),
                    kinect_rgb_path_sub1=gv("kinect_rgb_path_sub1"),
                    kinect_rgb_path_sub2=gv("kinect_rgb_path_sub2"),
                    kinect_rgb_path_sub3=gv("kinect_rgb_path_sub3"),
                    kinect_rgb_path_sub4=gv("kinect_rgb_path_sub4"),
                    kinect_depth_path_master=gv("kinect_depth_path_master"),
                    orbbec_rgb_path_master=gv("orbbec_rgb_path_master"),
                    orbbec_depth_path_master=gv("orbbec_depth_path_master"),
                    orbbec_rgb_path_subordinate=gv("orbbec_rgb_path_subordinate"),
                    orbbec_depth_path_subordinate=gv("orbbec_depth_path_subordinate"),
                    fused_depth_path=gv("fused_depth_path"),
                    combined_plot_filename=gv("combined_plot_filename"),
                )
            rows.append(row)
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


def build_canvas(panel_specs: list[tuple[str, np.ndarray, int, int]], delta_ms: float, delta_f_ms: float):
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
        f"delta_to_orbbec_ms={delta_ms:.3f} | delta_to_fused_ms={delta_f_ms:.3f}",
        (12, canvas_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    if args.depth_max_mm <= args.depth_min_mm:
        raise ValueError("--depth-max-mm must be greater than --depth-min-mm")

    trimmed_csv = args.trimmed_csv.resolve()
    if not trimmed_csv.exists():
        raise FileNotFoundError(f"Missing trimmed CSV: {trimmed_csv}")

    session_output_dir = trimmed_csv.parent
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (session_output_dir / "synced_data_from_kinect_orbbec" / "plots_trimmed")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_trimmed_csv(trimmed_csv)
    saved = 0
    for row in rows:
        k_root = session_output_dir / "kinect"
        o_root = session_output_dir / "orbbec"

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

        required = [k_rgb_master, k_rgb_sub1, k_rgb_sub2, k_rgb_sub3, k_rgb_sub4, k_depth, o_rgb_master, o_depth_master, o_rgb_sub, o_depth_sub, fused_img]
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
        ]
        canvas = build_canvas(panel_specs, row.delta_ms, row.delta_to_fused_ms)

        out_name = row.combined_plot_filename or (
            f"trimmed_kinect_orbbec_{row.kinect_frame_idx_master}_{row.kinect_save_timestamp_ns_master}__"
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
