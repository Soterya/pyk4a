import argparse
import csv
from bisect import bisect_left
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
class KinectDataRow:
    save_timestamp_ns: int
    frame_idx_master: int
    frame_idx_sub1: int
    frame_idx_sub2: int
    frame_idx_sub3: int
    frame_idx_sub4: int
    rgb_path_master: str
    rgb_path_sub1: str
    rgb_path_sub2: str
    rgb_path_sub3: str
    rgb_path_sub4: str
    depth_path_master: str


@dataclass
class OrbbecDataRow:
    save_timestamp_ns: int
    frame_idx_master: int
    frame_idx_subordinate: int
    rgb_path_master: str
    rgb_path_subordinate: str
    depth_path_master: str
    depth_path_subordinate: str


@dataclass
class PressureDataRow:
    frame_idx: int
    save_timestamp_ns: int
    pressure_data_path: str


@dataclass
class FusedDataRow:
    save_timestamp_ns: int
    frame_idx_master: int
    fused_depth_path: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create Kinect/Orbbec/Pressure composites from exported data companions "
            "using pressure timestamps as reference (nearest-neighbor timestamp matching)."
        )
    )
    parser.add_argument(
        "session_output_dir",
        type=Path,
        help=(
            "Session output directory containing kinect/, orbbec/, pressure/, or a raw data "
            "path under data/person_x/session_x/data_collection."
        ),
    )
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=None,
        help="Optional max allowed nearest-neighbor delta from pressure to Kinect/Orbbec.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of composites to save.",
    )
    parser.add_argument(
        "--keep-nonpositive-ts",
        action="store_true",
        help="Keep rows with non-positive timestamps (default drops them).",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=16,
        help="Scale factor per pressure sensor cell.",
    )
    parser.add_argument(
        "--depth-min-mm",
        type=float,
        default=0.0,
        help="Fixed minimum depth (mm) for depth colormap normalization.",
    )
    parser.add_argument(
        "--depth-max-mm",
        type=float,
        default=4000.0,
        help="Fixed maximum depth (mm) for depth colormap normalization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Default: "
            "<session_output_dir>/synced_data_from_pressure_kinect_orbbec"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate composite plot images. Otherwise only generate mapping CSV.",
    )
    return parser.parse_args()


def get_person_session_subpath_from_data_path(path: Path) -> Path:
    resolved = path.resolve()
    parts = resolved.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data" and i + 2 < len(parts):
            person_part = parts[i + 1]
            session_part = parts[i + 2]
            if person_part.startswith("person_") and session_part.startswith("session_"):
                return Path(person_part) / session_part
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data" and i + 1 < len(parts):
            return Path(parts[i + 1])
    return Path(resolved.name)


def resolve_session_output_dir(session_output_dir_arg: Path) -> Path:
    session_output_dir = session_output_dir_arg.resolve()
    if (
        (session_output_dir / "kinect").is_dir()
        and (session_output_dir / "orbbec").is_dir()
        and (session_output_dir / "pressure").is_dir()
    ):
        return session_output_dir

    repo_root = Path(__file__).resolve().parent.parent
    person_session_subpath = get_person_session_subpath_from_data_path(session_output_dir)
    inferred_output_dir = repo_root / "outputs" / person_session_subpath
    if (
        (inferred_output_dir / "kinect").is_dir()
        and (inferred_output_dir / "orbbec").is_dir()
        and (inferred_output_dir / "pressure").is_dir()
    ):
        return inferred_output_dir

    raise RuntimeError(
        "Could not resolve session output directory with kinect/, orbbec/, pressure/ folders from: "
        f"{session_output_dir_arg}"
    )


def find_nearest_index(target: int, sorted_values: list[int]):
    pos = bisect_left(sorted_values, target)
    best_idx = None
    best_delta = None
    for idx in (pos - 1, pos):
        if idx < 0 or idx >= len(sorted_values):
            continue
        delta = abs(sorted_values[idx] - target)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx, best_delta


def read_kinect_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[KinectDataRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = int(row["save_timestamp_ns_master"])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            rows.append(
                KinectDataRow(
                    save_timestamp_ns=ts,
                    frame_idx_master=int(row["frame_idx_master"]),
                    frame_idx_sub1=int(row["frame_idx_sub1"]),
                    frame_idx_sub2=int(row["frame_idx_sub2"]),
                    frame_idx_sub3=int(row["frame_idx_sub3"]),
                    frame_idx_sub4=int(row["frame_idx_sub4"]),
                    rgb_path_master=row["rgb_path_master"],
                    rgb_path_sub1=row["rgb_path_sub1"],
                    rgb_path_sub2=row["rgb_path_sub2"],
                    rgb_path_sub3=row["rgb_path_sub3"],
                    rgb_path_sub4=row["rgb_path_sub4"],
                    depth_path_master=row["depth_path_master"],
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns)
    return rows


def read_orbbec_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[OrbbecDataRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = int(row["save_timestamp_ns_master"])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            rows.append(
                OrbbecDataRow(
                    save_timestamp_ns=ts,
                    frame_idx_master=int(row["frame_idx_master"]),
                    frame_idx_subordinate=int(row["frame_idx_subordinate"]),
                    rgb_path_master=row["rgb_path_master"],
                    rgb_path_subordinate=row["rgb_path_subordinate"],
                    depth_path_master=row["depth_path_master"],
                    depth_path_subordinate=row["depth_path_subordinate"],
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns)
    return rows


def read_pressure_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[PressureDataRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = int(row["save_timestamp_ns"])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            rows.append(
                PressureDataRow(
                    frame_idx=int(row["frame_idx"]),
                    save_timestamp_ns=ts,
                    pressure_data_path=row["pressure_data_path"],
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns)
    return rows


def read_fused_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[FusedDataRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = int(row["save_timestamp_ns_master"])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            rows.append(
                FusedDataRow(
                    save_timestamp_ns=ts,
                    frame_idx_master=int(row["frame_idx_master"]),
                    fused_depth_path=row.get("fused_depth_path", ""),
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns)
    return rows


def colorize_depth_mm(depth_mm: np.ndarray, depth_min_mm: float, depth_max_mm: float) -> np.ndarray:
    if depth_max_mm <= depth_min_mm:
        raise ValueError("depth_max_mm must be greater than depth_min_mm")
    depth = depth_mm.astype(np.float32)
    depth_clipped = np.clip(depth, depth_min_mm, depth_max_mm)
    scale = 255.0 / (depth_max_mm - depth_min_mm)
    depth_norm = ((depth_clipped - depth_min_mm) * scale).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


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


def load_pressure_grid(npz_path: Path) -> np.ndarray | None:
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
                return None
    except Exception:
        return None
    return None


def colorize_pressure_grid(grid: np.ndarray, cell_size: int) -> np.ndarray:
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


def load_depth_visual(npz_path: Path, depth_min_mm: float, depth_max_mm: float) -> np.ndarray | None:
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as data:
            if "depth" in data:
                depth = data["depth"]
            else:
                return None
        if depth is None:
            return None
        depth = np.asarray(depth)
        if depth.ndim != 2:
            return None
        return colorize_depth_mm(depth, depth_min_mm, depth_max_mm)
    except Exception:
        return None


def load_fused_depth_visual(path: Path, depth_min_mm: float, depth_max_mm: float) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                if "depth" not in data:
                    return None
                depth = np.asarray(data["depth"])
            if depth.ndim != 2:
                return None
            return colorize_depth_mm(depth, depth_min_mm, depth_max_mm)
    except Exception:
        return None
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
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)

    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = resized
    return panel


def overlay_text(panel, title: str, frame_idx: int, save_ts_ns: int):
    h, w = panel.shape[:2]
    overlay_h = 64
    y0 = h - overlay_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    panel[:] = cv2.addWeighted(overlay, 0.60, panel, 0.40, 0)
    cv2.putText(panel, title, (10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"idx={frame_idx} ts={save_ts_ns}",
        (10, y0 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 255, 120),
        1,
        cv2.LINE_AA,
    )
    return panel


def build_canvas(panel_specs: list[tuple[str, np.ndarray, int, int]], delta_kinect_ms: float, delta_orbbec_ms: float, pressure_ts: int):
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
            f"ref=pressure | delta_kinect_ms={delta_kinect_ms:.3f} | "
            f"delta_orbbec_ms={delta_orbbec_ms:.3f} | pressure_ts={pressure_ts}"
        ),
        (12, canvas_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
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

    session_output_dir = resolve_session_output_dir(args.session_output_dir)

    kinect_root = session_output_dir / "kinect"
    orbbec_root = session_output_dir / "orbbec"
    pressure_root = session_output_dir / "pressure"
    fused_root = session_output_dir / "fused_depth_maps"

    kinect_csv = kinect_root / "kinect_synced_data_companion.csv"
    if not kinect_csv.exists():
        legacy_kinect_csv = kinect_root / "kinect_synced_device_ts_data_companion.csv"
        if legacy_kinect_csv.exists():
            kinect_csv = legacy_kinect_csv
    orbbec_csv = orbbec_root / "orbbec_synced_data_companion.csv"
    pressure_csv = pressure_root / "pressure_synced_data_companion.csv"
    fused_csv = fused_root / "fused_depth_maps_companion.csv"

    if not kinect_csv.exists():
        raise FileNotFoundError(f"Missing Kinect data companion CSV: {kinect_csv}")
    if not orbbec_csv.exists():
        raise FileNotFoundError(f"Missing Orbbec data companion CSV: {orbbec_csv}")
    if not pressure_csv.exists():
        raise FileNotFoundError(f"Missing Pressure data companion CSV: {pressure_csv}")
    if not fused_csv.exists():
        raise FileNotFoundError(f"Missing fused depth companion CSV: {fused_csv}")

    kinect_rows = read_kinect_companion_csv(kinect_csv, args.keep_nonpositive_ts)
    orbbec_rows = read_orbbec_companion_csv(orbbec_csv, args.keep_nonpositive_ts)
    pressure_rows = read_pressure_companion_csv(pressure_csv, args.keep_nonpositive_ts)
    fused_rows = read_fused_companion_csv(fused_csv, args.keep_nonpositive_ts)
    if not kinect_rows or not orbbec_rows or not pressure_rows or not fused_rows:
        raise RuntimeError("No usable rows found in one or more companion CSV files.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (session_output_dir / "synced_data_from_pressure_kinect_orbbec")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if args.plot:
        plots_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv_path = output_dir / "synced_data_from_pressure_kinect_orbbec.csv"

    kinect_ts = [r.save_timestamp_ns for r in kinect_rows]
    orbbec_ts = [r.save_timestamp_ns for r in orbbec_rows]
    fused_ts = [r.save_timestamp_ns for r in fused_rows]
    max_delta_ns = int(args.max_delta_ms * 1_000_000.0) if args.max_delta_ms is not None else None

    saved = 0
    with mapping_csv_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "pressure_save_timestamp_ns_ref",
                "kinect_save_timestamp_ns_master",
                "orbbec_save_timestamp_ns_master",
                "fused_save_timestamp_ns_master",
                "pressure_frame_idx_ref",
                "kinect_frame_idx_master",
                "orbbec_frame_idx_master",
                "fused_frame_idx_master",
                "delta_pressure_to_kinect_ms",
                "delta_pressure_to_orbbec_ms",
                "delta_pressure_to_fused_ms",
                "pressure_data_path",
                "kinect_rgb_path_master",
                "kinect_rgb_path_sub1",
                "kinect_rgb_path_sub2",
                "kinect_rgb_path_sub3",
                "kinect_rgb_path_sub4",
                "kinect_depth_path_master",
                "orbbec_rgb_path_master",
                "orbbec_depth_path_master",
                "orbbec_rgb_path_subordinate",
                "orbbec_depth_path_subordinate",
                "fused_depth_path",
            ]
        )

        for p_row in pressure_rows:
            k_idx, delta_k_ns = find_nearest_index(p_row.save_timestamp_ns, kinect_ts)
            o_idx, delta_o_ns = find_nearest_index(p_row.save_timestamp_ns, orbbec_ts)
            f_idx, delta_f_ns = find_nearest_index(p_row.save_timestamp_ns, fused_ts)
            if (
                k_idx is None
                or o_idx is None
                or f_idx is None
                or delta_k_ns is None
                or delta_o_ns is None
                or delta_f_ns is None
            ):
                continue
            if max_delta_ns is not None and (delta_k_ns > max_delta_ns or delta_o_ns > max_delta_ns or delta_f_ns > max_delta_ns):
                continue

            k_row = kinect_rows[k_idx]
            o_row = orbbec_rows[o_idx]
            f_row = fused_rows[f_idx]

            k_rgb_master = cv2.imread(str(kinect_root / k_row.rgb_path_master), cv2.IMREAD_COLOR)
            k_rgb_sub1 = cv2.imread(str(kinect_root / k_row.rgb_path_sub1), cv2.IMREAD_COLOR)
            k_rgb_sub2 = cv2.imread(str(kinect_root / k_row.rgb_path_sub2), cv2.IMREAD_COLOR)
            k_rgb_sub3 = cv2.imread(str(kinect_root / k_row.rgb_path_sub3), cv2.IMREAD_COLOR)
            k_rgb_sub4 = cv2.imread(str(kinect_root / k_row.rgb_path_sub4), cv2.IMREAD_COLOR)
            k_depth_master = load_depth_visual(
                kinect_root / k_row.depth_path_master,
                args.depth_min_mm,
                args.depth_max_mm,
            )

            o_rgb_master = cv2.imread(str(orbbec_root / o_row.rgb_path_master), cv2.IMREAD_COLOR)
            o_depth_master = load_depth_visual(
                orbbec_root / o_row.depth_path_master,
                args.depth_min_mm,
                args.depth_max_mm,
            )
            o_rgb_sub = cv2.imread(str(orbbec_root / o_row.rgb_path_subordinate), cv2.IMREAD_COLOR)
            o_depth_sub = load_depth_visual(
                orbbec_root / o_row.depth_path_subordinate,
                args.depth_min_mm,
                args.depth_max_mm,
            )
            fused_img = None
            if f_row.fused_depth_path:
                fused_img = load_fused_depth_visual(
                    session_output_dir / f_row.fused_depth_path,
                    args.depth_min_mm,
                    args.depth_max_mm,
                )

            p_grid = load_pressure_grid(pressure_root / p_row.pressure_data_path)
            p_img = colorize_pressure_grid(p_grid, args.cell_size) if p_grid is not None else None

            required_imgs = [
                k_rgb_master,
                k_rgb_sub1,
                k_rgb_sub2,
                k_rgb_sub3,
                k_rgb_sub4,
                k_depth_master,
                o_rgb_master,
                o_depth_master,
                o_rgb_sub,
                o_depth_sub,
                fused_img,
                p_img,
            ]
            if any(img is None for img in required_imgs):
                continue

            delta_k_ms = delta_k_ns / 1_000_000.0
            delta_o_ms = delta_o_ns / 1_000_000.0
            delta_f_ms = delta_f_ns / 1_000_000.0

            panel_specs = [
                ("Kinect RGB master", k_rgb_master, k_row.frame_idx_master, k_row.save_timestamp_ns),
                ("Kinect RGB sub1", k_rgb_sub1, k_row.frame_idx_sub1, k_row.save_timestamp_ns),
                ("Kinect RGB sub2", k_rgb_sub2, k_row.frame_idx_sub2, k_row.save_timestamp_ns),
                ("Kinect RGB sub3", k_rgb_sub3, k_row.frame_idx_sub3, k_row.save_timestamp_ns),
                ("Kinect RGB sub4", k_rgb_sub4, k_row.frame_idx_sub4, k_row.save_timestamp_ns),
                ("Kinect Depth master", k_depth_master, k_row.frame_idx_master, k_row.save_timestamp_ns),
                ("Orbbec RGB master", o_rgb_master, o_row.frame_idx_master, o_row.save_timestamp_ns),
                ("Orbbec Depth master", o_depth_master, o_row.frame_idx_master, o_row.save_timestamp_ns),
                ("Orbbec RGB subordinate", o_rgb_sub, o_row.frame_idx_subordinate, o_row.save_timestamp_ns),
                ("Orbbec Depth subordinate", o_depth_sub, o_row.frame_idx_subordinate, o_row.save_timestamp_ns),
                ("Fused Depth", fused_img, f_row.frame_idx_master, f_row.save_timestamp_ns),
                ("Pressure", p_img, p_row.frame_idx, p_row.save_timestamp_ns),
            ]

            out_name = (
                f"pressure_ref_synced_data_"
                f"{p_row.frame_idx}_{p_row.save_timestamp_ns}__"
                f"{k_row.frame_idx_master}_{k_row.save_timestamp_ns}__"
                f"{o_row.frame_idx_master}_{o_row.save_timestamp_ns}__"
                f"{f_row.frame_idx_master}_{f_row.save_timestamp_ns}.png"
            )
            if args.plot:
                canvas = build_canvas(
                    panel_specs,
                    delta_kinect_ms=delta_k_ms,
                    delta_orbbec_ms=delta_o_ms,
                    pressure_ts=p_row.save_timestamp_ns,
                )
                cv2.imwrite(str(plots_dir / out_name), canvas)
            else:
                out_name = ""

            writer.writerow(
                [
                    p_row.save_timestamp_ns,
                    k_row.save_timestamp_ns,
                    o_row.save_timestamp_ns,
                    f_row.save_timestamp_ns,
                    p_row.frame_idx,
                    k_row.frame_idx_master,
                    o_row.frame_idx_master,
                    f_row.frame_idx_master,
                    f"{delta_k_ms:.6f}",
                    f"{delta_o_ms:.6f}",
                    f"{delta_f_ms:.6f}",
                    p_row.pressure_data_path,
                    k_row.rgb_path_master,
                    k_row.rgb_path_sub1,
                    k_row.rgb_path_sub2,
                    k_row.rgb_path_sub3,
                    k_row.rgb_path_sub4,
                    k_row.depth_path_master,
                    o_row.rgb_path_master,
                    o_row.depth_path_master,
                    o_row.rgb_path_subordinate,
                    o_row.depth_path_subordinate,
                    f_row.fused_depth_path,
                ]
            )

            saved += 1
            if args.max_pairs is not None and saved >= args.max_pairs:
                break

    print(f"Reference pressure rows: {len(pressure_rows)}")
    print(f"Kinect companion rows used: {len(kinect_rows)}")
    print(f"Orbbec companion rows used: {len(orbbec_rows)}")
    print(f"Fused companion rows used: {len(fused_rows)}")
    print(f"Saved composites: {saved}")
    print(f"Plot images generated: {args.plot}")
    print(f"Saved mapping CSV: {mapping_csv_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
