import argparse
import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PANEL_WIDTH = 640
PANEL_HEIGHT = 540


@dataclass
class PlotCompanionRow:
    save_timestamp_ns: int
    frame_idx: int
    plot_filename: str


@dataclass
class PressureCompanionRow:
    frame_idx: int
    save_timestamp_ns: int
    pressure_data_path: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create Kinect/Orbbec/Pressure composites using pressure timestamps as reference "
            "(nearest-neighbor timestamp matching)."
        )
    )
    parser.add_argument("session_dir", type=Path, help="Session directory with Kinect + Orbbec plot companions.")
    parser.add_argument(
        "--pressure-companion",
        type=Path,
        default=None,
        help="Path to pressure companion CSV. Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--pressure-plots-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --pressure-data-dir.",
    )
    parser.add_argument(
        "--pressure-data-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for pressure_data_path entries from companion CSV. "
            "Defaults to companion CSV parent."
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
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <session_dir>/kinect_orbbec_pressure_side_by_side",
    )
    return parser.parse_args()


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


def read_plot_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[PlotCompanionRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = int(row["save_timestamp_ns_master"])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            rows.append(
                PlotCompanionRow(
                    save_timestamp_ns=ts,
                    frame_idx=int(row["frame_idx_master"]),
                    plot_filename=row["plot_filename"],
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns)
    return rows


def read_pressure_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[PressureCompanionRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_key = "save_timestamp_ns" if "save_timestamp_ns" in row else "save_timestamps_ns"
            ts = int(row[ts_key])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            pressure_data_path = row.get("pressure_data_path", "")
            rows.append(
                PressureCompanionRow(
                    frame_idx=int(row["frame_idx"]),
                    save_timestamp_ns=ts,
                    pressure_data_path=pressure_data_path,
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns)
    return rows


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
    overlay_h = 70
    y0 = h - overlay_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    panel[:] = cv2.addWeighted(overlay, 0.60, panel, 0.40, 0)
    cv2.putText(panel, title, (12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"frame_idx={frame_idx}  save_timestamp_ns={save_ts_ns}",
        (12, y0 + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return panel


def build_canvas(
    kinect_img,
    orbbec_img,
    pressure_img,
    k_row: PlotCompanionRow,
    o_row: PlotCompanionRow,
    p_row: PressureCompanionRow,
    delta_kinect_ms: float,
    delta_orbbec_ms: float,
):
    left = fit_to_panel_keep_aspect(kinect_img, PANEL_WIDTH, PANEL_HEIGHT)
    center = fit_to_panel_keep_aspect(orbbec_img, PANEL_WIDTH, PANEL_HEIGHT)
    right = fit_to_panel_keep_aspect(pressure_img, PANEL_WIDTH, PANEL_HEIGHT)
    left = overlay_text(left, "Kinect", k_row.frame_idx, k_row.save_timestamp_ns)
    center = overlay_text(center, "Orbbec", o_row.frame_idx, o_row.save_timestamp_ns)
    right = overlay_text(right, "Pressure", p_row.frame_idx, p_row.save_timestamp_ns)

    canvas = np.zeros((PANEL_HEIGHT + 46, PANEL_WIDTH * 3, 3), dtype=np.uint8)
    canvas[:PANEL_HEIGHT, :PANEL_WIDTH] = left
    canvas[:PANEL_HEIGHT, PANEL_WIDTH:PANEL_WIDTH * 2] = center
    canvas[:PANEL_HEIGHT, PANEL_WIDTH * 2:] = right
    cv2.rectangle(canvas, (0, PANEL_HEIGHT), (canvas.shape[1], canvas.shape[0]), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        (
            f"ref=pressure | delta_kinect_ms={delta_kinect_ms:.3f} | delta_orbbec_ms={delta_orbbec_ms:.3f} | "
            f"pressure_ts={p_row.save_timestamp_ns}"
        ),
        (12, PANEL_HEIGHT + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def resolve_pressure_companion(session_dir: Path, override: Path | None):
    if override is not None:
        return override.resolve()

    candidates = [
        session_dir / "pressure_maps_sync_by_device_ts_plots" / "pressure_maps_synced_companion.csv",
        session_dir.parent / "pressure_data" / "pressure_maps_sync_by_device_ts_plots" / "pressure_maps_synced_companion.csv",
        session_dir.parent.parent / "pressure_data" / "pressure_maps_sync_by_device_ts_plots" / "pressure_maps_synced_companion.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


NODE_ROWS = 8
NODE_COLS = 4
NODE_H = 6
NODE_W = 6
SENSORS_PER_NODE = NODE_H * NODE_W
NODES_TOTAL = NODE_ROWS * NODE_COLS
TOTAL_VALUES = NODES_TOTAL * SENSORS_PER_NODE


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


def colorize_pressure_grid(grid: np.ndarray, cell_size: int = 16) -> np.ndarray:
    max_value = float(np.max(grid))
    max_value = max(max_value, 1.0)
    img8 = np.clip((grid / max_value) * 255.0, 0, 255).astype(np.uint8)
    frame = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    frame = cv2.resize(
        frame,
        (grid.shape[1] * cell_size, grid.shape[0] * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )

    for r in range(1, NODE_ROWS):
        y = r * NODE_H * cell_size
        cv2.line(frame, (0, y), (frame.shape[1], y), (255, 255, 255), 2)
    for c in range(1, NODE_COLS):
        x = c * NODE_W * cell_size
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 255), 2)
    return frame


def main():
    args = parse_args()
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        raise RuntimeError(f"Session directory does not exist: {session_dir}")

    kinect_csv = session_dir / "kinect_rgb_sync_by_device_ts_plots" / "kinect_synced_device_ts_companion.csv"
    orbbec_csv = session_dir / "orbbec_master__orbbec_subordinate_sync_plots_with_save_ts" / "orbbec_synced_companion.csv"
    pressure_csv = resolve_pressure_companion(session_dir, args.pressure_companion)

    if not kinect_csv.exists():
        raise FileNotFoundError(f"Missing Kinect companion CSV: {kinect_csv}")
    if not orbbec_csv.exists():
        raise FileNotFoundError(f"Missing Orbbec companion CSV: {orbbec_csv}")
    if not pressure_csv.exists():
        raise FileNotFoundError(f"Missing Pressure companion CSV: {pressure_csv}")

    kinect_rows = read_plot_companion_csv(kinect_csv, args.keep_nonpositive_ts)
    orbbec_rows = read_plot_companion_csv(orbbec_csv, args.keep_nonpositive_ts)
    pressure_rows = read_pressure_companion_csv(pressure_csv, args.keep_nonpositive_ts)
    if not kinect_rows or not orbbec_rows or not pressure_rows:
        raise RuntimeError("No usable rows found in one or more companion CSV files.")

    output_dir = args.output_dir.resolve() if args.output_dir else (session_dir / "kinect_orbbec_pressure_side_by_side")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv_path = output_dir / "pressure_ref_kinect_orbbec_nn_mapping.csv"

    pressure_data_root = (
        args.pressure_data_dir.resolve()
        if args.pressure_data_dir
        else args.pressure_plots_dir.resolve() if args.pressure_plots_dir else pressure_csv.parent
    )
    kinect_ts = [r.save_timestamp_ns for r in kinect_rows]
    orbbec_ts = [r.save_timestamp_ns for r in orbbec_rows]
    max_delta_ns = int(args.max_delta_ms * 1_000_000.0) if args.max_delta_ms is not None else None

    saved = 0
    with mapping_csv_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "pressure_save_timestamp_ns_ref",
                "kinect_save_timestamp_ns_master",
                "orbbec_save_timestamp_ns_master",
                "pressure_frame_idx_ref",
                "kinect_frame_idx_master",
                "orbbec_frame_idx_master",
                "delta_pressure_to_kinect_ms",
                "delta_pressure_to_orbbec_ms",
                "pressure_data_path",
                "kinect_plot_filename",
                "orbbec_plot_filename",
                "combined_plot_filename",
            ]
        )

        for p_row in pressure_rows:
            k_idx, delta_k_ns = find_nearest_index(p_row.save_timestamp_ns, kinect_ts)
            o_idx, delta_o_ns = find_nearest_index(p_row.save_timestamp_ns, orbbec_ts)
            if (
                k_idx is None
                or o_idx is None
                or delta_k_ns is None
                or delta_o_ns is None
            ):
                continue
            if max_delta_ns is not None and (delta_k_ns > max_delta_ns or delta_o_ns > max_delta_ns):
                continue

            k_row = kinect_rows[k_idx]
            o_row = orbbec_rows[o_idx]
            k_img_path = kinect_csv.parent / k_row.plot_filename
            o_img_path = orbbec_csv.parent / o_row.plot_filename
            p_npz_path = pressure_data_root / p_row.pressure_data_path
            if not k_img_path.exists() or not o_img_path.exists() or not p_npz_path.exists():
                continue

            k_img = cv2.imread(str(k_img_path), cv2.IMREAD_COLOR)
            o_img = cv2.imread(str(o_img_path), cv2.IMREAD_COLOR)
            p_grid = load_pressure_grid(p_npz_path)
            p_img = colorize_pressure_grid(p_grid) if p_grid is not None else None
            if k_img is None or o_img is None or p_img is None:
                continue

            delta_k_ms = delta_k_ns / 1_000_000.0
            delta_o_ms = delta_o_ns / 1_000_000.0
            canvas = build_canvas(k_img, o_img, p_img, k_row, o_row, p_row, delta_k_ms, delta_o_ms)
            out_name = (
                f"pressure_ref_synced_"
                f"{p_row.frame_idx}_{p_row.save_timestamp_ns}__"
                f"{k_row.frame_idx}_{k_row.save_timestamp_ns}__"
                f"{o_row.frame_idx}_{o_row.save_timestamp_ns}.png"
            )
            cv2.imwrite(str(output_dir / out_name), canvas)

            writer.writerow(
                [
                    p_row.save_timestamp_ns,
                    k_row.save_timestamp_ns,
                    o_row.save_timestamp_ns,
                    p_row.frame_idx,
                    k_row.frame_idx,
                    o_row.frame_idx,
                    f"{delta_k_ms:.6f}",
                    f"{delta_o_ms:.6f}",
                    p_row.pressure_data_path,
                    k_row.plot_filename,
                    o_row.plot_filename,
                    out_name,
                ]
            )

            saved += 1
            if args.max_pairs is not None and saved >= args.max_pairs:
                break

    print(f"Reference pressure rows: {len(pressure_rows)}")
    print(f"Kinect companion rows used: {len(kinect_rows)}")
    print(f"Orbbec companion rows used: {len(orbbec_rows)}")
    print(f"Saved composites: {saved}")
    print(f"Saved mapping CSV: {mapping_csv_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
