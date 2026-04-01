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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Match Kinect and Orbbec exported-data companion rows by nearest save timestamp "
            "and optionally save all-stream side-by-side composites."
        )
    )
    parser.add_argument(
        "session_output_dir",
        type=Path,
        help=(
            "Session output directory containing kinect/ and orbbec/, or a raw data "
            "path under data/person_x/session_x/data_collection."
        ),
    )
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=None,
        help="Optional max allowed nearest-neighbor delta in milliseconds.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of synced entries.",
    )
    parser.add_argument(
        "--keep-nonpositive-ts",
        action="store_true",
        help="Keep rows with non-positive timestamps (default drops them).",
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
            "<session_output_dir>/synced_data_from_kinect_orbbec"
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
    if (session_output_dir / "kinect").is_dir() and (session_output_dir / "orbbec").is_dir():
        return session_output_dir

    repo_root = Path(__file__).resolve().parent.parent
    person_session_subpath = get_person_session_subpath_from_data_path(session_output_dir)
    inferred_output_dir = repo_root / "outputs" / person_session_subpath
    if (inferred_output_dir / "kinect").is_dir() and (inferred_output_dir / "orbbec").is_dir():
        return inferred_output_dir

    raise RuntimeError(
        "Could not resolve session output directory with kinect/ and orbbec/ folders from: "
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


def load_depth_visual(npz_path: Path, depth_min_mm: float, depth_max_mm: float) -> np.ndarray | None:
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as data:
            if "depth" not in data:
                return None
            depth = np.asarray(data["depth"])
        if depth.ndim != 2:
            return None
        if depth_max_mm <= depth_min_mm:
            raise ValueError("depth_max_mm must be greater than depth_min_mm")
        depth = depth.astype(np.float32)
        depth_clipped = np.clip(depth, depth_min_mm, depth_max_mm)
        scale = 255.0 / (depth_max_mm - depth_min_mm)
        depth_norm = ((depth_clipped - depth_min_mm) * scale).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
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


def build_canvas(panel_specs: list[tuple[str, np.ndarray, int, int]], delta_ms: float):
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
        f"delta_ms={delta_ms:.3f}",
        (12, canvas_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    if args.depth_max_mm <= args.depth_min_mm:
        raise ValueError("--depth-max-mm must be greater than --depth-min-mm")

    session_output_dir = resolve_session_output_dir(args.session_output_dir)

    kinect_root = session_output_dir / "kinect"
    orbbec_root = session_output_dir / "orbbec"

    kinect_csv = kinect_root / "kinect_synced_data_companion.csv"
    if not kinect_csv.exists():
        legacy_kinect_csv = kinect_root / "kinect_synced_device_ts_data_companion.csv"
        if legacy_kinect_csv.exists():
            kinect_csv = legacy_kinect_csv
    orbbec_csv = orbbec_root / "orbbec_synced_data_companion.csv"

    if not kinect_csv.exists():
        raise FileNotFoundError(f"Missing Kinect data companion CSV: {kinect_csv}")
    if not orbbec_csv.exists():
        raise FileNotFoundError(f"Missing Orbbec data companion CSV: {orbbec_csv}")

    kinect_rows = read_kinect_companion_csv(kinect_csv, args.keep_nonpositive_ts)
    orbbec_rows = read_orbbec_companion_csv(orbbec_csv, args.keep_nonpositive_ts)
    if not kinect_rows or not orbbec_rows:
        raise RuntimeError("No usable rows found in one or both companion CSV files.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (session_output_dir / "synced_data_from_kinect_orbbec")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if args.plot:
        plots_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv_path = output_dir / "synced_data_from_kinect_orbbec.csv"

    orbbec_ts = [r.save_timestamp_ns for r in orbbec_rows]
    max_delta_ns = int(args.max_delta_ms * 1_000_000.0) if args.max_delta_ms is not None else None

    saved = 0
    with mapping_csv_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "kinect_save_timestamp_ns_master",
                "orbbec_save_timestamp_ns_master",
                "kinect_frame_idx_master",
                "orbbec_frame_idx_master",
                "delta_ms",
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
                "combined_plot_filename",
            ]
        )

        for k_row in kinect_rows:
            o_idx, delta_ns = find_nearest_index(k_row.save_timestamp_ns, orbbec_ts)
            if o_idx is None or delta_ns is None:
                continue
            if max_delta_ns is not None and delta_ns > max_delta_ns:
                continue

            o_row = orbbec_rows[o_idx]

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
            ]
            if any(img is None for img in required_imgs):
                continue

            delta_ms = delta_ns / 1_000_000.0
            out_name = (
                f"kinect_orbbec_synced_data_"
                f"{k_row.frame_idx_master}_{k_row.save_timestamp_ns}__"
                f"{o_row.frame_idx_master}_{o_row.save_timestamp_ns}.png"
            )
            if args.plot:
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
                ]
                canvas = build_canvas(panel_specs, delta_ms=delta_ms)
                cv2.imwrite(str(plots_dir / out_name), canvas)
            else:
                out_name = ""

            writer.writerow(
                [
                    k_row.save_timestamp_ns,
                    o_row.save_timestamp_ns,
                    k_row.frame_idx_master,
                    o_row.frame_idx_master,
                    f"{delta_ms:.6f}",
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
                    out_name,
                ]
            )

            saved += 1
            if args.max_pairs is not None and saved >= args.max_pairs:
                break

    print(f"Kinect companion rows used: {len(kinect_rows)}")
    print(f"Orbbec companion rows used: {len(orbbec_rows)}")
    print(f"Saved synced entries: {saved}")
    print(f"Plot images generated: {args.plot}")
    print(f"Saved mapping CSV: {mapping_csv_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
