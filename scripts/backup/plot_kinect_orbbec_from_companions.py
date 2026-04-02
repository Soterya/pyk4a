import argparse
import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PANEL_WIDTH = 960
PANEL_HEIGHT = 540


@dataclass
class CompanionRow:
    save_timestamp_ns_master: int
    frame_idx_master: int
    plot_filename: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Match Kinect and Orbbec companion CSV rows by nearest save_timestamp_ns_master, "
            "and save side-by-side plot composites."
        )
    )
    parser.add_argument("session_dir", type=Path, help="Session directory containing both companion CSVs.")
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
        help="Optional cap on number of side-by-side composites to save.",
    )
    parser.add_argument(
        "--keep-nonpositive-ts",
        action="store_true",
        help="Keep rows with save_timestamp_ns_master <= 0 (default drops them).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <session_dir>/kinect_orbbec_side_by_side",
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


def read_companion_csv(path: Path, keep_nonpositive_ts: bool):
    rows: list[CompanionRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = int(row["save_timestamp_ns_master"])
            if not keep_nonpositive_ts and ts <= 0:
                continue
            rows.append(
                CompanionRow(
                    save_timestamp_ns_master=ts,
                    frame_idx_master=int(row["frame_idx_master"]),
                    plot_filename=row["plot_filename"],
                )
            )
    rows.sort(key=lambda r: r.save_timestamp_ns_master)
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
        f"master_frame_idx={frame_idx}  save_timestamp_ns={save_ts_ns}",
        (12, y0 + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return panel


def build_canvas(kinect_img, orbbec_img, k_row: CompanionRow, o_row: CompanionRow, delta_ms: float):
    left = fit_to_panel_keep_aspect(kinect_img, PANEL_WIDTH, PANEL_HEIGHT)
    right = fit_to_panel_keep_aspect(orbbec_img, PANEL_WIDTH, PANEL_HEIGHT)
    left = overlay_text(left, "Kinect", k_row.frame_idx_master, k_row.save_timestamp_ns_master)
    right = overlay_text(right, "Orbbec", o_row.frame_idx_master, o_row.save_timestamp_ns_master)

    canvas = np.zeros((PANEL_HEIGHT + 46, PANEL_WIDTH * 2, 3), dtype=np.uint8)
    canvas[:PANEL_HEIGHT, :PANEL_WIDTH] = left
    canvas[:PANEL_HEIGHT, PANEL_WIDTH:] = right
    cv2.rectangle(canvas, (0, PANEL_HEIGHT), (canvas.shape[1], canvas.shape[0]), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"delta_ms={delta_ms:.3f} | kinect_ts={k_row.save_timestamp_ns_master} | orbbec_ts={o_row.save_timestamp_ns_master}",
        (12, PANEL_HEIGHT + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        raise RuntimeError(f"Session directory does not exist: {session_dir}")

    kinect_csv = session_dir / "kinect_rgb_sync_by_device_ts_plots" / "kinect_synced_device_ts_companion.csv"
    orbbec_csv = session_dir / "orbbec_master__orbbec_subordinate_sync_plots_with_save_ts" / "orbbec_synced_companion.csv"
    if not kinect_csv.exists():
        raise FileNotFoundError(f"Missing Kinect companion CSV: {kinect_csv}")
    if not orbbec_csv.exists():
        raise FileNotFoundError(f"Missing Orbbec companion CSV: {orbbec_csv}")

    kinect_rows = read_companion_csv(kinect_csv, args.keep_nonpositive_ts)
    orbbec_rows = read_companion_csv(orbbec_csv, args.keep_nonpositive_ts)
    if not kinect_rows or not orbbec_rows:
        raise RuntimeError("No usable rows found in one or both companion CSV files.")

    output_dir = args.output_dir.resolve() if args.output_dir else (session_dir / "kinect_orbbec_side_by_side")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv_path = output_dir / "kinect_orbbec_nn_mapping.csv"

    orbbec_ts = [r.save_timestamp_ns_master for r in orbbec_rows]
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
                "kinect_plot_filename",
                "orbbec_plot_filename",
                "combined_plot_filename",
            ]
        )

        for k_row in kinect_rows:
            o_idx, delta_ns = find_nearest_index(k_row.save_timestamp_ns_master, orbbec_ts)
            if o_idx is None or delta_ns is None:
                continue
            if max_delta_ns is not None and delta_ns > max_delta_ns:
                continue

            o_row = orbbec_rows[o_idx]
            k_img_path = kinect_csv.parent / k_row.plot_filename
            o_img_path = orbbec_csv.parent / o_row.plot_filename
            if not k_img_path.exists() or not o_img_path.exists():
                continue

            k_img = cv2.imread(str(k_img_path), cv2.IMREAD_COLOR)
            o_img = cv2.imread(str(o_img_path), cv2.IMREAD_COLOR)
            if k_img is None or o_img is None:
                continue

            delta_ms = delta_ns / 1_000_000.0
            canvas = build_canvas(k_img, o_img, k_row, o_row, delta_ms)
            out_name = (
                f"kinect_orbbec_synced_"
                f"{k_row.frame_idx_master}_{k_row.save_timestamp_ns_master}__"
                f"{o_row.frame_idx_master}_{o_row.save_timestamp_ns_master}.png"
            )
            cv2.imwrite(str(output_dir / out_name), canvas)

            writer.writerow(
                [
                    k_row.save_timestamp_ns_master,
                    o_row.save_timestamp_ns_master,
                    k_row.frame_idx_master,
                    o_row.frame_idx_master,
                    f"{delta_ms:.6f}",
                    k_row.plot_filename,
                    o_row.plot_filename,
                    out_name,
                ]
            )

            saved += 1
            if args.max_pairs is not None and saved >= args.max_pairs:
                break

    print(f"Kinect companion rows used: {len(kinect_rows)}")
    print(f"Orbbec companion rows used: {len(orbbec_rows)}")
    print(f"Saved side-by-side composites: {saved}")
    print(f"Saved mapping CSV: {mapping_csv_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
