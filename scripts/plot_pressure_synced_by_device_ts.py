#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


NODE_ROWS = 8
NODE_COLS = 4
NODE_H = 6
NODE_W = 6
SENSORS_PER_NODE = NODE_H * NODE_W
NODES_TOTAL = NODE_ROWS * NODE_COLS
TOTAL_VALUES = NODES_TOTAL * SENSORS_PER_NODE


@dataclass
class PressureRecord:
    frame_idx: int
    timestamp_ns: int
    values: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read pressure_map_save CSV frames and export pressure-map plots named "
            "pressure_maps_<frame_idx>_<timestamp_ns>.png with a companion CSV."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("data14/pressure_data"),
        help="CSV path or directory containing pressure_map_save_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved plots and companion CSV. Defaults near input CSV.",
    )
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on exported frames.")
    parser.add_argument("--cell-size", type=int, default=16, help="Scale factor per sensor cell.")
    parser.add_argument("--display", action="store_true", help="Show preview window while exporting.")
    return parser.parse_args()


def parse_pressure_data(raw: str) -> np.ndarray:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    values = np.fromstring(text, sep=",", dtype=np.float32)
    if values.size < TOTAL_VALUES:
        raise ValueError(
            f"pressure_data has {values.size} values, expected at least {TOTAL_VALUES}"
        )
    if values.size > TOTAL_VALUES:
        values = values[:TOTAL_VALUES]

    return np.nan_to_num(values, nan=0.0)


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


def colorize_grid(grid: np.ndarray, max_value: float, cell_size: int) -> np.ndarray:
    scale = max(max_value, 1.0)
    img8 = np.clip((grid / scale) * 255.0, 0, 255).astype(np.uint8)
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


def overlay_metadata(frame: np.ndarray, frame_idx: int, timestamp_ns: int, start_ns: int) -> np.ndarray:
    elapsed_s = (timestamp_ns - start_ns) / 1e9
    line1 = f"frame_idx={frame_idx}"
    line2 = f"save_timestamp_ns={timestamp_ns}"
    line3 = f"elapsed_s={elapsed_s:8.3f}"

    margin = 10
    line_h = 26
    box_h = 3 * line_h + 12
    cv2.rectangle(frame, (margin - 4, margin - 4), (frame.shape[1] - margin, margin + box_h), (0, 0, 0), -1)
    cv2.putText(frame, line1, (margin, margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (margin, margin + 20 + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, line3, (margin, margin + 20 + 2 * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 120), 2, cv2.LINE_AA)
    return frame


def resolve_pressure_csv(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidates = sorted(input_path.glob("pressure_map_save_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No pressure_map_save_*.csv found in: {input_path}")
    return candidates[-1]


def load_records(csv_path: Path) -> list[PressureRecord]:
    records: list[PressureRecord] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"timestamp_ns", "pressure_data"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain 'timestamp_ns' and 'pressure_data' columns")

        for frame_idx, row in enumerate(reader):
            ts_raw = (row.get("timestamp_ns") or "").strip()
            pd_raw = row.get("pressure_data") or ""
            if not ts_raw:
                continue

            timestamp_ns = int(ts_raw)
            values = parse_pressure_data(pd_raw)
            records.append(
                PressureRecord(
                    frame_idx=frame_idx,
                    timestamp_ns=timestamp_ns,
                    values=values,
                )
            )

    if not records:
        raise RuntimeError(f"No valid pressure records found in {csv_path}")
    return records


def main():
    args = parse_args()
    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")
    if args.cell_size < 1:
        raise ValueError("--cell-size must be >= 1")

    csv_path = resolve_pressure_csv(args.input_path.resolve())
    all_records = load_records(csv_path)
    selected_records = [r for i, r in enumerate(all_records) if i % args.every_n == 0]
    if args.max_frames is not None:
        selected_records = selected_records[: args.max_frames]
    if not selected_records:
        raise RuntimeError("No records selected after --every-n/--max-frames filtering.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else csv_path.parent / "pressure_maps_sync_by_device_ts_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    companion_csv_path = output_dir / "pressure_maps_synced_companion.csv"

    global_max = float(np.max(np.concatenate([r.values for r in selected_records])))
    if global_max <= 0:
        global_max = 1.0

    start_ns = selected_records[0].timestamp_ns
    with companion_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "save_timestamps_ns"])

        for out_idx, record in enumerate(selected_records):
            grid = build_pressure_grid(record.values)
            frame = colorize_grid(grid, global_max, args.cell_size)
            frame = overlay_metadata(frame, record.frame_idx, record.timestamp_ns, start_ns)

            filename = f"pressure_maps_{record.frame_idx}_{record.timestamp_ns}.png"
            writer.writerow([record.frame_idx, record.timestamp_ns])
            cv2.imwrite(str(output_dir / filename), frame)

            if args.display:
                cv2.imshow("Pressure Maps by Device TS", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if (out_idx + 1) % 100 == 0:
                print(f"Saved {out_idx + 1}/{len(selected_records)} pressure maps...")

    cv2.destroyAllWindows()
    print(f"Input CSV: {csv_path}")
    print(f"Saved {len(selected_records)} pressure-map images to: {output_dir}")
    print(f"Saved companion CSV: {companion_csv_path}")


if __name__ == "__main__":
    main()
