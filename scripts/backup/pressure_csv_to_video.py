"""
Run Using: 

python3 /home/rutwik/ros2_ws/src/pressure_csv_to_video.py   --csv /home/rutwik/ros2_ws/src/data/pressure_map_save_20260318_130633.csv   --output /home/rutwik/ros2_ws/src/data/pressure_map_reconstructed.mp4   --frames-dir /home/rutwik/ros2_ws/src/data/frames
"""

#!/usr/bin/env python3
import argparse
import csv
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

    # Draw node boundaries to make 6x6 blocks easy to see.
    for r in range(1, NODE_ROWS):
        y = r * NODE_H * cell_size
        cv2.line(frame, (0, y), (frame.shape[1], y), (255, 255, 255), 2)
    for c in range(1, NODE_COLS):
        x = c * NODE_W * cell_size
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 255), 2)

    return frame


def overlay_timestamp(frame: np.ndarray, timestamp_ns: int, start_ns: int) -> np.ndarray:
    elapsed_s = (timestamp_ns - start_ns) / 1e9
    label = f"ts_ns={timestamp_ns}   t={elapsed_s:8.3f}s"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    margin = 10

    # Keep text visible even on narrow frames.
    while True:
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        if text_w <= frame.shape[1] - 2 * margin or scale <= 0.35:
            break
        scale -= 0.05

    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = margin
    y = margin + text_h
    top_left = (x - 6, y - text_h - 6)
    bottom_right = (x + text_w + 6, y + baseline + 6)

    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def load_records(csv_path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamps = []
    values_list = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_ns", "pressure_data"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain 'timestamp_ns' and 'pressure_data' columns")

        for row_idx, row in enumerate(reader):
            ts_raw = row.get("timestamp_ns", "").strip()
            pd_raw = row.get("pressure_data", "")
            if not ts_raw:
                continue
            try:
                ts = int(ts_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp_ns at row {row_idx}: {ts_raw}") from exc

            values = parse_pressure_data(pd_raw)
            timestamps.append(ts)
            values_list.append(values)

    if len(timestamps) < 2:
        raise ValueError("Need at least 2 frames in CSV to build a video")

    return np.asarray(timestamps, dtype=np.int64), values_list


def estimate_fps(timestamps_ns: np.ndarray, mode: str) -> float:
    dt_ns = np.diff(timestamps_ns).astype(np.float64)
    dt_ns = dt_ns[dt_ns > 0]
    if dt_ns.size == 0:
        raise ValueError("No positive timestamp deltas; cannot estimate FPS")

    effective_fps = (len(timestamps_ns) - 1) / (
        (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    )
    median_fps = 1e9 / float(np.median(dt_ns))

    return float(effective_fps if mode == "effective" else median_fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct all pressure frames from CSV and encode an MP4 video."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("pressure_map_save_20260317_234535.csv"),
        help="Input CSV containing timestamp_ns and pressure_data columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pressure_map_reconstructed.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=16,
        help="Display scale for each sensor cell.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video FPS. If omitted, estimated from timestamps.",
    )
    parser.add_argument(
        "--fps-mode",
        choices=["effective", "median"],
        default="effective",
        help="FPS estimation mode when --fps is not provided.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional directory to also save each reconstructed frame as PNG.",
    )
    args = parser.parse_args()

    if args.cell_size < 1:
        raise ValueError("--cell-size must be >= 1")

    timestamps_ns, values_list = load_records(args.csv)
    fps = args.fps if args.fps is not None else estimate_fps(timestamps_ns, args.fps_mode)
    if fps <= 0:
        raise ValueError(f"Invalid FPS value: {fps}")

    global_max = float(np.max(np.concatenate(values_list)))
    if global_max <= 0:
        global_max = 1.0

    width = NODE_COLS * NODE_W * args.cell_size
    height = NODE_ROWS * NODE_H * args.cell_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {args.output}")

    if args.frames_dir is not None:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    start_ns = int(timestamps_ns[0])
    for i, values in enumerate(values_list):
        grid = build_pressure_grid(values)
        frame = colorize_grid(grid, global_max, args.cell_size)
        frame = overlay_timestamp(frame, int(timestamps_ns[i]), start_ns)

        if args.frames_dir is not None:
            frame_name = args.frames_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_name), frame)

        writer.write(frame)

    writer.release()

    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    print(f"Input CSV: {args.csv}")
    print(f"Frames: {len(values_list)}")
    print(f"Timestamp span (s): {duration_s:.6f}")
    print(f"Video FPS used: {fps:.6f}")
    print(f"Output video: {args.output}")
    if args.frames_dir is not None:
        print(f"Output frames dir: {args.frames_dir}")


if __name__ == "__main__":
    main()
