#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

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
            "Read pressure_map_save CSV frames and export full-resolution pressure data "
            "(.npz) with a companion CSV containing paths."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("data"),
        help=(
            "CSV path, pressure_data directory, or session directory containing pressure_data."
        ),
    )
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on exported frames.")
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


def get_session_name_from_data_path(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data" and i + 1 < len(parts):
            return parts[i + 1]
    return resolved.name


def resolve_pressure_csv(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Accept session directory layout: <session>/pressure_data
    if input_path.is_dir() and input_path.name != "pressure_data":
        pressure_data_dir = input_path / "pressure_data"
        if pressure_data_dir.exists() and pressure_data_dir.is_dir():
            input_path = pressure_data_dir

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")

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

    input_path = args.input_path.resolve()
    csv_path = resolve_pressure_csv(input_path)
    all_records = load_records(csv_path)

    selected_records = [r for i, r in enumerate(all_records) if i % args.every_n == 0]
    if args.max_frames is not None:
        selected_records = selected_records[: args.max_frames]
    if not selected_records:
        raise RuntimeError("No records selected after --every-n/--max-frames filtering.")

    repo_root = Path(__file__).resolve().parent.parent
    session_name = get_session_name_from_data_path(csv_path)
    output_dir = repo_root / "outputs" / session_name / "pressure"
    pressure_data_dir = output_dir / "pressure_map_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    pressure_data_dir.mkdir(parents=True, exist_ok=True)

    companion_csv_path = output_dir / "pressure_synced_data_companion.csv"

    with companion_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame_idx",
                "save_timestamp_ns",
                "pressure_data_path",
            ]
        )

        for out_idx, record in enumerate(selected_records):
            grid = build_pressure_grid(record.values)
            basename = f"{out_idx:06d}"
            rel_path = Path("pressure_map_data") / f"{basename}.npz"
            abs_path = output_dir / rel_path

            np.savez_compressed(
                abs_path,
                values=record.values.astype(np.float32),
                grid=grid.astype(np.float32),
                frame_idx=np.int64(record.frame_idx),
                timestamp_ns=np.int64(record.timestamp_ns),
            )

            writer.writerow(
                [
                    record.frame_idx,
                    record.timestamp_ns,
                    str(rel_path),
                ]
            )

            if (out_idx + 1) % 100 == 0:
                print(f"Saved {out_idx + 1}/{len(selected_records)} pressure data files...")

    print(f"Input CSV: {csv_path}")
    print(f"Saved {len(selected_records)} pressure data files to: {pressure_data_dir}")
    print(f"Saved companion CSV: {companion_csv_path}")


if __name__ == "__main__":
    main()
