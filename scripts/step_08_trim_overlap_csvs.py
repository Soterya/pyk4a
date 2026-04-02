import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trim synced mapping CSVs to the valid overlap window where all relevant streams are active, "
            "and write session-named CSV copies."
        )
    )
    parser.add_argument(
        "--pressure-rgb-depth-csv",
        type=Path,
        required=True,
        help="Path to synced_data_from_pressure_kinect_orbbec.csv",
    )
    parser.add_argument(
        "--rgb-depth-csv",
        type=Path,
        required=True,
        help="Path to synced_data_from_kinect_orbbec.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write trimmed outputs. "
            "Default: inferred session directory under outputs/person_x/session_x."
        ),
    )
    return parser.parse_args()


def infer_person_session(path: Path):
    parts = path.resolve().parts
    for i in range(len(parts) - 1):
        if parts[i].startswith("person_") and i + 1 < len(parts) and parts[i + 1].startswith("session_"):
            return parts[i], parts[i + 1]
    return "person_x", "session_x"


def infer_session_output_dir(path_a: Path, path_b: Path) -> Path:
    for p in (path_a.resolve(), path_b.resolve()):
        parts = p.parts
        for i in range(len(parts) - 1):
            if parts[i] == "outputs" and i + 2 < len(parts):
                person = parts[i + 1]
                session = parts[i + 2]
                if person.startswith("person_") and session.startswith("session_"):
                    return Path(*parts[: i + 3])
    return path_a.resolve().parent


def _parse_int(value: str):
    try:
        return int(value)
    except Exception:
        return None


def detect_timestamp_columns(header: list[str], rows: list[list[str]]):
    ts_cols = [i for i, name in enumerate(header) if "save_timestamp_ns" in name]

    # Handle known malformed CSV shape where fused timestamp is written under kinect_frame_idx_master.
    if "fused_frame_idx_master" in header and not any("fused_save_timestamp_ns" in h for h in header):
        if "kinect_frame_idx_master" in header:
            idx = header.index("kinect_frame_idx_master")
            sample = []
            for row in rows[:200]:
                if idx < len(row):
                    v = _parse_int(row[idx])
                    if v is not None:
                        sample.append(v)
            if sample:
                large_ratio = sum(v > 10**15 for v in sample) / float(len(sample))
                if large_ratio >= 0.8 and idx not in ts_cols:
                    ts_cols.append(idx)
    return sorted(ts_cols)


def compute_overlap_window(rows: list[list[str]], ts_indices: list[int]):
    if not ts_indices:
        raise RuntimeError("[ERROR] No timestamp columns detected.")

    mins = []
    maxs = []
    for idx in ts_indices:
        vals = []
        for row in rows:
            if idx >= len(row):
                continue
            v = _parse_int(row[idx])
            if v is None or v <= 0:
                continue
            vals.append(v)
        if not vals:
            raise RuntimeError(f"[ERROR] No valid timestamp values found for timestamp column index {idx}.")
        mins.append(min(vals))
        maxs.append(max(vals))

    start_ns = max(mins)
    end_ns = min(maxs)
    if start_ns > end_ns:
        raise RuntimeError(
            f"[ERROR] No overlap found across timestamp columns. start_ns={start_ns}, end_ns={end_ns}"
        )
    return start_ns, end_ns


def filter_rows_to_overlap(rows: list[list[str]], ts_indices: list[int], start_ns: int, end_ns: int):
    kept = []
    dropped = 0
    for row in rows:
        valid = True
        for idx in ts_indices:
            if idx >= len(row):
                valid = False
                break
            v = _parse_int(row[idx])
            if v is None or v < start_ns or v > end_ns:
                valid = False
                break
        if valid:
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


def process_one_csv(input_csv: Path, output_csv: Path):
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    ts_indices = detect_timestamp_columns(header, rows)
    start_ns, end_ns = compute_overlap_window(rows, ts_indices)
    kept_rows, dropped = filter_rows_to_overlap(rows, ts_indices, start_ns, end_ns)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept_rows)

    return {
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "dropped_rows": dropped,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "ts_col_indices": ts_indices,
    }


def main():
    args = parse_args()
    pressure_csv = args.pressure_rgb_depth_csv.resolve()
    rgb_depth_csv = args.rgb_depth_csv.resolve()

    if not pressure_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {pressure_csv}")
    if not rgb_depth_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {rgb_depth_csv}")

    person, session = infer_person_session(pressure_csv)
    output_dir = args.output_dir.resolve() if args.output_dir else infer_session_output_dir(pressure_csv, rgb_depth_csv)

    out_pressure = output_dir / f"{person}_{session}_pressure_rgb_depth.csv"
    out_rgb_depth = output_dir / f"{person}_{session}_rgb_depth.csv"

    pressure_stats = process_one_csv(pressure_csv, out_pressure)
    rgb_depth_stats = process_one_csv(rgb_depth_csv, out_rgb_depth)

    print(f"[INFO] Wrote: {out_pressure}")
    print(
        f"[INFO] pressure_rgb_depth rows: {pressure_stats['input_rows']} -> {pressure_stats['kept_rows']} "
        f"(dropped {pressure_stats['dropped_rows']}) | window_ns=[{pressure_stats['start_ns']}, {pressure_stats['end_ns']}] "
        f"| ts_cols={pressure_stats['ts_col_indices']}"
    )
    print(f"[INFO] Wrote: {out_rgb_depth}")
    print(
        f"[INFO] rgb_depth rows: {rgb_depth_stats['input_rows']} -> {rgb_depth_stats['kept_rows']} "
        f"(dropped {rgb_depth_stats['dropped_rows']}) | window_ns=[{rgb_depth_stats['start_ns']}, {rgb_depth_stats['end_ns']}] "
        f"| ts_cols={rgb_depth_stats['ts_col_indices']}"
    )


if __name__ == "__main__":
    main()
