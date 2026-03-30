import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from pyk4a import PyK4APlayback


def convert_to_bgr_if_required(color_format, color_image: np.ndarray) -> np.ndarray:
    if color_format.name == "COLOR_MJPG":
        return cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    if color_format.name == "COLOR_NV12":
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_NV12)
    if color_format.name == "COLOR_YUY2":
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)
    if color_format.name == "COLOR_BGRA32":
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    return color_image


def colorize_depth(depth_image: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_image, 0, 5000)
    normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def draw_overlay(image: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    output = image.copy()
    lines = [line for line in lines if line]
    if not lines:
        return output

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 12
    line_gap = 10

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(size[0] for size in line_sizes)
    total_height = sum(size[1] for size in line_sizes) + line_gap * (len(lines) - 1)

    top_left = (margin, margin)
    bottom_right = (margin + max_width + 24, margin + total_height + 24)
    cv2.rectangle(output, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.rectangle(output, top_left, bottom_right, (255, 255, 255), thickness=1)

    y = margin + 20
    for line, size in zip(lines, line_sizes):
        cv2.putText(output, line, (margin + 12, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += size[1] + line_gap
    return output


def load_sidecar_csv(path: Path) -> Tuple[Dict, List[Dict]]:
    metadata = {
        "name": path.stem.replace(".save_timestamps", ""),
    }
    frames = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frames.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "save_timestamp_ns": int(row["save_timestamp_ns"]),
                    "color_timestamp_usec": int(row["color_timestamp_usec"]) if row["color_timestamp_usec"] else None,
                    "depth_timestamp_usec": int(row["depth_timestamp_usec"]) if row["depth_timestamp_usec"] else None,
                    "depth_enabled": int(row["depth_enabled"]) if row["depth_enabled"] else 0,
                }
            )
    return metadata, frames


def find_record_sets(input_dir: Path) -> List[Tuple[Path, Path]]:
    record_sets = []
    for mkv_path in sorted(input_dir.glob("*.mkv")):
        sidecar_path = mkv_path.with_suffix(".save_timestamps.csv")
        if sidecar_path.exists():
            record_sets.append((mkv_path, sidecar_path))
    return record_sets


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def format_lines(
    record_name: str,
    sidecar_frame: Optional[Dict],
    capture,
    frame_idx: int,
) -> List[str]:
    lines = [
        f"{record_name} frame={frame_idx}",
        f"depth_device_ts_ns={capture.depth_timestamp_usec * 1000}",
        f"depth_system_ts_ns={capture.depth_system_timestamp_nsec}",
    ]

    if capture.color is not None:
        lines.extend(
            [
                f"color_device_ts_ns={capture.color_timestamp_usec * 1000}",
                f"color_system_ts_ns={capture.color_system_timestamp_nsec}",
            ]
        )

    if sidecar_frame is not None:
        lines.extend(
            [
                f"save_timestamp_ns={sidecar_frame.get('save_timestamp_ns', 'n/a')}",
                f"csv_color_timestamp_ns={sidecar_frame['color_timestamp_usec'] * 1000 if sidecar_frame.get('color_timestamp_usec') is not None else 'n/a'}",
                f"csv_depth_timestamp_ns={sidecar_frame['depth_timestamp_usec'] * 1000 if sidecar_frame.get('depth_timestamp_usec') is not None else 'n/a'}",
                f"depth_enabled={sidecar_frame.get('depth_enabled', 'n/a')}",
            ]
        )
    return lines


def build_plot_series(frames: List[Dict], key: str) -> Dict[int, int]:
    series = {}
    for frame in frames:
        value = frame.get(key)
        if value is None:
            continue
        series[int(frame["frame_idx"])] = int(value)
    return series


def plot_pairwise_differences(
    records: List[Dict],
    output_dir: Path,
    key: str,
    ylabel: str,
    title_prefix: str,
    filename_prefix: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate plots. Install it in the active environment.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    for left, right in combinations(records, 2):
        left_series = build_plot_series(left["timing_frames"], key)
        right_series = build_plot_series(right["timing_frames"], key)
        common_frames = sorted(set(left_series) & set(right_series))
        if not common_frames:
            continue

        diffs = [left_series[idx] - right_series[idx] for idx in common_frames]
        plt.figure(figsize=(12, 5))
        plt.plot(common_frames, diffs, linewidth=1.2)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.xlabel("Frame index")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {left['record_name']} - {right['record_name']}")
        plt.tight_layout()

        output_path = output_dir / (
            f"{filename_prefix}_{sanitize_name(left['record_name'])}_minus_{sanitize_name(right['record_name'])}.png"
        )
        plt.savefig(output_path, dpi=160)
        plt.close()


def export_record(
    mkv_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    every_n: int,
    max_frames: Optional[int],
) -> Dict:
    metadata, sidecar_frames = load_sidecar_csv(sidecar_path)
    record_name = metadata.get("name", mkv_path.stem)
    record_output_dir = output_dir / mkv_path.stem
    color_output_dir = record_output_dir / "color"
    depth_output_dir = record_output_dir / "depth"
    color_output_dir.mkdir(parents=True, exist_ok=True)
    depth_output_dir.mkdir(parents=True, exist_ok=True)

    playback = PyK4APlayback(mkv_path)
    playback.open()

    timing_frames: List[Dict] = []
    try:
        frame_idx = 0
        exported = 0
        while True:
            try:
                capture = playback.get_next_capture()
            except EOFError:
                break

            sidecar_frame = sidecar_frames[frame_idx] if frame_idx < len(sidecar_frames) else None
            timing_entry = {
                "frame_idx": frame_idx,
                "save_timestamp_ns": sidecar_frame["save_timestamp_ns"] if sidecar_frame is not None else None,
                "color_device_timestamp_ns": capture.color_timestamp_usec * 1000 if capture.color is not None else None,
                "depth_device_timestamp_ns": capture.depth_timestamp_usec * 1000,
                "depth_system_timestamp_ns": capture.depth_system_timestamp_nsec,
            }
            if capture.color is not None:
                timing_entry["color_system_timestamp_ns"] = capture.color_system_timestamp_nsec
            timing_frames.append(timing_entry)

            if frame_idx % every_n != 0:
                frame_idx += 1
                continue

            if max_frames is not None and exported >= max_frames:
                break

            lines = format_lines(record_name, sidecar_frame, capture, frame_idx)

            if capture.color is not None:
                color_bgr = convert_to_bgr_if_required(playback.configuration["color_format"], capture.color)
                color_overlay = draw_overlay(color_bgr, lines)
                cv2.imwrite(str(color_output_dir / f"frame_{frame_idx:06d}.png"), color_overlay)

            if capture.depth is not None:
                depth_bgr = colorize_depth(capture.depth)
                depth_overlay = draw_overlay(depth_bgr, lines)
                cv2.imwrite(str(depth_output_dir / f"frame_{frame_idx:06d}.png"), depth_overlay)

            exported += 1
            frame_idx += 1
    finally:
        playback.close()

    return {
        "record_name": record_name,
        "mkv_path": mkv_path,
        "sidecar_path": sidecar_path,
        "timing_frames": timing_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract color and depth images from MKV recordings, overlay timestamps from .save_timestamps.csv, and plot pairwise timestamp differences."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing MKV files and matching .save_timestamps.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where overlaid color/depth images and plots will be written.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Only export every Nth frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of exported frames per recording.",
    )
    args = parser.parse_args()

    if args.every_n <= 0:
        raise ValueError("--every-n must be >= 1")

    record_sets = find_record_sets(args.input_dir)
    if not record_sets:
        raise FileNotFoundError(f"No MKV + .save_timestamps.csv pairs found in {args.input_dir}")

    exported_records = []
    for mkv_path, sidecar_path in record_sets:
        print(f"Exporting overlays for {mkv_path.name}")
        exported_records.append(export_record(mkv_path, sidecar_path, args.output_dir, args.every_n, args.max_frames))

    plot_output_dir = args.output_dir / "plots"
    plot_pairwise_differences(
        exported_records,
        plot_output_dir,
        key="save_timestamp_ns",
        ylabel="Timestamp difference (ns)",
        title_prefix="Host save timestamp difference",
        filename_prefix="save_timestamp_diff_ns",
    )
    plot_pairwise_differences(
        exported_records,
        plot_output_dir,
        key="depth_device_timestamp_ns",
        ylabel="Timestamp difference (ns)",
        title_prefix="Depth device timestamp difference",
        filename_prefix="depth_device_timestamp_diff_ns",
    )
    plot_pairwise_differences(
        exported_records,
        plot_output_dir,
        key="depth_system_timestamp_ns",
        ylabel="Timestamp difference (ns)",
        title_prefix="Depth system timestamp difference",
        filename_prefix="depth_system_timestamp_diff_ns",
    )


if __name__ == "__main__":
    main()
