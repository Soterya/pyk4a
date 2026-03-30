import argparse
import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline, PlaybackDevice


WINDOW_NAME = "Synchronized Two-Camera Playback (Orbbec)"
DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 900


@dataclass
class BagFrameSet:
    frame_idx: int
    timestamp_us: int
    color_timestamp_us: int | None
    depth_timestamp_us: int | None
    color_image: np.ndarray | None
    depth_vis: np.ndarray | None
    save_timestamp_ns: int | None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load two Orbbec .bag recordings, align by nearest playback SYSTEM timestamp, "
            "and render/save a synchronized 2x2 view with save_timestamp_ns overlay."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Either <camera1.bag> <camera2.bag> or one directory containing exactly two .bag files."
        ),
    )
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth frameset from each bag.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on synchronized pairs.")
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=20.0,
        help="Maximum allowed nearest-neighbor match delta in milliseconds.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Playback speed for interactive display.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved 2x2 plots. Defaults near the first bag.",
    )
    parser.add_argument("--no-display", action="store_true", help="Skip OpenCV display and only save plots.")
    return parser.parse_args()


def pick_bag_pair_from_dir(session_dir: Path):
    bag_paths = sorted(session_dir.glob("*.bag"))
    if len(bag_paths) != 2:
        raise RuntimeError(f"Expected exactly 2 .bag files in {session_dir}, found {len(bag_paths)}")

    master_bag = next((path for path in bag_paths if "master" in path.stem.lower()), None)
    subordinate_bag = next((path for path in bag_paths if "subordinate" in path.stem.lower()), None)

    if master_bag is not None and subordinate_bag is not None:
        return master_bag, subordinate_bag
    return bag_paths[0], bag_paths[1]


def resolve_input_bags(inputs: list[str]):
    if len(inputs) == 1:
        session_dir = Path(inputs[0])
        if not session_dir.is_dir():
            raise RuntimeError(f"Single input must be a directory containing two .bag files: {session_dir}")
        return pick_bag_pair_from_dir(session_dir)
    if len(inputs) == 2:
        return Path(inputs[0]), Path(inputs[1])
    raise RuntimeError("Pass either one directory path or exactly two .bag file paths.")


def find_nearest_index(target_us: int, timestamps_us: list[int]):
    pos = bisect_left(timestamps_us, target_us)
    best_idx = None
    best_delta = None
    for idx in (pos - 1, pos):
        if idx < 0 or idx >= len(timestamps_us):
            continue
        delta = abs(timestamps_us[idx] - target_us)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx, best_delta


def load_sidecar_save_map(bag_path: Path):
    sidecar_path = bag_path.with_suffix(".save_timestamps.csv")
    if not sidecar_path.exists():
        return [], []

    system_ts_us = []
    save_ts_ns = []
    with sidecar_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row.get("color_present", "0")) != 1:
                continue
            color_system_ts = row.get("color_system_timestamp_usec", "")
            save_ts = row.get("save_timestamp_ns", "")
            if not color_system_ts or not save_ts:
                continue
            system_ts_us.append(int(color_system_ts))
            save_ts_ns.append(int(save_ts))
    return system_ts_us, save_ts_ns


def lookup_save_timestamp_ns(target_system_ts_us: int, sidecar_system_ts_us: list[int], sidecar_save_ts_ns: list[int]):
    if not sidecar_system_ts_us:
        return None
    idx, _ = find_nearest_index(target_system_ts_us, sidecar_system_ts_us)
    if idx is None:
        return None
    return sidecar_save_ts_ns[idx]


def build_depth_visualization(depth_frame):
    if depth_frame is None:
        return None
    if depth_frame.get_format() != OBFormat.Y16:
        return None

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
    depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


def enable_available_streams(config: Config, playback: PlaybackDevice):
    sensor_list = playback.get_sensor_list()
    available_types = {sensor_list.get_type_by_index(i) for i in range(sensor_list.get_count())}

    if OBSensorType.COLOR_SENSOR in available_types:
        config.enable_stream(OBSensorType.COLOR_SENSOR)
    if OBSensorType.DEPTH_SENSOR in available_types:
        config.enable_stream(OBSensorType.DEPTH_SENSOR)


def representative_timestamp_us(color_frame, depth_frame):
    timestamps = []
    if color_frame is not None:
        timestamps.append(color_frame.get_system_timestamp_us())
    if depth_frame is not None:
        timestamps.append(depth_frame.get_system_timestamp_us())
    if not timestamps:
        return None
    return int(sum(timestamps) / len(timestamps))


def frame_to_bgr_image(color_frame):
    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None


def load_bag_frames(bag_path: Path, every_n: int):
    if not bag_path.exists():
        raise FileNotFoundError(f"Missing bag file: {bag_path}")

    sidecar_system_ts_us, sidecar_save_ts_ns = load_sidecar_save_map(bag_path)

    playback = PlaybackDevice(str(bag_path))
    pipeline = Pipeline(playback)
    config = Config()
    enable_available_streams(config, playback)
    pipeline.start(config)

    results = []
    seen_frames = 0
    idle_loops = 0

    print(f"Loading {bag_path} ...")
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(100)
            except Exception:
                frames = None

            if frames is None:
                idle_loops += 1
                if idle_loops > 50:
                    break
                continue

            idle_loops = 0
            seen_frames += 1
            if seen_frames % every_n != 0:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None and depth_frame is None:
                continue

            ts_us = representative_timestamp_us(color_frame, depth_frame)
            if ts_us is None:
                continue

            color_system_ts_us = color_frame.get_system_timestamp_us() if color_frame is not None else None
            depth_system_ts_us = depth_frame.get_system_timestamp_us() if depth_frame is not None else None

            save_ts = None
            if color_system_ts_us is not None:
                save_ts = lookup_save_timestamp_ns(
                    color_system_ts_us, sidecar_system_ts_us, sidecar_save_ts_ns
                )

            color_image = frame_to_bgr_image(color_frame) if color_frame is not None else None
            depth_vis = build_depth_visualization(depth_frame)

            results.append(
                BagFrameSet(
                    frame_idx=len(results),
                    timestamp_us=ts_us,
                    color_timestamp_us=color_system_ts_us,
                    depth_timestamp_us=depth_system_ts_us,
                    color_image=color_image,
                    depth_vis=depth_vis,
                    save_timestamp_ns=save_ts,
                )
            )
    finally:
        pipeline.stop()
        playback = None

    print(f"Loaded {len(results)} framesets from {bag_path.name}")
    return results


def find_nearest_match(timestamp_us: int, candidates: list[BagFrameSet], candidate_timestamps: list[int]):
    insert_index = bisect_left(candidate_timestamps, timestamp_us)
    best_index = None
    best_delta = None

    for index in (insert_index - 1, insert_index):
        if index < 0 or index >= len(candidates):
            continue
        delta = abs(candidates[index].timestamp_us - timestamp_us)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_index = index

    return best_index, best_delta


def build_matches(
    camera1_frames: list[BagFrameSet],
    camera2_frames: list[BagFrameSet],
    max_delta_us: int | None,
    max_pairs: int | None,
):
    camera2_timestamps = [frame.timestamp_us for frame in camera2_frames]
    matches = []

    for frame1 in camera1_frames:
        match_index, delta_us = find_nearest_match(frame1.timestamp_us, camera2_frames, camera2_timestamps)
        if match_index is None:
            continue
        if max_delta_us is not None and delta_us is not None and delta_us > max_delta_us:
            continue

        matches.append((frame1, camera2_frames[match_index], delta_us or 0))
        if max_pairs is not None and len(matches) >= max_pairs:
            break

    return matches


def fit_to_panel_keep_aspect(image, width: int, height: int):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if image is None:
        return panel

    ih, iw = image.shape[:2]
    scale = min(width / iw, height / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)

    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = resized
    return panel


def prepare_panel(image, title: str, width: int, height: int, save_timestamp_ns: int | None):
    panel = fit_to_panel_keep_aspect(image, width, height)

    overlay_h = 68
    y0 = height - overlay_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, y0), (width, height), (20, 20, 20), -1)
    panel = cv2.addWeighted(overlay, 0.60, panel, 0.40, 0)

    save_ts_text = str(save_timestamp_ns) if save_timestamp_ns is not None else "N/A"
    cv2.putText(panel, title, (12, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"save_timestamp_ns={save_ts_text}", (12, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 120), 2, cv2.LINE_AA)
    return panel


def build_canvas(frame1: BagFrameSet, frame2: BagFrameSet, delta_us: int, pair_index: int, total_pairs: int):
    cell_w = DISPLAY_WIDTH // 2
    cell_h = DISPLAY_HEIGHT // 2
    canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

    panels = [
        prepare_panel(frame1.color_image, f"Camera 1 RGB | ts={frame1.color_timestamp_us}", cell_w, cell_h, frame1.save_timestamp_ns),
        prepare_panel(frame1.depth_vis, f"Camera 1 Depth | ts={frame1.depth_timestamp_us}", cell_w, cell_h, frame1.save_timestamp_ns),
        prepare_panel(frame2.color_image, f"Camera 2 RGB | ts={frame2.color_timestamp_us}", cell_w, cell_h, frame2.save_timestamp_ns),
        prepare_panel(frame2.depth_vis, f"Camera 2 Depth | ts={frame2.depth_timestamp_us}", cell_w, cell_h, frame2.save_timestamp_ns),
    ]

    canvas[0:cell_h, 0:cell_w] = panels[0]
    canvas[0:cell_h, cell_w:DISPLAY_WIDTH] = panels[1]
    canvas[cell_h:DISPLAY_HEIGHT, 0:cell_w] = panels[2]
    canvas[cell_h:DISPLAY_HEIGHT, cell_w:DISPLAY_WIDTH] = panels[3]

    footer = (
        f"Pair {pair_index + 1}/{total_pairs} | match_delta={delta_us / 1000.0:.3f} ms | "
        f"cam1_rep_ts={frame1.timestamp_us} cam2_rep_ts={frame2.timestamp_us}"
    )
    cv2.rectangle(canvas, (0, DISPLAY_HEIGHT - 36), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (20, 20, 20), -1)
    cv2.putText(canvas, footer, (12, DISPLAY_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resolve_output_dir(camera1_bag: Path, camera2_bag: Path, output_dir: str | None):
    if output_dir:
        return Path(output_dir)
    return camera1_bag.parent / f"{camera1_bag.stem}__{camera2_bag.stem}_sync_plots_with_save_ts"


def save_matches(
    matches: list[tuple[BagFrameSet, BagFrameSet, int]],
    output_dir: Path,
    master_is_camera1: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    companion_csv_path = output_dir / "orbbec_synced_companion.csv"
    with companion_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "save_timestamp_ns_master",
                "frame_idx_master",
                "frame_idx_subordinate",
                "plot_filename",
            ]
        )

        for index, (frame1, frame2, delta_us) in enumerate(matches):
            canvas = build_canvas(frame1, frame2, delta_us, index, len(matches))
            master_frame = frame1 if master_is_camera1 else frame2
            subordinate_frame = frame2 if master_is_camera1 else frame1
            master_save_ts = (
                master_frame.save_timestamp_ns
                if master_frame.save_timestamp_ns is not None
                else 0
            )
            filename = (
                f"orbbec_synced_{master_frame.frame_idx}_{master_save_ts}.png"
            )

            # Write companion row before saving the plot image.
            writer.writerow(
                [
                    master_save_ts,
                    master_frame.frame_idx,
                    subordinate_frame.frame_idx,
                    filename,
                ]
            )
            cv2.imwrite(str(output_dir / filename), canvas)
    print(f"Saved {len(matches)} plots to {output_dir}")
    print(f"Saved companion CSV: {companion_csv_path}")


def show_matches(matches: list[tuple[BagFrameSet, BagFrameSet, int]], fps: float):
    if not matches:
        print("No synchronized pairs found.")
        return

    delay_ms = max(1, int(1000 / fps))
    index = 0
    autoplay = True

    while 0 <= index < len(matches):
        frame1, frame2, delta_us = matches[index]
        canvas = build_canvas(frame1, frame2, delta_us, index, len(matches))
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(delay_ms if autoplay else 0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("p"), ord("b")):
            autoplay = False
            index = max(0, index - 1)
            continue
        if key in (ord(" "), ord("n")):
            autoplay = False
            index += 1
            continue
        if key == ord("a"):
            autoplay = not autoplay
            if not autoplay:
                continue

        if autoplay:
            index += 1

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    max_delta_us = int(args.max_delta_ms * 1000) if args.max_delta_ms is not None else None
    camera1_bag, camera2_bag = resolve_input_bags(args.inputs)

    print(f"Camera 1 bag: {camera1_bag}")
    print(f"Camera 2 bag: {camera2_bag}")

    camera1_frames = load_bag_frames(camera1_bag, args.every_n)
    camera2_frames = load_bag_frames(camera2_bag, args.every_n)
    matches = build_matches(camera1_frames, camera2_frames, max_delta_us, args.max_pairs)

    print(f"Built {len(matches)} synchronized pairs")
    output_dir = resolve_output_dir(camera1_bag, camera2_bag, args.output_dir)
    cam1_is_master = "master" in camera1_bag.stem.lower()
    cam2_is_master = "master" in camera2_bag.stem.lower()
    master_is_camera1 = True
    if cam1_is_master and not cam2_is_master:
        master_is_camera1 = True
    elif cam2_is_master and not cam1_is_master:
        master_is_camera1 = False
    save_matches(matches, output_dir, master_is_camera1)
    if not args.no_display:
        show_matches(matches, args.fps)


if __name__ == "__main__":
    main()
