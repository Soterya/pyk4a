import argparse
import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline, PlaybackDevice


@dataclass
class BagFrameSet:
    frame_idx: int
    timestamp_us: int
    color_timestamp_us: int | None
    depth_timestamp_us: int | None
    color_image: np.ndarray | None
    depth_data_u16: np.ndarray | None
    save_timestamp_ns: int | None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export synchronized Orbbec data by device timestamp: full-resolution RGB JPG and "
            "full-resolution depth NPZ for master/subordinate, plus companion CSV."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Either <master.bag> <subordinate.bag> or one directory containing exactly two .bag files "
            "(supports data/person_x/session_x/data_collection and rgb_depth_data layouts)."
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
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality [0-100]. Resolution is always kept full-size.",
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


def find_bag_dir_from_input_dir(input_dir: Path) -> Path:
    if input_dir.name == "rgb_depth_data":
        return input_dir
    data_collection_rgb_depth_dir = input_dir / "data_collection" / "rgb_depth_data"
    if data_collection_rgb_depth_dir.exists():
        return data_collection_rgb_depth_dir
    data_collection_dir = input_dir / "data_collection"
    if data_collection_dir.exists():
        return data_collection_dir
    rgb_depth_data_dir = input_dir / "rgb_depth_data"
    if rgb_depth_data_dir.exists():
        return rgb_depth_data_dir
    return input_dir


def pick_bag_pair_from_dir(input_dir: Path):
    bag_dir = find_bag_dir_from_input_dir(input_dir)
    bag_paths = sorted(bag_dir.glob("orbbec*.bag"))
    if len(bag_paths) != 2:
        bag_paths = sorted(bag_dir.glob("*.bag"))
    if len(bag_paths) != 2:
        raise RuntimeError(f"Expected exactly 2 .bag files in {bag_dir}, found {len(bag_paths)}")

    master_bag = next((path for path in bag_paths if "master" in path.stem.lower()), None)
    subordinate_bag = next((path for path in bag_paths if "subordinate" in path.stem.lower()), None)

    if master_bag is not None and subordinate_bag is not None:
        return master_bag, subordinate_bag
    return bag_paths[0], bag_paths[1]


def resolve_input_bags(inputs: list[str]):
    if len(inputs) == 1:
        input_dir = Path(inputs[0])
        if not input_dir.is_dir():
            raise RuntimeError(f"Single input must be a directory containing two .bag files: {input_dir}")
        return pick_bag_pair_from_dir(input_dir)
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


def frame_to_depth_u16(depth_frame):
    if depth_frame is None:
        return None
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    return data.reshape((height, width))


def intrinsic_to_dict(intrinsic):
    if intrinsic is None:
        return None
    return {
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
    }


def distortion_to_dict(distortion):
    if distortion is None:
        return None
    return {
        "k1": float(distortion.k1),
        "k2": float(distortion.k2),
        "k3": float(distortion.k3),
        "k4": float(distortion.k4),
        "k5": float(distortion.k5),
        "k6": float(distortion.k6),
        "p1": float(distortion.p1),
        "p2": float(distortion.p2),
    }


def extract_camera_calibration_json(pipeline: Pipeline) -> dict:
    try:
        camera_param = pipeline.get_camera_param()
        return {
            "color": {
                "intrinsic": intrinsic_to_dict(camera_param.rgb_intrinsic),
                "distortion": distortion_to_dict(camera_param.rgb_distortion),
            },
            "depth": {
                "intrinsic": intrinsic_to_dict(camera_param.depth_intrinsic),
                "distortion": distortion_to_dict(camera_param.depth_distortion),
            },
            "format": "pipeline_camera_param",
        }
    except Exception as e:
        return {
            "format": "error",
            "error": str(e),
        }


def load_bag_frames(bag_path: Path, every_n: int):
    if not bag_path.exists():
        raise FileNotFoundError(f"Missing bag file: {bag_path}")

    sidecar_system_ts_us, sidecar_save_ts_ns = load_sidecar_save_map(bag_path)

    playback = PlaybackDevice(str(bag_path))
    pipeline = Pipeline(playback)
    config = Config()
    enable_available_streams(config, playback)
    pipeline.start(config)

    calibration_json = extract_camera_calibration_json(pipeline)

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
            depth_data_u16 = frame_to_depth_u16(depth_frame)

            results.append(
                BagFrameSet(
                    frame_idx=len(results),
                    timestamp_us=ts_us,
                    color_timestamp_us=color_system_ts_us,
                    depth_timestamp_us=depth_system_ts_us,
                    color_image=color_image,
                    depth_data_u16=depth_data_u16,
                    save_timestamp_ns=save_ts,
                )
            )
    finally:
        pipeline.stop()
        playback = None

    print(f"Loaded {len(results)} framesets from {bag_path.name}")
    return results, calibration_json


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


def resolve_output_dir(camera1_bag: Path) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    person_session_subpath = get_person_session_subpath_from_data_path(camera1_bag)
    return repo_root / "outputs" / person_session_subpath / "orbbec"


def save_matches(
    matches: list[tuple[BagFrameSet, BagFrameSet, int]],
    output_dir: Path,
    master_is_camera1: bool,
    camera1_calibration: dict,
    camera2_calibration: dict,
    jpg_quality: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_master_dir = output_dir / "orbbec_rgb_master"
    rgb_subordinate_dir = output_dir / "orbbec_rgb_subordinate"
    depth_master_dir = output_dir / "orbbec_depth_master"
    depth_subordinate_dir = output_dir / "orbbec_depth_subordinate"
    for p in (rgb_master_dir, rgb_subordinate_dir, depth_master_dir, depth_subordinate_dir):
        p.mkdir(parents=True, exist_ok=True)

    companion_csv_path = output_dir / "orbbec_synced_data_companion.csv"

    master_calib = camera1_calibration if master_is_camera1 else camera2_calibration
    subordinate_calib = camera2_calibration if master_is_camera1 else camera1_calibration
    master_color_calib_json = json.dumps(master_calib.get("color", {}), separators=(",", ":"))
    master_depth_calib_json = json.dumps(master_calib.get("depth", {}), separators=(",", ":"))
    subordinate_color_calib_json = json.dumps(subordinate_calib.get("color", {}), separators=(",", ":"))
    subordinate_depth_calib_json = json.dumps(subordinate_calib.get("depth", {}), separators=(",", ":"))

    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    skipped_missing_color = 0
    skipped_missing_depth = 0
    exported_count = 0

    with companion_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "save_timestamp_ns_master",
                "frame_idx_master",
                "frame_idx_subordinate",
                "rgb_path_master",
                "rgb_path_subordinate",
                "depth_path_master",
                "depth_path_subordinate",
                "calib_color_master_json",
                "calib_depth_master_json",
                "calib_color_subordinate_json",
                "calib_depth_subordinate_json",
            ]
        )

        for index, (frame1, frame2, _delta_us) in enumerate(matches):
            master_frame = frame1 if master_is_camera1 else frame2
            subordinate_frame = frame2 if master_is_camera1 else frame1

            if master_frame.color_image is None or subordinate_frame.color_image is None:
                skipped_missing_color += 1
                continue
            if master_frame.depth_data_u16 is None or subordinate_frame.depth_data_u16 is None:
                skipped_missing_depth += 1
                continue

            basename = f"{exported_count:06d}"
            rgb_master_rel = Path("orbbec_rgb_master") / f"{basename}.jpg"
            rgb_subordinate_rel = Path("orbbec_rgb_subordinate") / f"{basename}.jpg"
            depth_master_rel = Path("orbbec_depth_master") / f"{basename}.npz"
            depth_subordinate_rel = Path("orbbec_depth_subordinate") / f"{basename}.npz"

            rgb_master_path = output_dir / rgb_master_rel
            rgb_subordinate_path = output_dir / rgb_subordinate_rel
            depth_master_path = output_dir / depth_master_rel
            depth_subordinate_path = output_dir / depth_subordinate_rel

            ok = cv2.imwrite(str(rgb_master_path), master_frame.color_image, jpg_params)
            ok = ok and cv2.imwrite(str(rgb_subordinate_path), subordinate_frame.color_image, jpg_params)
            if not ok:
                raise RuntimeError(f"Failed writing one or more RGB JPG files for synced index {index}")

            np.savez_compressed(depth_master_path, depth=master_frame.depth_data_u16)
            np.savez_compressed(depth_subordinate_path, depth=subordinate_frame.depth_data_u16)

            master_save_ts = master_frame.save_timestamp_ns if master_frame.save_timestamp_ns is not None else 0
            writer.writerow(
                [
                    master_save_ts,
                    master_frame.frame_idx,
                    subordinate_frame.frame_idx,
                    str(rgb_master_rel),
                    str(rgb_subordinate_rel),
                    str(depth_master_rel),
                    str(depth_subordinate_rel),
                    master_color_calib_json,
                    master_depth_calib_json,
                    subordinate_color_calib_json,
                    subordinate_depth_calib_json,
                ]
            )
            exported_count += 1

            if (exported_count % 100) == 0:
                print(f"Exported {exported_count}/{len(matches)} synced pairs...")

    if skipped_missing_color or skipped_missing_depth:
        print(
            "Skipped pairs due to missing data: "
            f"missing_color={skipped_missing_color}, missing_depth={skipped_missing_depth}"
        )
    print(f"Saved {exported_count} synced pairs to {output_dir}")
    print(f"Saved companion CSV: {companion_csv_path}")


def main():
    args = parse_args()

    if args.every_n <= 0:
        raise ValueError("--every-n must be >= 1")
    if not (0 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality must be in [0, 100]")

    max_delta_us = int(args.max_delta_ms * 1000) if args.max_delta_ms is not None else None
    camera1_bag, camera2_bag = resolve_input_bags(args.inputs)

    print(f"Camera 1 bag: {camera1_bag}")
    print(f"Camera 2 bag: {camera2_bag}")

    camera1_frames, camera1_calibration = load_bag_frames(camera1_bag, args.every_n)
    camera2_frames, camera2_calibration = load_bag_frames(camera2_bag, args.every_n)
    matches = build_matches(camera1_frames, camera2_frames, max_delta_us, args.max_pairs)

    print(f"Built {len(matches)} synchronized pairs")
    cam1_is_master = "master" in camera1_bag.stem.lower()
    cam2_is_master = "master" in camera2_bag.stem.lower()
    master_is_camera1 = True
    if cam1_is_master and not cam2_is_master:
        master_is_camera1 = True
    elif cam2_is_master and not cam1_is_master:
        master_is_camera1 = False

    output_dir = resolve_output_dir(camera1_bag)
    save_matches(
        matches,
        output_dir,
        master_is_camera1,
        camera1_calibration,
        camera2_calibration,
        args.jpg_quality,
    )


if __name__ == "__main__":
    main()
