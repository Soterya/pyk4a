import argparse
import csv
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyk4a import CalibrationType, PyK4APlayback


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export synced Kinect data by device timestamp: full-resolution RGB JPGs for "
            "master/subordinates and full-resolution master depth NPZ."
        )
    )
    parser.add_argument("session_dir", type=Path, help="Path to session folder with Kinect MKVs + sidecars.")
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth reference frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on exported frames.")
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=20.0,
        help="Maximum nearest-neighbor delta for device timestamp matching.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality [0-100]. Resolution is always kept full-size.",
    )
    return parser.parse_args()


@dataclass
class FeedInfo:
    name: str
    media_path: Path
    sidecar_path: Path
    device_timestamps_usec: list[int]
    save_timestamps_ns: list[int]
    frame_indices: list[int]


def load_sidecar_csv(path: Path):
    device_timestamps_usec = []
    save_timestamps_ns = []
    frame_indices = []

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            color_present = int(row.get("color_present", "1"))
            color_ts = row.get("color_timestamp_usec", "")
            save_ts = row.get("save_timestamp_ns", "")

            if color_present != 1 or not color_ts or not save_ts:
                continue

            device_timestamps_usec.append(int(color_ts))
            save_timestamps_ns.append(int(save_ts))
            frame_indices.append(int(row["frame_idx"]))

    return device_timestamps_usec, save_timestamps_ns, frame_indices


def discover_feeds(session_dir: Path):
    candidate_roots = []
    if session_dir.name == "rgb_depth_data":
        candidate_roots.append(session_dir)
    rgb_depth_dir = session_dir / "rgb_depth_data"
    if rgb_depth_dir.exists():
        candidate_roots.append(rgb_depth_dir)
    candidate_roots.append(session_dir)

    # Preserve order while deduplicating.
    seen = set()
    search_roots = []
    for root in candidate_roots:
        resolved = root.resolve()
        if resolved not in seen:
            search_roots.append(root)
            seen.add(resolved)

    feeds = []
    seen_media = set()
    for root in search_roots:
        for media_path in sorted(root.glob("kinect*.mkv")):
            resolved_media = media_path.resolve()
            if resolved_media in seen_media:
                continue
            seen_media.add(resolved_media)

            sidecar_path = media_path.with_suffix(".save_timestamps.csv")
            if not sidecar_path.exists():
                continue

            dev_ts, save_ts, frame_idx = load_sidecar_csv(sidecar_path)
            if not dev_ts:
                continue

            feeds.append(
                FeedInfo(
                    name=media_path.stem,
                    media_path=media_path,
                    sidecar_path=sidecar_path,
                    device_timestamps_usec=dev_ts,
                    save_timestamps_ns=save_ts,
                    frame_indices=frame_idx,
                )
            )
    return feeds


def pick_reference_feed(feeds: list[FeedInfo]):
    for feed in feeds:
        if feed.name == "kinect_master":
            return feed
    return feeds[0]


def find_nearest_index(target_usec: int, timestamps_usec: list[int]):
    pos = bisect_left(timestamps_usec, target_usec)
    best_idx = None
    best_delta = None
    for idx in (pos - 1, pos):
        if idx < 0 or idx >= len(timestamps_usec):
            continue
        delta = abs(timestamps_usec[idx] - target_usec)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx, best_delta


def build_matches(
    feeds: list[FeedInfo],
    reference_feed: FeedInfo,
    every_n: int,
    max_frames: int | None,
    max_delta_ms: float | None,
):
    max_delta_usec = None if max_delta_ms is None else int(max_delta_ms * 1000.0)
    matches = []

    ref_entries = list(
        zip(
            reference_feed.frame_indices,
            reference_feed.device_timestamps_usec,
            reference_feed.save_timestamps_ns,
        )
    )

    for list_index, (_, ref_dev_ts_usec, _) in enumerate(ref_entries):
        if list_index % every_n != 0:
            continue

        match = {
            "reference_device_timestamp_usec": ref_dev_ts_usec,
            "frames": {},
        }
        valid = True

        for feed in feeds:
            nearest_idx, delta_usec = find_nearest_index(ref_dev_ts_usec, feed.device_timestamps_usec)
            if nearest_idx is None:
                valid = False
                break
            if max_delta_usec is not None and delta_usec is not None and delta_usec > max_delta_usec:
                valid = False
                break

            match["frames"][feed.name] = {
                "frame_idx": feed.frame_indices[nearest_idx],
                "device_timestamp_usec": feed.device_timestamps_usec[nearest_idx],
                "save_timestamp_ns": feed.save_timestamps_ns[nearest_idx],
                "delta_usec": delta_usec or 0,
            }

        if valid:
            matches.append(match)
            if max_frames is not None and len(matches) >= max_frames:
                break

    return matches


def convert_kinect_to_bgr(color_format, color_image: np.ndarray):
    if color_format.name == "COLOR_MJPG":
        return cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    if color_format.name == "COLOR_NV12":
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_NV12)
    if color_format.name == "COLOR_YUY2":
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)
    if color_format.name == "COLOR_BGRA32":
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    return color_image


class KinectCaptureLoader:
    def __init__(self, media_path: Path):
        self.playback = PyK4APlayback(media_path)
        self.playback.open()
        self.current_idx = -1
        self.current_color_bgr = None
        self.current_depth = None

    def get_capture(self, target_idx: int):
        if target_idx < self.current_idx:
            raise RuntimeError(
                f"Requested frame {target_idx} after already advancing to {self.current_idx}. "
                "Matches must be non-decreasing by frame index per stream."
            )

        while self.current_idx < target_idx:
            capture = self.playback.get_next_capture()
            self.current_idx += 1

            if capture.color is not None:
                self.current_color_bgr = convert_kinect_to_bgr(
                    self.playback.configuration["color_format"], capture.color
                )
            else:
                self.current_color_bgr = None

            self.current_depth = capture.depth

        return self.current_color_bgr, self.current_depth

    def close(self):
        self.playback.close()


def get_key_for_sub(frames_by_name: dict[str, FeedInfo], index: int):
    target = f"kinect_subordinate{index}"
    if target in frames_by_name:
        return target
    target = f"kinect_subordinate_{index}"
    if target in frames_by_name:
        return target
    for key in frames_by_name:
        if f"subordinate{index}" in key or f"subordinate_{index}" in key:
            return key
    raise KeyError(f"Could not find subordinate stream for index {index}")


def ensure_uint16_depth(depth: np.ndarray):
    if depth is None:
        return None
    if depth.dtype == np.uint16:
        return depth
    return depth.astype(np.uint16, copy=False)


def extract_intrinsics_json(playback: PyK4APlayback) -> dict:
    def find_intrinsics_nodes(node, path, out):
        if isinstance(node, dict):
            for k, v in node.items():
                next_path = path + [str(k)]
                if str(k).lower() == "intrinsics" and isinstance(v, dict):
                    out.append((next_path, v))
                find_intrinsics_nodes(v, next_path, out)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                find_intrinsics_nodes(v, path + [str(i)], out)

    def extract_from_raw(camera_label: str) -> dict:
        raw_text = playback.calibration_raw
        try:
            raw_obj = json.loads(raw_text)
        except Exception:
            return {
                "format": "raw_text",
                "camera_label": camera_label,
                "calibration_raw": raw_text,
            }

        nodes = []
        find_intrinsics_nodes(raw_obj, [], nodes)
        if not nodes:
            return {
                "format": "raw_json",
                "camera_label": camera_label,
                "calibration_raw_json": raw_obj,
            }

        # Prefer path matches to color/depth; otherwise use first intrinsic node.
        camera_label_lower = camera_label.lower()
        for path, intr in nodes:
            path_joined = "/".join(path).lower()
            if camera_label_lower in path_joined:
                return {
                    "format": "raw_intrinsics",
                    "camera_label": camera_label,
                    "path": path,
                    "intrinsics": intr,
                }

        path, intr = nodes[0]
        return {
            "format": "raw_intrinsics",
            "camera_label": camera_label,
            "path": path,
            "intrinsics": intr,
        }

    calibration = playback.calibration
    out = {}
    for label, camera in (("color", CalibrationType.COLOR), ("depth", CalibrationType.DEPTH)):
        try:
            camera_matrix = calibration.get_camera_matrix(camera).tolist()
            distortion = calibration.get_distortion_coefficients(camera).tolist()
            out[label] = {
                "format": "opencv",
                "camera_matrix": camera_matrix,
                "distortion_coefficients": distortion,
            }
        except Exception as e:
            fallback = extract_from_raw(label)
            fallback["opencv_error"] = str(e)
            out[label] = fallback
    return out


def get_session_name_from_data_path(session_dir: Path) -> str:
    resolved = session_dir.resolve()
    parts = resolved.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data" and i + 1 < len(parts):
            return parts[i + 1]
    return resolved.name


def main():
    args = parse_args()
    session_dir = args.session_dir.resolve()

    if args.every_n <= 0:
        raise ValueError("--every-n must be >= 1")
    if not (0 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality must be in [0, 100]")

    feeds = discover_feeds(session_dir)
    if len(feeds) != 5:
        raise RuntimeError(f"Expected 5 Kinect feeds in {session_dir}, found {len(feeds)}")

    ref_feed = pick_reference_feed(feeds)
    matches = build_matches(
        feeds=feeds,
        reference_feed=ref_feed,
        every_n=args.every_n,
        max_frames=args.max_frames,
        max_delta_ms=args.max_delta_ms,
    )
    if not matches:
        raise RuntimeError("No synced matches found. Try increasing --max-delta-ms.")

    repo_root = Path(__file__).resolve().parent.parent
    session_name = get_session_name_from_data_path(session_dir)
    output_dir = repo_root / "outputs" / session_name / "kinect"
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_master_dir = output_dir / "kinect_rgb_master"
    rgb_sub1_dir = output_dir / "kinect_rgb_sub1"
    rgb_sub2_dir = output_dir / "kinect_rgb_sub2"
    rgb_sub3_dir = output_dir / "kinect_rgb_sub3"
    rgb_sub4_dir = output_dir / "kinect_rgb_sub4"
    depth_master_dir = output_dir / "kinect_depth_master"

    for p in (
        rgb_master_dir,
        rgb_sub1_dir,
        rgb_sub2_dir,
        rgb_sub3_dir,
        rgb_sub4_dir,
        depth_master_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    companion_csv_path = output_dir / "kinect_synced_data_companion.csv"

    loaders = {feed.name: KinectCaptureLoader(feed.media_path) for feed in feeds}
    frames_by_name = {feed.name: feed for feed in feeds}

    sub1_key = get_key_for_sub(frames_by_name, 1)
    sub2_key = get_key_for_sub(frames_by_name, 2)
    sub3_key = get_key_for_sub(frames_by_name, 3)
    sub4_key = get_key_for_sub(frames_by_name, 4)

    intrinsics_by_feed = {
        name: extract_intrinsics_json(loader.playback) for name, loader in loaders.items()
    }
    master_intr = intrinsics_by_feed["kinect_master"]
    sub1_intr = intrinsics_by_feed[sub1_key]
    sub2_intr = intrinsics_by_feed[sub2_key]
    sub3_intr = intrinsics_by_feed[sub3_key]
    sub4_intr = intrinsics_by_feed[sub4_key]

    calib_color_master_json = json.dumps(master_intr["color"], separators=(",", ":"))
    calib_color_sub1_json = json.dumps(sub1_intr["color"], separators=(",", ":"))
    calib_color_sub2_json = json.dumps(sub2_intr["color"], separators=(",", ":"))
    calib_color_sub3_json = json.dumps(sub3_intr["color"], separators=(",", ":"))
    calib_color_sub4_json = json.dumps(sub4_intr["color"], separators=(",", ":"))
    calib_depth_master_json = json.dumps(master_intr["depth"], separators=(",", ":"))
    calib_depth_sub1_json = json.dumps(sub1_intr["depth"], separators=(",", ":"))
    calib_depth_sub2_json = json.dumps(sub2_intr["depth"], separators=(",", ":"))
    calib_depth_sub3_json = json.dumps(sub3_intr["depth"], separators=(",", ":"))
    calib_depth_sub4_json = json.dumps(sub4_intr["depth"], separators=(",", ":"))

    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)]

    try:
        with companion_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "save_timestamp_ns_master",
                    "frame_idx_master",
                    "frame_idx_sub1",
                    "frame_idx_sub2",
                    "frame_idx_sub3",
                    "frame_idx_sub4",
                    "rgb_path_master",
                    "rgb_path_sub1",
                    "rgb_path_sub2",
                    "rgb_path_sub3",
                    "rgb_path_sub4",
                    "depth_path_master",
                    "calib_color_master_json",
                    "calib_color_sub1_json",
                    "calib_color_sub2_json",
                    "calib_color_sub3_json",
                    "calib_color_sub4_json",
                    "calib_depth_master_json",
                    "calib_depth_sub1_json",
                    "calib_depth_sub2_json",
                    "calib_depth_sub3_json",
                    "calib_depth_sub4_json",
                ]
            )

            for i, match in enumerate(matches):
                master_info = match["frames"]["kinect_master"]
                sub1_info = match["frames"][sub1_key]
                sub2_info = match["frames"][sub2_key]
                sub3_info = match["frames"][sub3_key]
                sub4_info = match["frames"][sub4_key]

                master_rgb, master_depth = loaders["kinect_master"].get_capture(master_info["frame_idx"])
                sub1_rgb, _ = loaders[sub1_key].get_capture(sub1_info["frame_idx"])
                sub2_rgb, _ = loaders[sub2_key].get_capture(sub2_info["frame_idx"])
                sub3_rgb, _ = loaders[sub3_key].get_capture(sub3_info["frame_idx"])
                sub4_rgb, _ = loaders[sub4_key].get_capture(sub4_info["frame_idx"])

                if master_rgb is None or sub1_rgb is None or sub2_rgb is None or sub3_rgb is None or sub4_rgb is None:
                    raise RuntimeError(
                        f"Missing RGB image in synced match {i}; sidecar indicated color-present frames."
                    )

                depth_u16 = ensure_uint16_depth(master_depth)
                if depth_u16 is None:
                    raise RuntimeError(
                        f"Missing master depth image in synced match {i}; expected depth for master stream."
                    )

                basename = f"{i:06d}"

                rgb_master_rel = Path("kinect_rgb_master") / f"{basename}.jpg"
                rgb_sub1_rel = Path("kinect_rgb_sub1") / f"{basename}.jpg"
                rgb_sub2_rel = Path("kinect_rgb_sub2") / f"{basename}.jpg"
                rgb_sub3_rel = Path("kinect_rgb_sub3") / f"{basename}.jpg"
                rgb_sub4_rel = Path("kinect_rgb_sub4") / f"{basename}.jpg"
                depth_master_rel = Path("kinect_depth_master") / f"{basename}.npz"

                rgb_master_path = output_dir / rgb_master_rel
                rgb_sub1_path = output_dir / rgb_sub1_rel
                rgb_sub2_path = output_dir / rgb_sub2_rel
                rgb_sub3_path = output_dir / rgb_sub3_rel
                rgb_sub4_path = output_dir / rgb_sub4_rel
                depth_master_path = output_dir / depth_master_rel

                ok = cv2.imwrite(str(rgb_master_path), master_rgb, jpg_params)
                ok = ok and cv2.imwrite(str(rgb_sub1_path), sub1_rgb, jpg_params)
                ok = ok and cv2.imwrite(str(rgb_sub2_path), sub2_rgb, jpg_params)
                ok = ok and cv2.imwrite(str(rgb_sub3_path), sub3_rgb, jpg_params)
                ok = ok and cv2.imwrite(str(rgb_sub4_path), sub4_rgb, jpg_params)
                if not ok:
                    raise RuntimeError(f"Failed writing one or more RGB JPG files for synced index {i}")

                np.savez_compressed(depth_master_path, depth=depth_u16)

                writer.writerow(
                    [
                        master_info["save_timestamp_ns"],
                        master_info["frame_idx"],
                        sub1_info["frame_idx"],
                        sub2_info["frame_idx"],
                        sub3_info["frame_idx"],
                        sub4_info["frame_idx"],
                        str(rgb_master_rel),
                        str(rgb_sub1_rel),
                        str(rgb_sub2_rel),
                        str(rgb_sub3_rel),
                        str(rgb_sub4_rel),
                        str(depth_master_rel),
                        calib_color_master_json,
                        calib_color_sub1_json,
                        calib_color_sub2_json,
                        calib_color_sub3_json,
                        calib_color_sub4_json,
                        calib_depth_master_json,
                        calib_depth_sub1_json,
                        calib_depth_sub2_json,
                        calib_depth_sub3_json,
                        calib_depth_sub4_json,
                    ]
                )

                if (i + 1) % 100 == 0:
                    print(f"Exported {i + 1}/{len(matches)} synced samples...")
    finally:
        for loader in loaders.values():
            loader.close()

    print(f"Saved {len(matches)} synced samples to: {output_dir}")
    print(f"Saved companion CSV: {companion_csv_path}")


if __name__ == "__main__":
    main()
