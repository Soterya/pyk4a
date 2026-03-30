import argparse
import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyk4a import PyK4APlayback


CELL_WIDTH = 560
CELL_HEIGHT = 320
GRID_COLS = 3
GRID_ROWS = 2
OUTPUT_FPS = 10.0


@dataclass
class FeedInfo:
    name: str
    media_path: Path
    sidecar_path: Path
    device_timestamps_usec: list[int]
    save_timestamps_ns: list[int]
    frame_indices: list[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 5 Kinect RGB feeds synced by device timestamps (color_timestamp_usec)."
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
    parser.add_argument("--display", action="store_true", help="Show preview window while exporting.")
    return parser.parse_args()


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
    feeds = []
    for media_path in sorted(session_dir.glob("kinect*.mkv")):
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


def fit_to_panel_keep_aspect(image: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    if image is None:
        return canvas

    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return canvas

    scale = min(panel_w / src_w, panel_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


class KinectLoader:
    def __init__(self, media_path: Path):
        self.playback = PyK4APlayback(media_path)
        self.playback.open()
        self.current_idx = -1
        self.current_image = None

    def get_frame(self, target_idx: int):
        while self.current_idx < target_idx:
            capture = self.playback.get_next_capture()
            self.current_idx += 1
            if capture.color is not None:
                self.current_image = convert_kinect_to_bgr(
                    self.playback.configuration["color_format"], capture.color
                )
            else:
                self.current_image = None
        return self.current_image

    def close(self):
        self.playback.close()


def make_panel(image, title: str, frame_idx: int, dev_ts_usec: int, save_ts_ns: int, delta_usec: int):
    panel = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    if image is not None:
        panel[:] = fit_to_panel_keep_aspect(image, CELL_WIDTH, CELL_HEIGHT)

    # Use a smaller semi-transparent footer so the image is not cut off.
    overlay_h = 86
    y0 = CELL_HEIGHT - overlay_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, y0), (CELL_WIDTH, CELL_HEIGHT), (18, 18, 18), -1)
    panel = cv2.addWeighted(overlay, 0.60, panel, 0.40, 0)

    cv2.putText(panel, title, (10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame={frame_idx}  delta_ms={delta_usec / 1000.0:.3f}", (10, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"device_ts_usec={dev_ts_usec}", (10, y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"save_timestamp_ns={save_ts_ns}", (10, y0 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 255, 120), 2, cv2.LINE_AA)
    return panel


def build_canvas(feed_order: list[FeedInfo], loaders: dict[str, KinectLoader], match, i: int, total: int):
    canvas = np.zeros((GRID_ROWS * CELL_HEIGHT, GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)

    for idx, feed in enumerate(feed_order):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        info = match["frames"][feed.name]
        image = loaders[feed.name].get_frame(info["frame_idx"])
        panel = make_panel(
            image=image,
            title=feed.name,
            frame_idx=info["frame_idx"],
            dev_ts_usec=info["device_timestamp_usec"],
            save_ts_ns=info["save_timestamp_ns"],
            delta_usec=info["delta_usec"],
        )
        y0 = row * CELL_HEIGHT
        x0 = col * CELL_WIDTH
        canvas[y0:y0 + CELL_HEIGHT, x0:x0 + CELL_WIDTH] = panel

    max_delta_ms = max(v["delta_usec"] for v in match["frames"].values()) / 1000.0
    footer = (
        f"Kinect sync by DEVICE timestamp | {i + 1}/{total} | "
        f"ref_device_ts_usec={match['reference_device_timestamp_usec']} | "
        f"max_delta_ms={max_delta_ms:.3f}"
    )
    cv2.rectangle(canvas, (0, canvas.shape[0] - 40), (canvas.shape[1], canvas.shape[0]), (18, 18, 18), -1)
    cv2.putText(canvas, footer, (12, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    session_dir = args.session_dir.resolve()

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

    output_dir = session_dir / "kinect_rgb_sync_by_device_ts_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    companion_csv_path = output_dir / "kinect_synced_device_ts_companion.csv"

    feed_order = sorted(feeds, key=lambda f: (0 if f.name == "kinect_master" else 1, f.name))
    loaders = {feed.name: KinectLoader(feed.media_path) for feed in feeds}
    frames_by_name = {feed.name: feed for feed in feeds}

    def get_key_for_sub(index: int):
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
                    "plot_filename",
                ]
            )

            sub1_key = get_key_for_sub(1)
            sub2_key = get_key_for_sub(2)
            sub3_key = get_key_for_sub(3)
            sub4_key = get_key_for_sub(4)

            for i, match in enumerate(matches):
                canvas = build_canvas(feed_order, loaders, match, i, len(matches))
                master_info = match["frames"]["kinect_master"]
                sub1_info = match["frames"][sub1_key]
                sub2_info = match["frames"][sub2_key]
                sub3_info = match["frames"][sub3_key]
                sub4_info = match["frames"][sub4_key]

                master_save_ts = master_info["save_timestamp_ns"]
                plot_filename = f"kinect_synced_{master_info['frame_idx']}_{master_save_ts}.png"

                # Companion mapping row is written before saving each plot image.
                writer.writerow(
                    [
                        master_save_ts,
                        master_info["frame_idx"],
                        sub1_info["frame_idx"],
                        sub2_info["frame_idx"],
                        sub3_info["frame_idx"],
                        sub4_info["frame_idx"],
                        plot_filename,
                    ]
                )
                cv2.imwrite(str(output_dir / plot_filename), canvas)

                if args.display:
                    cv2.imshow("Kinect Sync by Device TS", canvas)
                    if cv2.waitKey(max(1, int(1000 / OUTPUT_FPS))) & 0xFF in (27, ord("q")):
                        break
    finally:
        for loader in loaders.values():
            loader.close()
        cv2.destroyAllWindows()

    print(f"Saved {len(matches)} plots to: {output_dir}")
    print(f"Saved companion CSV: {companion_csv_path}")


if __name__ == "__main__":
    main()
