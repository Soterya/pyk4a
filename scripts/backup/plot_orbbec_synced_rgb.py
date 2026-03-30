import argparse
import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import Config, OBSensorType, Pipeline, PlaybackDevice


CELL_WIDTH = 800
CELL_HEIGHT = 450
GRID_COLS = 2
GRID_ROWS = 1
OUTPUT_FPS = 10.0


@dataclass
class FeedInfo:
    name: str
    media_path: Path
    sidecar_path: Path
    timestamps_ns: list[int]
    frame_indices: list[int]


def frame_to_bgr(color_frame):
    if color_frame is None:
        return None
    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    format_name = str(color_format)
    if "RGB" in format_name and "BGR" not in format_name:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if "BGR" in format_name:
        return np.resize(data, (height, width, 3))
    if "MJPG" in format_name:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if "YUYV" in format_name or "YUY2" in format_name:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build synchronized 2-camera Orbbec RGB composite plots from one session folder."
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to recordings/session_<timestamp> containing Orbbec bags and .save_timestamps.csv sidecars.",
    )
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth reference frame. Default: 1")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on exported synchronized frames.")
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=None,
        help="Optional maximum nearest-neighbor match delta in milliseconds.",
    )
    parser.add_argument("--display", action="store_true", help="Display the composite while exporting.")
    return parser.parse_args()


def load_sidecar_csv(path: Path):
    timestamps_ns = []
    frame_indices = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row["color_timestamp_usec"]:
                continue
            timestamps_ns.append(int(row["save_timestamp_ns"]))
            frame_indices.append(int(row["frame_idx"]))
    return timestamps_ns, frame_indices


def discover_feeds(session_dir: Path):
    feeds = []
    for media_path in sorted(session_dir.glob("orbbec*.bag")):
        sidecar_path = media_path.with_suffix(".save_timestamps.csv")
        if not sidecar_path.exists():
            continue
        timestamps_ns, frame_indices = load_sidecar_csv(sidecar_path)
        if not timestamps_ns:
            continue
        feeds.append(
            FeedInfo(
                name=media_path.stem,
                media_path=media_path,
                sidecar_path=sidecar_path,
                timestamps_ns=timestamps_ns,
                frame_indices=frame_indices,
            )
        )
    return feeds


def pick_reference_feed(feeds: list[FeedInfo]):
    for feed in feeds:
        if feed.name == "orbbec_master":
            return feed
    return feeds[0]


def find_nearest_index(target_ns: int, timestamps_ns: list[int]):
    pos = bisect_left(timestamps_ns, target_ns)
    best_idx = None
    best_delta = None
    for idx in (pos - 1, pos):
        if idx < 0 or idx >= len(timestamps_ns):
            continue
        delta = abs(timestamps_ns[idx] - target_ns)
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
    max_delta_ns = None if max_delta_ms is None else int(max_delta_ms * 1_000_000)
    matches = []
    ref_entries = list(zip(reference_feed.frame_indices, reference_feed.timestamps_ns))

    for list_index, (_, ref_timestamp_ns) in enumerate(ref_entries):
        if list_index % every_n != 0:
            continue

        match = {"reference_timestamp_ns": ref_timestamp_ns, "frames": {}}
        valid = True
        for feed in feeds:
            nearest_idx, delta_ns = find_nearest_index(ref_timestamp_ns, feed.timestamps_ns)
            if nearest_idx is None:
                valid = False
                break
            if max_delta_ns is not None and delta_ns is not None and delta_ns > max_delta_ns:
                valid = False
                break
            match["frames"][feed.name] = {
                "frame_idx": feed.frame_indices[nearest_idx],
                "save_timestamp_ns": feed.timestamps_ns[nearest_idx],
                "delta_ns": delta_ns or 0,
            }

        if valid:
            matches.append(match)
            if max_frames is not None and len(matches) >= max_frames:
                break

    return matches


class OrbbecLoader:
    def __init__(self, media_path: Path):
        self.playback = PlaybackDevice(str(media_path))
        self.pipeline = Pipeline(self.playback)
        config = Config()
        config.enable_stream(OBSensorType.COLOR_SENSOR)
        self.pipeline.start(config)
        self.current_idx = -1
        self.current_image = None

    def get_frame(self, target_idx: int):
        while self.current_idx < target_idx:
            frames = self.pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            self.current_idx += 1
            self.current_image = frame_to_bgr(frames.get_color_frame())
        return self.current_image

    def close(self):
        self.pipeline.stop()
        self.playback = None


def make_panel(image, title: str, save_timestamp_ns: int, delta_ns: int, frame_idx: int):
    panel = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    if image is not None:
        resized = cv2.resize(image, (CELL_WIDTH, CELL_HEIGHT), interpolation=cv2.INTER_AREA)
        panel[:] = resized

    cv2.rectangle(panel, (0, 0), (CELL_WIDTH, 78), (20, 20, 20), -1)
    cv2.putText(
        panel,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame={frame_idx} delta_ms={delta_ns / 1_000_000.0:.3f}",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"save_timestamp_ns={save_timestamp_ns}",
        (10, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return panel


def build_canvas(feed_order: list[FeedInfo], loaders: dict[str, OrbbecLoader], match, frame_number: int, total_frames: int):
    canvas = np.zeros((GRID_ROWS * CELL_HEIGHT, GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)
    for idx, feed in enumerate(feed_order):
        col = idx % GRID_COLS
        frame_info = match["frames"][feed.name]
        image = loaders[feed.name].get_frame(frame_info["frame_idx"])
        panel = make_panel(
            image,
            feed.name,
            frame_info["save_timestamp_ns"],
            frame_info["delta_ns"],
            frame_info["frame_idx"],
        )
        x0 = col * CELL_WIDTH
        canvas[0:CELL_HEIGHT, x0:x0 + CELL_WIDTH] = panel

    footer_text = (
        f"Orbbec RGB sync {frame_number + 1}/{total_frames} | "
        f"reference save_timestamp_ns={match['reference_timestamp_ns']}"
    )
    cv2.rectangle(
        canvas,
        (0, canvas.shape[0] - 40),
        (canvas.shape[1], canvas.shape[0]),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        canvas,
        footer_text,
        (12, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    session_dir = args.session_dir.resolve()
    feeds = discover_feeds(session_dir)
    if len(feeds) != 2:
        raise RuntimeError(f"Expected 2 Orbbec feeds in {session_dir}, found {len(feeds)}")

    reference_feed = pick_reference_feed(feeds)
    matches = build_matches(
        feeds,
        reference_feed,
        args.every_n,
        args.max_frames,
        args.max_delta_ms,
    )
    if not matches:
        raise RuntimeError("No synchronized nearest-neighbor matches could be built")

    output_dir = session_dir / "orbbec_rgb_sync_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {feed.name: OrbbecLoader(feed.media_path) for feed in feeds}
    feed_order = sorted(
        feeds,
        key=lambda feed: (0 if feed.name == "orbbec_master" else 1, feed.name),
    )

    try:
        for idx, match in enumerate(matches):
            canvas = build_canvas(feed_order, loaders, match, idx, len(matches))
            cv2.imwrite(str(output_dir / f"orbbec_rgb_sync_{idx:06d}.png"), canvas)
            if args.display:
                cv2.imshow("2-Orbbec RGB Sync", canvas)
                if cv2.waitKey(max(1, int(1000 / OUTPUT_FPS))) & 0xFF in (27, ord("q")):
                    break
    finally:
        for loader in loaders.values():
            loader.close()
        cv2.destroyAllWindows()

    print(f"Saved Orbbec RGB sync plots to {output_dir}")


if __name__ == "__main__":
    main()
