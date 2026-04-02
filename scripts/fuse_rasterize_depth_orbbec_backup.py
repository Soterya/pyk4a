# ---------------
# 0) IMPORTS ----
# ---------------
import argparse
import json
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import Pipeline, PlaybackDevice, Config


# ----------------------
# 1) ARGPARSER SETUP ---
# ----------------------
parser = argparse.ArgumentParser(
    description="Depth-only fusion (cam1+cam2) from Orbbec .bag and rasterize into cam0 depth view using refined_extrinsics.json"
)

parser.add_argument("--calib_json", required=True, help="Path to refined_extrinsics.json (from hybrid calibration)")
parser.add_argument("--cam1_bag", required=True, help="Orbbec .bag for camera 1")
parser.add_argument("--cam2_bag", required=True, help="Orbbec .bag for camera 2")
parser.add_argument(
    "--out_dir",
    required=False,
    default=None,
    help=(
        "Base output directory. If omitted, defaults to "
        "./outputs/person_x/session_x/fused_depth_maps (derived from --calib_json)."
    ),
)

parser.add_argument("--start_idx", type=int, default=0, help="Frame index to start from (skip first N captures)")
parser.add_argument(
    "--max_frames",
    type=int,
    default=500,
    help="Maximum number of frames to process; use <=0 to process all available frames",
)
parser.add_argument("--wait_ms", type=int, default=2000, help="Timeout for waiting on each frame")
parser.add_argument("--read_retries", type=int, default=8, help="Retries per output frame to handle transient missing frames")
parser.add_argument("--eos_patience", type=int, default=40, help="Stop only after this many consecutive failed paired reads")
parser.add_argument(
    "--sync_tolerance_ms",
    type=float,
    default=5.0,
    help="Maximum timestamp delta (ms) allowed to pair cam1/cam2 frames for fusion",
)
parser.add_argument(
    "--sync_timestamp",
    type=str,
    default="auto",
    choices=["auto", "device", "global", "system"],
    help="Timestamp source used for cross-camera synchronization",
)

parser.add_argument("--z_min", type=float, default=1.2, help="Min valid depth (meters) for rasterization")
parser.add_argument("--z_max", type=float, default=1.9, help="Max valid depth (meters) for rasterization")
parser.add_argument(
    "--depth_colormap",
    type=str,
    default="inferno",
    choices=["viridis", "turbo", "magma", "inferno", "plasma", "jet"],
    help="OpenCV colormap for depth visualization output",
)

parser.add_argument("--save_png16", action="store_true", help="Also save raw 16-bit PNG depth in mm as *_mm16.png")
parser.add_argument("--save_npy", action="store_true", help="Save float32 depth (meters) as .npy")
parser.add_argument("--preview", action="store_true", help="Show a live preview window (scaled)")
parser.add_argument("--save_video", action="store_true", help="Save an MP4 video from the rendered depth visualization frames")
parser.add_argument("--video_fps", type=float, default=15.0, help="FPS for output video when --save_video is enabled")
parser.add_argument("--video_name", default="cam0_view_depth.mp4", help="Output video filename (placed inside out_dir)")

args = parser.parse_args()


def resolve_output_base_dir(calib_json: str, out_dir: str | None) -> Path:
    """Resolve output base directory for fused depth outputs."""
    if out_dir:
        return Path(out_dir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent.parent
    outputs_root = (repo_root / "outputs").resolve()
    calib_path = Path(calib_json).expanduser().resolve()

    try:
        # Expect calib under outputs/person_x/session_x/depth_calibration/depth_calibration.json
        rel = calib_path.relative_to(outputs_root)
        if len(rel.parts) >= 3:
            session_rel = rel.parent.parent
            return (outputs_root / session_rel / "fused_depth_maps").resolve()
    except ValueError:
        pass

    # Fallback for non-standard calib paths.
    return (calib_path.parent.parent / "fused_depth_maps").resolve()


# --------------------------
# 2) LOAD CALIBRATION JSON -
# --------------------------
def load_refined_extrinsics(path: str):
    """Load refined extrinsics and cam0 depth intrinsics from JSON file."""
    d = json.loads(Path(path).read_text())
    depthcam0_width = int(d["depthcam_0"]["width"])
    depthcam0_height = int(d["depthcam_0"]["height"])
    depthcam0_intrinsic_matrix = np.asarray(
        d["depthcam_0"]["depth_intrinsic_calibration_matrix"], dtype=np.float64
    ).reshape(3, 3)

    T_depthcam0_depthcam1 = np.asarray(d["T_depthcam_0_from_depthcam_1"], dtype=np.float64).reshape(4, 4)
    T_depthcam0_depthcam2 = np.asarray(d["T_depthcam_0_from_depthcam_2"], dtype=np.float64).reshape(4, 4)

    return depthcam0_intrinsic_matrix, depthcam0_width, depthcam0_height, T_depthcam0_depthcam1, T_depthcam0_depthcam2


# -----------------------------
# 3) ORBBEC DEPTH IO/PROJECT --
# -----------------------------
@dataclass
class DepthProjector:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    u: np.ndarray
    v: np.ndarray


@dataclass
class DepthPacket:
    depth16: np.ndarray
    timestamp_us: int


class OrbbecBagDepthReader:
    def __init__(self, bag_path: str, wait_ms: int, read_retries: int, timestamp_source: str):
        self.bag_path = bag_path
        self.wait_ms = int(wait_ms)
        self.read_retries = max(1, int(read_retries))
        self.timestamp_source = str(timestamp_source)
        self.pipeline = Pipeline(PlaybackDevice(bag_path))

        config = Config()
        device = self.pipeline.get_device()
        sensor_list = device.get_sensor_list()
        for i in range(len(sensor_list)):
            try:
                config.enable_stream(sensor_list[i].get_type())
            except Exception:
                pass

        self.pipeline.start(config)

        camera_param = self.pipeline.get_camera_param()
        intr = camera_param.depth_intrinsic
        self.projector = DepthProjector(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            width=0,
            height=0,
            u=np.empty((0, 0), dtype=np.float64),
            v=np.empty((0, 0), dtype=np.float64),
        )

    def close(self):
        self.pipeline.stop()

    def _frame_to_depth_u16(self, depth_frame) -> np.ndarray:
        depth_raw = depth_frame.get_data()
        h = depth_frame.get_height()
        w = depth_frame.get_width()
        depth = depth_raw.view(np.uint16).reshape(h, w).copy()
        return depth

    def _extract_timestamp_us(self, frame) -> int:
        candidate_sets = {
            "device": ["get_timestamp_us", "get_timestamp"],
            "global": ["get_global_timestamp_us"],
            "system": ["get_system_timestamp_us", "get_system_timestamp"],
            "auto": [
                "get_timestamp_us",
                "get_global_timestamp_us",
                "get_system_timestamp_us",
                "get_timestamp",
                "get_system_timestamp",
            ],
        }

        for method_name in candidate_sets[self.timestamp_source]:
            method = getattr(frame, method_name, None)
            if callable(method):
                try:
                    ts = int(method())
                except Exception:
                    continue
                if ts > 0:
                    # Legacy get_timestamp() commonly reports ms.
                    if method_name == "get_timestamp" and ts < 1_000_000_000:
                        ts *= 1000
                    return ts

        get_index = getattr(frame, "get_index", None)
        if callable(get_index):
            try:
                idx = int(get_index())
            except Exception:
                idx = -1
            if idx >= 0:
                return idx
        return -1

    def get_next_depth_packet(self) -> DepthPacket:
        for _ in range(self.read_retries):
            frameset = self.pipeline.wait_for_frames(self.wait_ms)
            if frameset is None:
                continue
            depth_frame = frameset.get_depth_frame()
            if depth_frame is None:
                continue
            ts_us = self._extract_timestamp_us(depth_frame)
            if ts_us < 0:
                continue
            return DepthPacket(
                depth16=self._frame_to_depth_u16(depth_frame),
                timestamp_us=ts_us,
            )
        return None

    def depth_to_points_in_depth_frame(self, depth16: np.ndarray) -> np.ndarray:
        h, w = depth16.shape

        if (self.projector.height != h) or (self.projector.width != w):
            u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            self.projector.width = w
            self.projector.height = h
            self.projector.u = u
            self.projector.v = v

        z = depth16.astype(np.float64) / 1000.0  # mm -> m
        valid = (z > 0.0) & np.isfinite(z)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)

        x = (self.projector.u - self.projector.cx) * z / self.projector.fx
        y = (self.projector.v - self.projector.cy) * z / self.projector.fy

        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        return pts


# --------------------------
# 4) GEOMETRY/RASTER UTILS -
# --------------------------
def apply_transform_to_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points in homogeneous coordinates."""
    if pts.shape[0] == 0:
        return pts
    num_pts = pts.shape[0]
    homo = np.ones((num_pts, 4), dtype=np.float64)
    homo[:, :3] = pts
    out = (T @ homo.T).T
    return out[:, :3]


def rasterize_cam0_depth(points_d0: np.ndarray, K0: np.ndarray, W: int, H: int, z_min: float, z_max: float) -> np.ndarray:
    """Z-buffer rasterizer into cam0 depth image plane."""
    fx = float(K0[0, 0])
    fy = float(K0[1, 1])
    cx = float(K0[0, 2])
    cy = float(K0[1, 2])

    x = points_d0[:, 0]
    y = points_d0[:, 1]
    z = points_d0[:, 2]

    valid = (z > z_min) & (z < z_max)
    x, y, z = x[valid], y[valid], z[valid]
    if z.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    u = (fx * (x / z) + cx).astype(np.int32)
    v = (fy * (y / z) + cy).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inside], v[inside], z[inside]

    depth = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), z.astype(np.float32))
    depth[~np.isfinite(depth)] = 0.0
    return depth


def depth_m_to_png16_mm(depth_m: np.ndarray) -> np.ndarray:
    return np.round(depth_m * 1000.0).astype(np.uint16)


def get_cv2_colormap(colormap_name: str):
    cmap = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "jet": cv2.COLORMAP_JET,
    }
    return cmap[colormap_name]


def depth_m_to_colormap_bgr(depth_m: np.ndarray, z_min: float, z_max: float, cv2_colormap: int) -> np.ndarray:
    denom = max(z_max - z_min, 1e-9)
    disp = np.clip((depth_m - z_min) / denom, 0.0, 1.0)
    disp_u8 = (disp * 255.0).astype(np.uint8)
    vis_bgr = cv2.applyColorMap(disp_u8, cv2_colormap)
    vis_bgr[depth_m <= 0] = 0
    return vis_bgr


def create_video_writer(video_path: Path, width: int, height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] Failed to open video writer for {video_path}")
    return writer


def get_next_synced_depth_pair(
    reader1: OrbbecBagDepthReader,
    reader2: OrbbecBagDepthReader,
    queue1: deque,
    queue2: deque,
    sync_tolerance_us: int,
):
    while True:
        if not queue1:
            packet1 = reader1.get_next_depth_packet()
            if packet1 is None:
                return None
            queue1.append(packet1)

        if not queue2:
            packet2 = reader2.get_next_depth_packet()
            if packet2 is None:
                return None
            queue2.append(packet2)

        p1 = queue1[0]
        p2 = queue2[0]
        delta_us = int(p1.timestamp_us - p2.timestamp_us)
        abs_delta_us = abs(delta_us)

        if abs_delta_us <= sync_tolerance_us:
            queue1.popleft()
            queue2.popleft()
            return p1, p2, abs_delta_us

        if delta_us < 0:
            queue1.popleft()
            packet1 = reader1.get_next_depth_packet()
            if packet1 is None:
                return None
            queue1.append(packet1)
        else:
            queue2.popleft()
            packet2 = reader2.get_next_depth_packet()
            if packet2 is None:
                return None
            queue2.append(packet2)


# -----------------------
# 5) MAIN LOOP ----------
# -----------------------
def main():
    base_out_dir = resolve_output_base_dir(args.calib_json, args.out_dir)
    plots_out_dir = base_out_dir / "plots"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    need_plots_dir = args.save_video or args.preview or args.save_png16 or (not args.save_npy)
    if need_plots_dir:
        plots_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Fused depth map directory: {base_out_dir}")
    if need_plots_dir:
        print(f"[INFO] Plot/video directory: {plots_out_dir}")

    K0, W0, H0, T_d0_d1, T_d0_d2 = load_refined_extrinsics(args.calib_json)
    cv2_colormap = get_cv2_colormap(args.depth_colormap)

    video_writer = None
    if args.save_video:
        video_path = plots_out_dir / args.video_name
        video_writer = create_video_writer(video_path, W0, H0, args.video_fps)
        print(f"[INFO] Saving video to: {video_path}")

    reader1 = OrbbecBagDepthReader(args.cam1_bag, args.wait_ms, args.read_retries, args.sync_timestamp)
    reader2 = OrbbecBagDepthReader(args.cam2_bag, args.wait_ms, args.read_retries, args.sync_timestamp)
    sync_tolerance_us = int(max(0.0, args.sync_tolerance_ms) * 1000.0)
    queue1 = deque()
    queue2 = deque()
    print(
        f"[INFO] Sync mode: timestamp_source={args.sync_timestamp}, "
        f"tolerance={args.sync_tolerance_ms:.3f} ms"
    )

    try:
        skip_count = 0
        skip_misses = 0
        while skip_count < args.start_idx and skip_misses < args.eos_patience:
            synced = get_next_synced_depth_pair(reader1, reader2, queue1, queue2, sync_tolerance_us)
            if synced is None:
                skip_misses += 1
                continue
            skip_count += 1
            skip_misses = 0
        if skip_count < args.start_idx:
            print("[INFO] Reached end while skipping.")
            return

        f = 0
        misses = 0
        no_frame_limit = args.max_frames <= 0
        while (no_frame_limit or f < args.max_frames) and misses < args.eos_patience:
            synced = get_next_synced_depth_pair(reader1, reader2, queue1, queue2, sync_tolerance_us)
            if synced is None:
                misses += 1
                continue
            misses = 0
            packet1, packet2, delta_us = synced
            depth1 = packet1.depth16
            depth2 = packet2.depth16

            pts1_d1 = reader1.depth_to_points_in_depth_frame(depth1)
            pts2_d2 = reader2.depth_to_points_in_depth_frame(depth2)

            pts1_d0 = apply_transform_to_points(T_d0_d1, pts1_d1)
            pts2_d0 = apply_transform_to_points(T_d0_d2, pts2_d2)
            fused_d0 = np.vstack([pts1_d0, pts2_d0])

            depth_m = rasterize_cam0_depth(fused_d0, K0, W0, H0, args.z_min, args.z_max)

            npy_stem = base_out_dir / f"cam0_view_depth_{f:06d}"
            plot_stem = plots_out_dir / f"cam0_view_depth_{f:06d}"

            if args.save_npy:
                np.save(str(npy_stem.with_suffix(".npy")), depth_m.astype(np.float32))

            vis_bgr = None
            need_vis = args.save_video or args.preview or args.save_png16 or (not args.save_npy)
            if need_vis:
                vis_bgr = depth_m_to_colormap_bgr(depth_m, args.z_min, args.z_max, cv2_colormap)

            if args.save_png16 or (not args.save_npy):
                cv2.imwrite(str(plot_stem.with_suffix(".png")), vis_bgr)

            if args.save_png16:
                depth_mm = depth_m_to_png16_mm(depth_m)
                cv2.imwrite(str(plot_stem.parent / f"{plot_stem.name}_mm16.png"), depth_mm)

            if video_writer is not None:
                video_writer.write(vis_bgr)

            if args.preview:
                cv2.imshow("cam0-view depth (preview)", vis_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            print(f"[INFO] Saved frame {f} (|delta_ts|={delta_us} us)")
            f += 1
        if misses >= args.eos_patience:
            print("[INFO] End of recording.")
    finally:
        reader1.close()
        reader2.close()
        if video_writer is not None:
            video_writer.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
