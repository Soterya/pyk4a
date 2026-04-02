# ---------------
# 0) IMPORTS ----
# ---------------
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ----------------------
# 1) ARGPARSER SETUP ---
# ----------------------
parser = argparse.ArgumentParser(
    description="Depth-only fusion (master+subordinate) from synchronized Orbbec depth maps into cam0 depth view using depth_calibration.json"
)

parser.add_argument("--calib_json", required=True, help="Path to depth_calibration.json (from hybrid calibration)")
parser.add_argument("--sync_csv", required=True, help="Path to orbbec_synced_data_companion.csv")
parser.add_argument("--cam1_depth_dir", required=True, help="Path to master depth map folder (e.g. .../orbbec_depth_master)")
parser.add_argument("--cam2_depth_dir", required=True, help="Path to subordinate depth map folder (e.g. .../orbbec_depth_subordinate)")
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
    help="Maximum number of synchronized rows to process; use <=0 to process all available rows",
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
parser.add_argument("--save_npy", action="store_true", help="Save depth as .npz with uint16 millimeters")
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
    frame_idx_master: int
    frame_idx_subordinate: int
    save_timestamp_ns_master: int


@dataclass
class SyncedDepthPair:
    packet_master: DepthPacket
    packet_subordinate: DepthPacket
    source_row: dict


def _projector_from_depth_calib_json(calib_depth_json: str) -> DepthProjector:
    calib = json.loads(calib_depth_json)
    intr = calib["intrinsic"]
    return DepthProjector(
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        width=0,
        height=0,
        u=np.empty((0, 0), dtype=np.float64),
        v=np.empty((0, 0), dtype=np.float64),
    )


def _depth_to_points_in_depth_frame(depth16: np.ndarray, projector: DepthProjector) -> np.ndarray:
    h, w = depth16.shape

    if (projector.height != h) or (projector.width != w):
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        projector.width = w
        projector.height = h
        projector.u = u
        projector.v = v

    z = depth16.astype(np.float64) / 1000.0  # mm -> m
    valid = (z > 0.0) & np.isfinite(z)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    x = (projector.u - projector.cx) * z / projector.fx
    y = (projector.v - projector.cy) * z / projector.fy

    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return pts


def _load_depth_npz(path: Path) -> np.ndarray:
    with np.load(path) as npz_data:
        if "depth" in npz_data:
            depth = npz_data["depth"]
        elif len(npz_data.files) == 1:
            depth = npz_data[npz_data.files[0]]
        else:
            raise RuntimeError(f"[ERROR] Could not infer depth array key in npz file: {path}")

    if depth.ndim != 2:
        raise RuntimeError(f"[ERROR] Expected depth array shape (H, W), got {depth.shape} in {path}")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    return depth


def load_synced_depth_packets(sync_csv: Path, cam1_depth_dir: Path, cam2_depth_dir: Path):
    pairs: list[SyncedDepthPair] = []
    projector_cam1 = None
    projector_cam2 = None

    with sync_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            master_rel = row["depth_path_master"]
            subordinate_rel = row["depth_path_subordinate"]

            master_name = Path(master_rel).name
            subordinate_name = Path(subordinate_rel).name

            depth1_path = cam1_depth_dir / master_name
            depth2_path = cam2_depth_dir / subordinate_name

            if not depth1_path.exists() or not depth2_path.exists():
                print(
                    f"[WARN] Skipping row {row_idx}: missing depth file(s) "
                    f"{depth1_path if not depth1_path.exists() else ''} "
                    f"{depth2_path if not depth2_path.exists() else ''}"
                )
                continue

            if projector_cam1 is None:
                projector_cam1 = _projector_from_depth_calib_json(row["calib_depth_master_json"])
            if projector_cam2 is None:
                projector_cam2 = _projector_from_depth_calib_json(row["calib_depth_subordinate_json"])

            packet_master = DepthPacket(
                depth16=_load_depth_npz(depth1_path),
                frame_idx_master=int(row["frame_idx_master"]),
                frame_idx_subordinate=int(row["frame_idx_subordinate"]),
                save_timestamp_ns_master=int(row["save_timestamp_ns_master"]),
            )
            packet_subordinate = DepthPacket(
                depth16=_load_depth_npz(depth2_path),
                frame_idx_master=int(row["frame_idx_master"]),
                frame_idx_subordinate=int(row["frame_idx_subordinate"]),
                save_timestamp_ns_master=int(row["save_timestamp_ns_master"]),
            )
            pairs.append(SyncedDepthPair(packet_master=packet_master, packet_subordinate=packet_subordinate, source_row=row))

    if projector_cam1 is None or projector_cam2 is None:
        raise RuntimeError("[ERROR] Could not load camera depth intrinsics from sync CSV.")
    return pairs, projector_cam1, projector_cam2


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

    all_pairs, projector1, projector2 = load_synced_depth_packets(
        sync_csv=Path(args.sync_csv).expanduser().resolve(),
        cam1_depth_dir=Path(args.cam1_depth_dir).expanduser().resolve(),
        cam2_depth_dir=Path(args.cam2_depth_dir).expanduser().resolve(),
    )
    print(f"[INFO] Loaded {len(all_pairs)} synchronized depth-map pairs from CSV.")

    if args.start_idx >= len(all_pairs):
        print("[INFO] start_idx is beyond available synchronized pairs. Nothing to process.")
        if video_writer is not None:
            video_writer.release()
        return

    process_pairs = all_pairs[args.start_idx:]
    if args.max_frames > 0:
        process_pairs = process_pairs[: args.max_frames]

    # Companion CSV for fused depth exports: keeps original synchronized row fields and appends fused outputs.
    fused_companion_csv = base_out_dir / "fused_depth_maps_companion.csv"
    fused_plot_companion_csv = plots_out_dir / "fused_depth_maps_plots_companion.csv"

    original_columns = []
    if process_pairs:
        original_columns = list(process_pairs[0].source_row.keys())
    fused_extra_columns = [
        "fused_depth_path",
        "fused_plot_path",
        "fused_plot_mm16_path",
    ]

    try:
        with fused_companion_csv.open("w", newline="", encoding="utf-8") as fused_fh:
            fused_writer = csv.DictWriter(fused_fh, fieldnames=original_columns + fused_extra_columns)
            fused_writer.writeheader()

            plot_writer = None
            plot_fh = None
            if need_plots_dir:
                plot_fh = fused_plot_companion_csv.open("w", newline="", encoding="utf-8")
                plot_writer = csv.DictWriter(
                    plot_fh,
                    fieldnames=["save_timestamp_ns_master", "frame_idx_master", "plot_filename"],
                )
                plot_writer.writeheader()

            for f, pair in enumerate(process_pairs):
                packet1 = pair.packet_master
                packet2 = pair.packet_subordinate

                depth1 = packet1.depth16
                depth2 = packet2.depth16

                pts1_d1 = _depth_to_points_in_depth_frame(depth1, projector1)
                pts2_d2 = _depth_to_points_in_depth_frame(depth2, projector2)

                pts1_d0 = apply_transform_to_points(T_d0_d1, pts1_d1)
                pts2_d0 = apply_transform_to_points(T_d0_d2, pts2_d2)
                fused_d0 = np.vstack([pts1_d0, pts2_d0])

                depth_m = rasterize_cam0_depth(fused_d0, K0, W0, H0, args.z_min, args.z_max)

                depth_stem = base_out_dir / f"{f:06d}"
                plot_stem = plots_out_dir / f"cam0_view_depth_{f:06d}"

                if args.save_npy:
                    depth_mm_u16 = depth_m_to_png16_mm(depth_m)
                    np.savez_compressed(
                        str(depth_stem.with_suffix(".npz")),
                        depth=depth_mm_u16,
                    )

                vis_bgr = None
                need_vis = args.save_video or args.preview or args.save_png16 or (not args.save_npy)
                if need_vis:
                    vis_bgr = depth_m_to_colormap_bgr(depth_m, args.z_min, args.z_max, cv2_colormap)

                plot_png_path = ""
                plot_mm16_path = ""
                if args.save_png16 or (not args.save_npy):
                    plot_png_path = str(plot_stem.with_suffix(".png").relative_to(base_out_dir.parent))
                    cv2.imwrite(str(plot_stem.with_suffix(".png")), vis_bgr)

                if args.save_png16:
                    depth_mm = depth_m_to_png16_mm(depth_m)
                    mm16_path = plot_stem.parent / f"{plot_stem.name}_mm16.png"
                    cv2.imwrite(str(mm16_path), depth_mm)
                    plot_mm16_path = str(mm16_path.relative_to(base_out_dir.parent))

                if video_writer is not None:
                    video_writer.write(vis_bgr)

                if args.preview:
                    cv2.imshow("cam0-view depth (preview)", vis_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                fused_npy_path = str(depth_stem.with_suffix(".npz").relative_to(base_out_dir.parent))
                if not args.save_npy:
                    fused_npy_path = ""

                row_out = dict(pair.source_row)
                row_out["fused_depth_path"] = fused_npy_path
                row_out["fused_plot_path"] = plot_png_path
                row_out["fused_plot_mm16_path"] = plot_mm16_path
                fused_writer.writerow(row_out)

                if plot_writer is not None and plot_png_path:
                    plot_writer.writerow(
                        {
                            "save_timestamp_ns_master": packet1.save_timestamp_ns_master,
                            "frame_idx_master": packet1.frame_idx_master,
                            "plot_filename": Path(plot_png_path).name,
                        }
                    )

                print(
                    f"[INFO] Saved frame {f} "
                    f"(master_idx={packet1.frame_idx_master}, subordinate_idx={packet1.frame_idx_subordinate})"
                )

            if plot_fh is not None:
                plot_fh.close()
    finally:
        if video_writer is not None:
            video_writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"[INFO] Saved fused companion CSV: {fused_companion_csv}")
    if need_plots_dir:
        print(f"[INFO] Saved fused plot companion CSV: {fused_plot_companion_csv}")


if __name__ == "__main__":
    main()
