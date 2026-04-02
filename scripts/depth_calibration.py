# TODO: Remove Debug Prints and Code after debugging is complete
# ---------------
# 0) IMPORTS ----
# ---------------
# typing and data structures imports
from typing import Optional, List, Literal
from dataclasses import dataclass

# general imports
import argparse
import json
from pathlib import Path

# math and vision libs
import numpy as np
import cv2
import open3d as o3d

# Kinect SDK imports
from pyk4a import PyK4APlayback, CalibrationType
from pyk4a.transformation import depth_image_to_color_camera, depth_image_to_point_cloud

# Orbbec SDK imports
from pyorbbecsdk import (
    PlaybackDevice, Pipeline, Config,
    AlignFilter, OBStreamType,
    OBFormat, OBCameraParam, PointCloudFilter
)

from pupil_apriltags import Detector

# ----------------------
# 1) ARGPARSER SETUP ---
# ----------------------
argparse = argparse.ArgumentParser(description="Hybrid calibration script: camera 0 from Kinect (.mkv) and cameras 1..N from Orbbec (.bag).")
argparse.add_argument(
    "--session-folder",
    type=str,
    default=None,
    help=(
        "Path under ./data (for example: person_x/session_x/depth_calibration) "
        "or an explicit path to a depth_calibration/session directory. If provided, "
        "Kinect/Orbbec playback files are auto-discovered from <input>/rgb_depth_data."
    ),
)
argparse.add_argument("--k4a_playback_path", required=False, default=None, help="Path to Kinect recording (.mkv) for camera 0.")
argparse.add_argument("--orbbec_playback_paths", nargs="+", required=False, default=None, help="Paths to Orbbec recording files (.bag) for cameras 1..N.")

argparse.add_argument("--frame_idx"             , type=int  , default=30                            , help="Index of the frame to analyze.")
argparse.add_argument("--tag_size"              , type=float, default=0.15                          , help="Size of the AprilTag in meters.")
argparse.add_argument("--tag_id"                , type=int  , default=0                             , help="ID of the AprilTag to detect.")
argparse.add_argument("--debug_calib_dir"       , type=str  , default="./data/trial19/debug_calib"  , help="Directory to save debug images and logs for calibration.")
argparse.add_argument("--out_dir"               , type=str  , default="./data/trial19/outputs"      , help="Directory to save output point clouds and results.")

argparse.add_argument("--icp_voxel_sizes"       , type=float, nargs="+", default=[0.04, 0.02, 0.01]            , help="Voxel sizes for downsampling point clouds before each ICP stage.")
argparse.add_argument("--icp_max_iterations"    , type=int  , nargs="+", default=[60, 30, 14]                  , help="Iteration counts for each ICP stage.")
argparse.add_argument("--icp_lambda_geometric"  , type=float, default=0.968                         , help="Lambda parameter for geometric consistency in ICP.")

argparse.add_argument("--undistort" , action="store_true", help="Whether to undistort the grayscale image before AprilTag detection.")
argparse.add_argument("--debug"     , action="store_true", help="Whether to enable debug mode with additional visualizations and prints.")

args = argparse.parse_args()


def _resolve_inputs_from_session_folder(session_folder: str) -> tuple[Path, list[Path], Path]:
    repo_root = Path(__file__).resolve().parent.parent
    session_input = Path(session_folder).expanduser()

    candidate_session_dirs: list[Path] = []
    if session_input.is_absolute():
        candidate_session_dirs.append(session_input)
    else:
        candidate_session_dirs.append((repo_root / "data" / session_input).resolve())
        candidate_session_dirs.append(session_input.resolve())

    session_dir = next((p for p in candidate_session_dirs if p.exists()), None)
    if session_dir is None:
        raise FileNotFoundError(
            f"[ERROR] Could not find session folder '{session_folder}'. "
            f"Tried: {[str(p) for p in candidate_session_dirs]}"
        )

    if session_dir.name == "rgb_depth_data":
        rgb_depth_dir = session_dir
        depth_calibration_dir = session_dir.parent
    else:
        rgb_depth_dir = session_dir / "rgb_depth_data"
        if not rgb_depth_dir.exists():
            raise FileNotFoundError(
                f"[ERROR] Could not find rgb_depth_data under session folder: {session_dir}"
            )
        depth_calibration_dir = session_dir

    kinect_candidates = sorted(rgb_depth_dir.glob("kinect*.mkv"))
    if not kinect_candidates:
        raise FileNotFoundError(
            f"[ERROR] No Kinect .mkv files found in: {rgb_depth_dir}"
        )

    kinect_master = next((p for p in kinect_candidates if "master" in p.stem.lower()), None)
    if kinect_master is None:
        raise FileNotFoundError(
            f"[ERROR] Could not identify Kinect master file in: {rgb_depth_dir}. "
            "Expected something like kinect_master.mkv."
        )

    orbbec_candidates = sorted(
        rgb_depth_dir.glob("orbbec*.bag"),
        key=lambda p: ("master" not in p.stem.lower(), p.name.lower()),
    )
    if not orbbec_candidates:
        raise FileNotFoundError(
            f"[ERROR] No Orbbec .bag files found in: {rgb_depth_dir}"
        )

    return kinect_master, orbbec_candidates, depth_calibration_dir


def _resolve_cli_paths():
    if args.session_folder:
        repo_root = Path(__file__).resolve().parent.parent
        data_root = (repo_root / "data").resolve()

        kinect_master, orbbec_bags, depth_calibration_dir = _resolve_inputs_from_session_folder(args.session_folder)
        args.k4a_playback_path = str(kinect_master)
        args.orbbec_playback_paths = [str(p) for p in orbbec_bags]

        # Keep user-provided paths untouched; only override legacy defaults.
        if args.debug_calib_dir == "./data/trial19/debug_calib":
            args.debug_calib_dir = str((depth_calibration_dir / "debug_calib").resolve())
        if args.out_dir == "./data/trial19/outputs":
            try:
                rel_from_data = depth_calibration_dir.resolve().relative_to(data_root)
                args.out_dir = str((repo_root / "outputs" / rel_from_data).resolve())
            except ValueError:
                # Fallback for paths outside the repository's ./data tree.
                args.out_dir = str((repo_root / "outputs" / depth_calibration_dir.name).resolve())
    else:
        if not args.k4a_playback_path or not args.orbbec_playback_paths:
            raise ValueError(
                "[ERROR] Provide either --session-folder, or both "
                "--k4a_playback_path and --orbbec_playback_paths."
            )

    args.k4a_playback_path = str(Path(args.k4a_playback_path).expanduser().resolve())
    args.orbbec_playback_paths = [str(Path(p).expanduser().resolve()) for p in args.orbbec_playback_paths]


_resolve_cli_paths()


# -------------------------------------------
# 1.5) ORBBEC CALIBRATION WRAPPER CLASS -----
# -------------------------------------------
class OBCalibration:
    """
    Wraps OBCameraParam to provide the same calibration interface that was
    previously provided by pyk4a's calibration object.

    Key SDK facts (from types.cpp binding):
      - OBExtrinsic.rot        → (3, 3) float32 numpy array   (rotation)
      - OBExtrinsic.transform  → (3,)   float32 numpy array   (translation in mm)
      - The OBCameraParam.transform extrinsic goes from depth camera → color camera.
    """

    def __init__(self, camera_param: OBCameraParam):
        self.camera_param = camera_param

    def get_color_intrinsic_matrix(self) -> np.ndarray:
        intr = self.camera_param.rgb_intrinsic
        return np.array([[intr.fx, 0.0,     intr.cx],
                         [0.0,     intr.fy, intr.cy],
                         [0.0,     0.0,     1.0     ]], dtype=np.float64)

    def get_color_distortion_coefficients(self) -> np.ndarray:
        """Returns [k1, k2, p1, p2, k3] — the 5-coefficient OpenCV convention."""
        d = self.camera_param.rgb_distortion
        return np.array([d.k1, d.k2, d.p1, d.p2, d.k3], dtype=np.float64)

    def get_depth_intrinsic_matrix(self) -> np.ndarray:
        intr = self.camera_param.depth_intrinsic
        return np.array([[intr.fx, 0.0,     intr.cx],
                         [0.0,     intr.fy, intr.cy],
                         [0.0,     0.0,     1.0     ]], dtype=np.float64)

    def get_colorcam_from_depthcam_transform(self) -> np.ndarray:
        """
        Returns a 4×4 rigid-body transform T_colorcam_from_depthcam.
        Translation is converted from mm (SDK) to meters.
        """
        ext = self.camera_param.transform          # OBExtrinsic
        R   = ext.rot.astype(np.float64)           # (3, 3)
        t   = ext.transform.astype(np.float64) / 1000.0  # (3,)  mm → m
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T


@dataclass
class K4ACalibration:
    """Adapter around pyk4a calibration so the downstream pipeline is SDK-agnostic."""
    calibration: object

    def get_color_intrinsic_matrix(self) -> np.ndarray:
        return self.calibration.get_camera_matrix(CalibrationType.COLOR)

    def get_color_distortion_coefficients(self) -> np.ndarray:
        return self.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    def get_depth_intrinsic_matrix(self) -> np.ndarray:
        return self.calibration.get_camera_matrix(CalibrationType.DEPTH)

    def get_colorcam_from_depthcam_transform(self) -> np.ndarray:
        rotation, translation = self.calibration.get_extrinsic_parameters(CalibrationType.DEPTH, CalibrationType.COLOR)
        if (rotation is None) or (translation is None):
            raise RuntimeError("[ERROR] Failed to get depth->color extrinsics from Kinect calibration.")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return T


@dataclass
class HybridCapture:
    color: np.ndarray
    depth: np.ndarray
    source: Literal["k4a", "orbbec"]
    depth_in_color: Optional[np.ndarray] = None
    rgb_points: Optional[np.ndarray] = None


@dataclass
class PlaybackWrapper:
    calibration: object
    handle: Optional[object] = None

    def close(self):
        if self.handle is not None:
            self.handle.close()


# ------------------------
# 2) CAPTURE UTILS -------
# ------------------------
def _decode_color_frame(raw: np.ndarray, fmt: OBFormat, width: int, height: int) -> np.ndarray:
    """Convert a raw Orbbec color frame to a BGR uint8 (H, W, 3) numpy array."""
    if fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("[ERROR] Failed to decode MJPG color frame.")
        return bgr
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return raw.reshape(height, width, 3)
    if fmt == OBFormat.BGRA:
        return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if fmt == OBFormat.RGBA:
        return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    raise ValueError(f"[ERROR] Unsupported color format: {fmt}")


def _depth_frame_to_image(depth_frame) -> np.ndarray:
    """Convert an Orbbec depth frame buffer to a uint16 depth image in mm."""
    depth_raw = depth_frame.get_data()
    return depth_raw.view(np.uint16).reshape(
        depth_frame.get_height(), depth_frame.get_width()
    ).copy()


def _get_k4a_capture_at_idx(playback: PyK4APlayback, idx: int):
    capture = None
    for ii in range(idx + 1):
        capture = playback.get_next_capture()
        if capture is None:
            raise RuntimeError(
                f"[ERROR] Failed to read Kinect capture at index {idx}. "
                f"Only {ii} captures were available."
            )
    return capture


def _load_k4a_capture(playback_path: str):
    playback = PyK4APlayback(playback_path)
    playback.open()
    capture = _get_k4a_capture_at_idx(playback, args.frame_idx)
    wrapper = PlaybackWrapper(calibration=K4ACalibration(playback.calibration), handle=playback)
    hybrid_capture = HybridCapture(color=capture.color, depth=capture.depth, source="k4a")
    return wrapper, hybrid_capture


def _load_orbbec_capture(playback_path: str):
    pb_device = PlaybackDevice(playback_path)
    pipeline = Pipeline(pb_device)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    config = Config()
    device = pipeline.get_device()
    sensor_list = device.get_sensor_list()
    for i in range(len(sensor_list)):
        try:
            config.enable_stream(sensor_list[i].get_type())
        except Exception:
            pass

    pipeline.start(config)
    try:
        calibration = OBCalibration(pipeline.get_camera_param())

        frameset = None
        for i in range(args.frame_idx + 1):
            frameset = pipeline.wait_for_frames(2000)
            if frameset is None:
                raise RuntimeError(
                    f"[ERROR] Reached end of recording before frame {args.frame_idx} "
                    f"in {playback_path} (stopped at frame {i})."
                )

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError(
                f"[ERROR] Missing color or depth frame at index {args.frame_idx} "
                f"in {playback_path}."
            )

        aligned_frameset = align_filter.process(frameset)
        if aligned_frameset is None:
            raise RuntimeError(
                f"[ERROR] Failed to align depth to color at frame {args.frame_idx} "
                f"in {playback_path}."
            )
        aligned_frameset = aligned_frameset.as_frame_set()
        aligned_color_frame = aligned_frameset.get_color_frame() or color_frame
        aligned_depth_frame = aligned_frameset.get_depth_frame()
        if aligned_depth_frame is None:
            raise RuntimeError(
                f"[ERROR] Missing aligned depth frame at index {args.frame_idx} "
                f"in {playback_path}."
            )

        point_cloud_frame = point_cloud_filter.process(aligned_frameset)
        if point_cloud_frame is None:
            raise RuntimeError(f"[ERROR] Failed to create RGB point cloud frame in {playback_path}.")

        color_image = _decode_color_frame(
            aligned_color_frame.get_data(),
            aligned_color_frame.get_format(),
            aligned_color_frame.get_width(),
            aligned_color_frame.get_height(),
        )
        depth_image = _depth_frame_to_image(depth_frame)
        depth_in_color_image = _depth_frame_to_image(aligned_depth_frame)
        rgb_points = point_cloud_filter.calculate(point_cloud_frame)
    finally:
        pipeline.stop()

    wrapper = PlaybackWrapper(calibration=calibration, handle=None)
    capture = HybridCapture(
        color=color_image,
        depth=depth_image,
        source="orbbec",
        depth_in_color=depth_in_color_image,
        rgb_points=rgb_points,
    )
    return wrapper, capture


def get_captures():
    """
    Hybrid capture loader:
    - camera 0 from Kinect .mkv
    - cameras 1..N from Orbbec .bag
    """
    print(f"[INFO] Using camera 0 as reference (depthcam_0): {args.k4a_playback_path}")

    playbacks: List[PlaybackWrapper] = []
    captures: List[HybridCapture] = []

    k4a_wrapper, k4a_capture = _load_k4a_capture(args.k4a_playback_path)
    playbacks.append(k4a_wrapper)
    captures.append(k4a_capture)

    for playback_path in args.orbbec_playback_paths:
        ob_wrapper, ob_capture = _load_orbbec_capture(playback_path)
        playbacks.append(ob_wrapper)
        captures.append(ob_capture)

    if len(captures) < 2:
        raise RuntimeError("[ERROR] Hybrid calibration requires at least 2 cameras in total.")

    return playbacks, captures


# ---------------------------
# 3) CONVERSION UTILS -------
# ---------------------------
def decode_from_mjpg_to_bgr(color_image: np.ndarray):
    """Convert either Kinect MJPG/BGRA/BGR or Orbbec decoded BGR into BGR."""
    if isinstance(color_image, np.ndarray) and color_image.ndim == 1:
        bgr = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("[ERROR] Failed to decode MJPG color frame.")
        return bgr
    if color_image.ndim == 3 and color_image.shape[2] == 3:
        return color_image
    if color_image.ndim == 3 and color_image.shape[2] == 4:
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"[ERROR] Unexpected color format: shape={color_image.shape}")


def ensure_bgra_format(color_image: np.ndarray):
    """Ensure color image is (H, W, 4) BGRA."""
    if isinstance(color_image, np.ndarray) and color_image.ndim == 1:
        bgr = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("[ERROR] Failed to decode MJPG color frame.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    if color_image.ndim == 3 and color_image.shape[2] == 4:
        return color_image
    if color_image.ndim == 3 and color_image.shape[2] == 3:
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2BGRA)
    raise ValueError(f"[ERROR] Unexpected color format: shape={color_image.shape}")


def ensure_depth_format(depth_image: np.ndarray):
    """Ensure depth image is (H, W) uint16 (already guaranteed by get_captures)."""
    if depth_image.ndim == 2 and depth_image.dtype == np.uint16:
        return depth_image
    raise ValueError(
        f"[ERROR] Unexpected depth format: shape={depth_image.shape}, dtype={depth_image.dtype}"
    )


# -------------------------
# 4) 2D PLOTTING UTILS ----
# -------------------------
def plot_bgr_image(bgr_image: np.ndarray):
    """
    Plot a BGR image using OpenCV for debugging.
    """
    window_name = f"Camera - BGR"
    cv2.imshow(window_name, bgr_image)
    print(f"[{window_name}] Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# -----------------------------
# 5) APRILTAG DETECTION UTIL --
# -----------------------------
def get_apriltag_pose(
    gray_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    tag_size: float = 0.15,
    tag_id: Optional[int] = 0,
    undistort: bool = False,
):
    """
    Detect AprilTags in the grayscale image and estimate their poses using the camera intrinsics.
    Unchanged from the K4A version — operates purely on numpy arrays.
    """
    detector = Detector(families="tagStandard41h12", refine_edges=True)

    if undistort:
        if distortion_coeffs is None:
            raise ValueError("[ERROR] distortion_coeffs must be provided when undistort=True")
        print("[INFO] Undistorting grayscale image before AprilTag detection.")
        h, w = gray_image.shape[:2]
        distortion_coeffs = distortion_coeffs.reshape(-1, 1)
        intrinsic_matrix_undistorted, _ = cv2.getOptimalNewCameraMatrix(
            intrinsic_matrix, distortion_coeffs, (w, h), alpha=0.0
        )
        gray_image_used       = cv2.undistort(gray_image, intrinsic_matrix, distortion_coeffs, None, intrinsic_matrix_undistorted)
        fx, fy                = float(intrinsic_matrix_undistorted[0, 0]), float(intrinsic_matrix_undistorted[1, 1])
        cx, cy                = float(intrinsic_matrix_undistorted[0, 2]), float(intrinsic_matrix_undistorted[1, 2])
        intrinsic_matrix_used = intrinsic_matrix_undistorted
    else:
        fx, fy                = float(intrinsic_matrix[0, 0]), float(intrinsic_matrix[1, 1])
        cx, cy                = float(intrinsic_matrix[0, 2]), float(intrinsic_matrix[1, 2])
        intrinsic_matrix_used = intrinsic_matrix
        gray_image_used       = gray_image

    tags = detector.detect(gray_image_used, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size)
    print(f"[INFO] Detected {len(tags)} AprilTags in the image.")

    chosen_tag = None
    for tag in tags:
        if tag.tag_id == tag_id:
            chosen_tag = tag
            print(f"[INFO] Tag ID {tag_id} detected with pose:\nRotation:\n{tag.pose_R}\nTranslation:\n{tag.pose_t}")
            break

    if chosen_tag is None:
        raise RuntimeError(f"[ERROR] Tag ID {tag_id} not detected in the image.")

    tag_corners_2d  = np.asarray(chosen_tag.corners, dtype=np.float64)
    tag_center_2d   = np.asarray(chosen_tag.center, dtype=np.float64)
    tag_rotation    = np.asarray(chosen_tag.pose_R, dtype=np.float64)
    tag_translation = np.asarray(chosen_tag.pose_t, dtype=np.float64).reshape(3)
    tag_pose_error  = float(chosen_tag.pose_err)
    print(f"[INFO] Tag ID {tag_id} pose estimation error: {tag_pose_error}"); print_separator()

    return tag_corners_2d, tag_center_2d, tag_rotation, tag_translation, tag_pose_error, gray_image_used, intrinsic_matrix_used


# --------------------------
# 6) TRANSFORMATION UTILS --
# --------------------------
def convert_RT_to_homogeneous_transform(rotation: np.ndarray, translation: np.ndarray):
    if rotation.shape != (3, 3):
        raise ValueError(f"[ERROR] Rotation must be a 3x3 matrix, but got shape {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"[ERROR] Translation must be a 3-element vector, but got shape {translation.shape}")
    homogeneous_transform = np.eye(4)
    homogeneous_transform[:3, :3] = rotation
    homogeneous_transform[:3, 3]  = translation
    return homogeneous_transform


def invert_homogeneous_transform(transform: np.ndarray):
    if transform.shape != (4, 4):
        raise ValueError(f"[ERROR] Transform must be a 4x4 matrix, but got shape {transform.shape}")
    R     = transform[:3, :3]
    t     = transform[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    transform_inv = np.eye(4)
    transform_inv[:3, :3] = R_inv
    transform_inv[:3, 3]  = t_inv
    return transform_inv


def get_colorcam_from_depthcam_transform(calibration) -> np.ndarray:
    """Returns T_colorcam_from_depthcam (4x4, translation in meters)."""
    return calibration.get_colorcam_from_depthcam_transform()


# -------------------------
# 6) POINT CLOUD UTILS ----
# -------------------------
def _depth_to_point_cloud_mm(depth_in_colorcam: np.ndarray, calibration: OBCalibration) -> np.ndarray:
    """
    Back-project an aligned depth image (color-cam space, uint16 mm) to a
    (H, W, 3) point cloud in millimeters (matches K4A convention so the
    /1000 in build_colored_point_cloud still works).

    Replaces pyk4a's depth_image_to_point_cloud().
    """
    color_intr = calibration.camera_param.rgb_intrinsic
    color_h, color_w = depth_in_colorcam.shape

    u, v = np.meshgrid(np.arange(color_w), np.arange(color_h))
    z = depth_in_colorcam.astype(np.float64)   # mm

    x = (u - color_intr.cx) * z / color_intr.fx  # mm
    y = (v - color_intr.cy) * z / color_intr.fy  # mm

    return np.stack([x, y, z], axis=-1)  # (H, W, 3), mm


def build_colored_point_cloud_orbbec_aligned_depth(color_image_bgra: np.ndarray, depth_image: np.ndarray, calibration: OBCalibration):
    """
    Build a colored point cloud in the color-camera frame from SDK-aligned depth + color images.

    depth_image   : (H_color, W_color) uint16, mm, already aligned to the color camera by Orbbec AlignFilter
    color_image_bgra : (H_color, W_color, 4) uint8 BGRA
    calibration   : OBCalibration

    Replaces the K4A path that called depth_image_to_color_camera() first.
    Depth-to-color reprojection is handled by Orbbec's built-in AlignFilter in get_captures().
    """
    point_cloud_in_colorcam_mm = _depth_to_point_cloud_mm(depth_image, calibration)
    point_cloud_in_colorcam    = point_cloud_in_colorcam_mm.reshape(-1, 3) / 1000.0  # mm → m

    # BGRA → BGR (Nx3), each row corresponds to the matching point
    color_image_rgb = color_image_bgra.reshape(-1, 4)[:, :3]

    z = point_cloud_in_colorcam[:, 2]
    valid_mask     = (z > 0) & (z < 5)
    non_black_mask = np.any(color_image_rgb > 0, axis=1)
    final_mask     = valid_mask & non_black_mask

    points_final = point_cloud_in_colorcam[final_mask]
    colors_final = color_image_rgb[final_mask][:, ::-1].astype(np.float64) / 255.0  # BGR → RGB, [0,1]

    if points_final.shape != colors_final.shape:
        raise ValueError(
            f"[ERROR] Number of points and colors must match, "
            f"but got {points_final.shape[0]} points and {colors_final.shape[0]} colors."
        )

    colored_point_cloud = o3d.geometry.PointCloud()
    colored_point_cloud.points = o3d.utility.Vector3dVector(points_final)
    colored_point_cloud.colors = o3d.utility.Vector3dVector(colors_final)
    return colored_point_cloud


def build_colored_point_cloud_k4a(color_image_bgra: np.ndarray, depth_image: np.ndarray, calibration: K4ACalibration):
    """Build a colored point cloud from Kinect depth/color using pyk4a transforms."""
    calibration_obj = calibration.calibration
    thread_safe = getattr(calibration_obj, "thread_safe", True)

    depth_in_colorcam = depth_image_to_color_camera(depth_image, calibration_obj, thread_safe=thread_safe)
    if depth_in_colorcam is None:
        raise RuntimeError("[ERROR] Failed to transform Kinect depth to color camera frame.")

    point_cloud_in_colorcam_mm = depth_image_to_point_cloud(
        depth_in_colorcam,
        calibration_obj,
        thread_safe=thread_safe,
        calibration_type_depth=False,
    )
    if point_cloud_in_colorcam_mm is None:
        raise RuntimeError("[ERROR] Failed to convert Kinect depth to point cloud.")

    point_cloud_in_colorcam = point_cloud_in_colorcam_mm.reshape(-1, 3) / 1000.0
    color_image_rgb = color_image_bgra.reshape(-1, 4)[:, :3]

    z = point_cloud_in_colorcam[:, 2]
    valid_mask = (z > 0) & (z < 5)
    non_black_mask = np.any(color_image_rgb > 0, axis=1)
    final_mask = valid_mask & non_black_mask

    points_final = point_cloud_in_colorcam[final_mask]
    colors_final = color_image_rgb[final_mask][:, ::-1].astype(np.float64) / 255.0

    if points_final.shape != colors_final.shape:
        raise ValueError(
            f"[ERROR] Number of points and colors must match, "
            f"but got {points_final.shape[0]} points and {colors_final.shape[0]} colors."
        )

    colored_point_cloud = o3d.geometry.PointCloud()
    colored_point_cloud.points = o3d.utility.Vector3dVector(points_final)
    colored_point_cloud.colors = o3d.utility.Vector3dVector(colors_final)
    return colored_point_cloud


def build_colored_point_cloud_pointcloudfilter(rgb_points: np.ndarray):
    if rgb_points.ndim != 2 or rgb_points.shape[1] != 6:
        raise ValueError(f"Expected (N, 6), got {rgb_points.shape}")

    points = rgb_points[:, :3].astype(np.float64) / 1000.0
    colors = rgb_points[:, 3:6].astype(np.float64) / 255.0

    z = points[:, 2]
    valid_mask = (z > 0) & (z < 5)
    non_black_mask = np.any(colors > 0, axis=1)
    mask = valid_mask & non_black_mask

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return pcd



def downsample_point_cloud_and_estimate_normals(point_cloud: o3d.geometry.PointCloud, voxel_size: float):
    p = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    if p.is_empty():
        return p
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30),
        fast_normal_computation=True,
    )
    p.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    return p


def run_color_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_correspondence_distance: float,
    max_iterations: int,
    lambda_geometric: float = 0.968,
):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iterations
    )
    est = o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=lambda_geometric)

    candidate_distances = [max_correspondence_distance, max_correspondence_distance * 2.0]
    for attempt_idx, correspondence_distance in enumerate(candidate_distances):
        try:
            res = o3d.pipelines.registration.registration_colored_icp(
                source=source_pcd,
                target=target_pcd,
                max_correspondence_distance=correspondence_distance,
                init=init_transform,
                estimation_method=est,
                criteria=criteria,
            )
            used_fallback = attempt_idx > 0
            return res.transformation, float(res.fitness), float(res.inlier_rmse), correspondence_distance, used_fallback
        except RuntimeError as exc:
            no_correspondence_error = "No correspondences found" in str(exc)
            last_attempt = attempt_idx == (len(candidate_distances) - 1)
            if (not no_correspondence_error) or last_attempt:
                raise
            print(
                f"[WARN] Colored ICP found no correspondences at distance={correspondence_distance:.6f}. "
                f"Retrying with distance={candidate_distances[attempt_idx + 1]:.6f}."
            )


# -------------------------
# 7) 3D PLOTTING UTILS ----
# -------------------------
def plot_colored_point_cloud(colored_point_cloud: o3d.geometry.PointCloud, window_name: str = "Colored Point Cloud", y_axis_up: bool = False):
    if colored_point_cloud is None or len(colored_point_cloud.points) == 0:
        raise ValueError("[ERROR] Colored point cloud is empty or None.")

    pcd_vis = o3d.geometry.PointCloud(colored_point_cloud)

    if y_axis_up:
        T_vis = np.eye(4, dtype=np.float64)
        T_vis[1, 1] = -1
        T_vis[2, 2] = -1
        pcd_vis.transform(T_vis)

    o3d.visualization.draw_geometries([pcd_vis], window_name=window_name)


# -------------------------
# 7.5) MAKE PRETTY PRINTS -
# -------------------------
def print_separator():
    print(f"----------------------------------------------------------------------------")


# ---------------------
# 8) MAIN FUNCTION ----
# ---------------------
def main():

    # -------------------------------------------------------------
    # 0) get the capture objects for each camera
    # -------------------------------------------------------------
    playbacks, captures = get_captures()

    # ----------------------------------------------------------------------
    # 1) tag detection and pose estimation for each camera
    # ----------------------------------------------------------------------
    print_separator(); print(f"Performing AprilTag detection and pose estimation for each camera..."); print_separator()

    initial_T_colorcam_from_tag_list = []
    tag_pose_error_list              = []

    for ii, (playback, capture) in enumerate(zip(playbacks, captures)):

        # capture.color is already a decoded BGR (H, W, 3) array
        bgr_image  = decode_from_mjpg_to_bgr(capture.color)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        calibration      = playback.calibration                         # OBCalibration
        intrinsic_matrix = calibration.get_color_intrinsic_matrix()
        distortion_coeffs = calibration.get_color_distortion_coefficients()
        print(f"[INFO] Camera {ii} intrinsic matrix:\n{intrinsic_matrix}"); print_separator()
        print(f"[INFO] Camera {ii} distortion coefficients:\n{distortion_coeffs}"); print_separator()



        _, _, tag_rotation_ii, tag_translation_ii, tag_pose_error_ii, _, _ = get_apriltag_pose(
            gray_image, intrinsic_matrix, distortion_coeffs,
            tag_size=args.tag_size, tag_id=args.tag_id, undistort=args.undistort,
        )

        T_colorcam_ii_from_tag = convert_RT_to_homogeneous_transform(tag_rotation_ii, tag_translation_ii)

        initial_T_colorcam_from_tag_list.append(T_colorcam_ii_from_tag)
        tag_pose_error_list.append(tag_pose_error_ii)

    if args.debug:
        print(f"[INFO] Initial transforms from tag frame to color camera frame for each camera:")
        for ii, T_colorcam_ii_from_tag in enumerate(initial_T_colorcam_from_tag_list):
            print(f"Camera {ii}:\n{T_colorcam_ii_from_tag}\n"); print_separator()

    # ------------------------------------------------------------------------------
    # 2) Initial Camera Pose Estimation from the AprilTag poses wrt COLOR camera 0
    # ------------------------------------------------------------------------------
    print_separator(); print(f"Performing initial camera pose estimation..."); print_separator()

    initial_T_colorcam_0_from_colorcam_ii_list = [np.eye(4, dtype=np.float64)]

    for ii in range(1, len(initial_T_colorcam_from_tag_list)):
        initial_T_colorcam_0_from_colorcam_ii = (
            initial_T_colorcam_from_tag_list[0]
            @ invert_homogeneous_transform(initial_T_colorcam_from_tag_list[ii])
        )
        initial_T_colorcam_0_from_colorcam_ii_list.append(initial_T_colorcam_0_from_colorcam_ii)

    if args.debug:
        print(f"[INFO] Initial camera pose estimates (transform from color camera i to color camera 0):")
        for ii, T in enumerate(initial_T_colorcam_0_from_colorcam_ii_list):
            print(f"Camera {ii}:\n{T}\n"); print_separator()

    # ------------------------------------------------------------------------------
    # 3) Build Colored Point Clouds (in each camera's color camera frame)
    # ------------------------------------------------------------------------------
    print_separator(); print(f"Building colored point clouds for each camera in their respective color camera frames..."); print_separator()

    colored_point_clouds_colorcam_ii_list = []

    for ii, (playback, capture) in enumerate(zip(playbacks, captures)):
        color_image_bgra = ensure_bgra_format(capture.color)
        calibration = playback.calibration

        if capture.source == "k4a":
            depth_image = ensure_depth_format(capture.depth)
            colored_point_cloud_ii = build_colored_point_cloud_k4a(color_image_bgra, depth_image, calibration)
        elif capture.source == "orbbec":
            if capture.rgb_points is None:
                raise RuntimeError(f"[ERROR] Camera {ii} is Orbbec but rgb_points is missing.")
            colored_point_cloud_ii = build_colored_point_cloud_pointcloudfilter(capture.rgb_points)
        else:
            raise RuntimeError(f"[ERROR] Unsupported capture source: {capture.source}")

        colored_point_clouds_colorcam_ii_list.append(colored_point_cloud_ii)

    if args.debug:
        print(f"Plotting the colored point clouds for each camera in their respective color camera frames for debugging...")
        for ii, pcd in enumerate(colored_point_clouds_colorcam_ii_list):
            if pcd is not None:
                print(f"[INFO] Camera {ii}: Point cloud has {len(pcd.points)} points.")
                plot_colored_point_cloud(pcd, window_name=f"Camera {ii} Point Cloud in Color Camera Frame", y_axis_up=True)
            else:
                raise RuntimeError(f"[ERROR] Camera {ii}: Failed to build point cloud.")

    if args.debug:
        debug_calib_dir = Path(args.debug_calib_dir)
        print_separator(); print(f"[INFO] Saving debug images and logs to directory: {debug_calib_dir}"); print_separator()
        debug_calib_dir.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(
            str(debug_calib_dir / "pcd_camera_0_wrt_colorcam0.ply"),
            colored_point_clouds_colorcam_ii_list[0], write_ascii=True,
        )

        for ii in range(1, len(colored_point_clouds_colorcam_ii_list)):
            pcd = o3d.geometry.PointCloud(colored_point_clouds_colorcam_ii_list[ii])
            pcd.transform(initial_T_colorcam_0_from_colorcam_ii_list[ii])
            o3d.io.write_point_cloud(
                str(debug_calib_dir / f"pcd_camera_{ii}_wrt_colorcam0_before_refinement.ply"),
                pcd, write_ascii=True,
            )

        print_separator(); print(f"Plotting all colored point clouds transformed to color camera 0 frame for debugging..."); print_separator()
        for ii in range(len(colored_point_clouds_colorcam_ii_list)):
            pcd = o3d.geometry.PointCloud(colored_point_clouds_colorcam_ii_list[ii])
            pcd.transform(initial_T_colorcam_0_from_colorcam_ii_list[ii])
            plot_colored_point_cloud(pcd, window_name=f"Camera {ii} Point Cloud Transformed to Color Camera 0 Frame", y_axis_up=True)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 4) Multi-stage Colored ICP Refinement
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    print_separator(); print(f"Performing multi-stage colored ICP refinement..."); print_separator()

    if len(args.icp_voxel_sizes) != len(args.icp_max_iterations):
        raise ValueError(
            f"[ERROR] The number of ICP voxel sizes and iteration counts must match, "
            f"but got {len(args.icp_voxel_sizes)} voxel sizes and {len(args.icp_max_iterations)} iteration counts."
        )

    icp_report = {}

    refined_T_colorcam_0_from_colorcam_ii_list = initial_T_colorcam_0_from_colorcam_ii_list.copy()

    for stage, (voxel_size, max_iterations) in enumerate(zip(args.icp_voxel_sizes, args.icp_max_iterations)):

        print_separator(); print(f"Stage {stage+1}: Performing ICP refinement with voxel size {voxel_size} and {max_iterations} iterations..."); print_separator()

        target_pcd = downsample_point_cloud_and_estimate_normals(colored_point_clouds_colorcam_ii_list[0], voxel_size=voxel_size)

        if target_pcd.is_empty():
            raise RuntimeError(f"[ERROR] Stage {stage+1}: Target point cloud is empty after downsampling with voxel_size={voxel_size}.")

        for ii in range(1, len(colored_point_clouds_colorcam_ii_list)):

            source_pcd_ii = downsample_point_cloud_and_estimate_normals(colored_point_clouds_colorcam_ii_list[ii], voxel_size=voxel_size)
            if source_pcd_ii.is_empty():
                raise RuntimeError(f"[ERROR] Stage {stage+1}: Source point cloud for camera {ii} is empty after downsampling with voxel_size={voxel_size}.")

            refined_T, icp_fitness, icp_inlier_rmse, used_correspondence_distance, used_fallback = run_color_icp(
                source_pcd=source_pcd_ii,
                target_pcd=target_pcd,
                init_transform=refined_T_colorcam_0_from_colorcam_ii_list[ii],
                max_correspondence_distance=voxel_size,
                max_iterations=max_iterations,
                lambda_geometric=args.icp_lambda_geometric,
            )

            refined_T_colorcam_0_from_colorcam_ii_list[ii] = refined_T

            icp_report.setdefault(f"stage_{stage+1}", {})[f"camera_{ii}"] = {
                "voxel_size"     : voxel_size,
                "max_iterations" : max_iterations,
                "used_max_correspondence_distance": used_correspondence_distance,
                "used_no_correspondence_fallback": bool(used_fallback),
                "icp_fitness"    : icp_fitness,
                "icp_inlier_rmse": icp_inlier_rmse,
            }
            print(
                f"[INFO] Stage {stage+1}, Camera {ii}: ICP fitness = {icp_fitness:.6f}, "
                f"ICP inlier RMSE = {icp_inlier_rmse:.6f}, correspondence_distance = {used_correspondence_distance:.6f}"
            )

    if args.debug:
        debug_calib_dir = Path(args.debug_calib_dir)
        for ii in range(1, len(colored_point_clouds_colorcam_ii_list)):
            pcd = o3d.geometry.PointCloud(colored_point_clouds_colorcam_ii_list[ii])
            pcd.transform(refined_T_colorcam_0_from_colorcam_ii_list[ii])
            o3d.io.write_point_cloud(
                str(debug_calib_dir / f"pcd_camera_{ii}_wrt_colorcam0_after_refinement.ply"),
                pcd, write_ascii=True,
            )

        print_separator(); print(f"Plotting all colored point clouds transformed to color camera 0 frame after ICP refinement for debugging..."); print_separator()
        for ii in range(len(colored_point_clouds_colorcam_ii_list)):
            pcd = o3d.geometry.PointCloud(colored_point_clouds_colorcam_ii_list[ii])
            pcd.transform(refined_T_colorcam_0_from_colorcam_ii_list[ii])
            plot_colored_point_cloud(pcd, window_name=f"Camera {ii} Point Cloud Transformed to Color Camera 0 Frame After ICP Refinement", y_axis_up=True)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 5) Convert Color Camera Extrinsics to Depth Camera Extrinsics and Save Results
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    print_separator(); print(f"Converting color camera extrinsics to depth camera extrinsics and saving results..."); print_separator()

    T_colorcam_0_from_depthcam_0  = get_colorcam_from_depthcam_transform(playbacks[0].calibration)
    T_depthcam_0_from_colorcam_0  = invert_homogeneous_transform(T_colorcam_0_from_depthcam_0)

    cam_count = len(refined_T_colorcam_0_from_colorcam_ii_list)
    T_depthcam_0_from_depthcam_ii_list = [np.eye(4, dtype=np.float64) for _ in range(cam_count)]

    for ii in range(1, cam_count):
        T_colorcam_ii_from_depthcam_ii = get_colorcam_from_depthcam_transform(playbacks[ii].calibration)
        T_colorcam_0_from_colorcam_ii  = refined_T_colorcam_0_from_colorcam_ii_list[ii]

        T_colorcam_0_from_depthcam_ii       = T_colorcam_0_from_colorcam_ii @ T_colorcam_ii_from_depthcam_ii
        T_depthcam_0_from_depthcam_ii_list[ii] = T_depthcam_0_from_colorcam_0 @ T_colorcam_0_from_depthcam_ii

    if args.debug:
        print(f"[INFO] Final refined transforms from depth camera i to depth camera 0:"); print_separator()
        for ii, T in enumerate(T_depthcam_0_from_depthcam_ii_list):
            print(f"Camera {ii}:\n{T}\n"); print_separator()

        print_separator(); print(f"Plotting all colored point clouds transformed to depth camera 0 frame after ICP refinement for debugging..."); print_separator()
        for ii in range(len(colored_point_clouds_colorcam_ii_list)):
            pcd = o3d.geometry.PointCloud(colored_point_clouds_colorcam_ii_list[ii])
            pcd.transform(T_depthcam_0_from_depthcam_ii_list[ii])
            plot_colored_point_cloud(pcd, window_name=f"Camera {ii} Point Cloud Transformed to Depth Camera 0 Frame After ICP Refinement", y_axis_up=True)

    # ------------------------------------------------------------------------------------------------
    # 6) Save the final refined extrinsics to a JSON file
    # ------------------------------------------------------------------------------------------------
    print_separator(); print(f"Saving final refined extrinsics to JSON file..."); print_separator()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Depth intrinsics for camera 0 (reference)
    depth_intrinsic_calibration_matrix_cam0 = playbacks[0].calibration.get_depth_intrinsic_matrix()
    depth_height_cam0, depth_width_cam0     = captures[0].depth.shape[:2]

    save_data = {
        "reference_frame": "depthcam_0",
        "depthcam_0": {
            "width" : int(depth_width_cam0),
            "height": int(depth_height_cam0),
            "depth_intrinsic_calibration_matrix": depth_intrinsic_calibration_matrix_cam0.tolist(),
        },
        "meta": {
            "k4a_playback_path"   : args.k4a_playback_path,
            "orbbec_playback_paths": args.orbbec_playback_paths,
            "frame_idx"           : int(args.frame_idx),
            "tag_size"            : float(args.tag_size),
            "tag_id"              : int(args.tag_id),
            "undistort"           : bool(args.undistort),
            "apriltag_pose_error" : [float(e) for e in tag_pose_error_list],
            "colored_icp_report"  : icp_report,
            "icp_voxels_sizes"    : [float(s) for s in args.icp_voxel_sizes],
            "icp_max_iterations"  : [int(n)   for n in args.icp_max_iterations],
            "icp_lambda_geometric": float(args.icp_lambda_geometric),
        },
    }
    for ii in range(1, cam_count):
        save_data[f"T_depthcam_0_from_depthcam_{ii}"] = T_depthcam_0_from_depthcam_ii_list[ii].tolist()

    out_json = out_dir / "depth_calibration.json"
    out_json.write_text(json.dumps(save_data, indent=4))
    print(f"[INFO] Final refined extrinsics and metadata saved to {out_json}")

    # ------------------------------------------------
    # 7) Close the playbacks and enter debug mode
    # ------------------------------------------------
    for playback in playbacks:
        playback.close()

    if args.debug:
        import ipdb; ipdb.set_trace()


# ----------------------
# 9) ENTRY POINT -------
# ----------------------
if __name__ == "__main__":
    main()
