# ----------------------------------------------------------
# 0) IMPORTS
# ----------------------------------------------------------
# --- system ---
import argparse
import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any

# --- pyk4a ---
from pyk4a import (
    ColorResolution,
    Config,
    DepthMode,
    FPS,
    ImageFormat,
    PyK4A,
    PyK4ARecord,
    WiredSyncMode,
    connected_device_count,
)

# --- debugging ---
import ipdb as pdb


# -----------------------------------------------------------
# 1) KINECT 5 DEVICE CONFIGURATION
# -----------------------------------------------------------
@dataclass(frozen=True)
class DeviceSpec:
    device_id: int
    name: str
    mode: WiredSyncMode
    sub_delay_usec: int


@dataclass(frozen=True)
class CameraBaseConfig:
    color_format: ImageFormat
    color_resolution: ColorResolution
    depth_mode: DepthMode
    camera_fps: FPS
    synchronized_images_only: bool


@dataclass
class KinectRigConfig:
    devices: List[DeviceSpec]
    master_base_config: CameraBaseConfig
    subordinate_base_config: CameraBaseConfig


class LocalRigProfiles:
    @staticmethod
    def five_camera_profile() -> KinectRigConfig:
        devices = [
            DeviceSpec(device_id=1, name="master", mode=WiredSyncMode.MASTER, sub_delay_usec=0),
            DeviceSpec(device_id=2, name="subordinate_1", mode=WiredSyncMode.SUBORDINATE, sub_delay_usec=200),
            DeviceSpec(device_id=4, name="subordinate_2", mode=WiredSyncMode.SUBORDINATE, sub_delay_usec=400),
            DeviceSpec(device_id=3, name="subordinate_3", mode=WiredSyncMode.SUBORDINATE, sub_delay_usec=600),
            DeviceSpec(device_id=0, name="subordinate_4", mode=WiredSyncMode.SUBORDINATE, sub_delay_usec=800),
        ]
        master_base = CameraBaseConfig(
            color_format=ImageFormat.COLOR_MJPG,
            color_resolution=ColorResolution.RES_3072P,
            depth_mode=DepthMode.WFOV_2X2BINNED,
            camera_fps=FPS.FPS_15,
            synchronized_images_only=True,
        )
        subordinate_base = CameraBaseConfig(
            color_format=ImageFormat.COLOR_MJPG,
            color_resolution=ColorResolution.RES_3072P,
            depth_mode=DepthMode.OFF,
            camera_fps=FPS.FPS_15,
            synchronized_images_only=False,
        )
        return KinectRigConfig(
            devices=devices,
            master_base_config=master_base,
            subordinate_base_config=subordinate_base,
        )


# -----------------------------------------------------------
# 2) RUNTIME DATA CLASSES
# -----------------------------------------------------------
@dataclass
class CameraRuntime:
    spec: DeviceSpec
    config: Config
    device: PyK4A
    record: Optional[PyK4ARecord] = None
    ts_csv_path: Optional[str] = None
    ts_fh: Optional[Any] = None
    ts_writer: Optional[csv.writer] = None
    frame_idx: int = 0


# -----------------------------------------------------------
# 3) HELPER FUNCTIONS
# -----------------------------------------------------------
def make_recording_dir(base_path: str) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_dir = os.path.join(base_path, f"recording_{timestamp}")
    os.makedirs(recording_dir, exist_ok=True)
    return recording_dir, timestamp


def normalize_camera_name(name: str) -> str:
    return name.replace("_", "")


# -----------------------------------------------------------
# 4) MANAGER
# -----------------------------------------------------------
class KinectRigManager:
    def __init__(self, rig_config: KinectRigConfig, recording_dir: str, capture_timeout_ms: int = 10000) -> None:
        self.rig_config = rig_config
        self.recording_dir = recording_dir
        self.capture_timeout_ms = capture_timeout_ms
        self.cameras: List[CameraRuntime] = []

    def _build_config(self, spec: DeviceSpec) -> Config:
        base = (
            self.rig_config.master_base_config
            if spec.mode == WiredSyncMode.MASTER
            else self.rig_config.subordinate_base_config
        )

        if spec.mode == WiredSyncMode.MASTER:
            return Config(
                color_format=base.color_format,
                color_resolution=base.color_resolution,
                depth_mode=base.depth_mode,
                camera_fps=base.camera_fps,
                synchronized_images_only=base.synchronized_images_only,
                wired_sync_mode=WiredSyncMode.MASTER,
            )

        return Config(
            color_format=base.color_format,
            color_resolution=base.color_resolution,
            depth_mode=base.depth_mode,
            camera_fps=base.camera_fps,
            synchronized_images_only=base.synchronized_images_only,
            wired_sync_mode=WiredSyncMode.SUBORDINATE,
            subordinate_delay_off_master_usec=spec.sub_delay_usec,
        )

    def initialize_cameras(self) -> None:
        self.cameras = []
        for spec in self.rig_config.devices:
            cfg = self._build_config(spec)
            dev = PyK4A(config=cfg, device_id=spec.device_id)
            self.cameras.append(CameraRuntime(spec=spec, config=cfg, device=dev))

    def _start_order(self) -> List[CameraRuntime]:
        subs = [c for c in self.cameras if c.spec.mode == WiredSyncMode.SUBORDINATE]
        masters = [c for c in self.cameras if c.spec.mode == WiredSyncMode.MASTER]
        return subs + masters

    def _stop_order(self) -> List[CameraRuntime]:
        masters = [c for c in self.cameras if c.spec.mode == WiredSyncMode.MASTER]
        subs = [c for c in self.cameras if c.spec.mode == WiredSyncMode.SUBORDINATE]
        return masters + subs

    def start_all_devices(self) -> None:
        for cam in self._start_order():
            cam.device.start()
            print(
                f"started device_id={cam.spec.device_id} serial={cam.device.serial} "
                f"name={cam.spec.name} mode={cam.spec.mode.name} "
                f"sub_delay_usec={cam.spec.sub_delay_usec} sync_jack={cam.device.sync_jack_status}"
            )

    def open_all_recordings(self) -> None:
        for cam in self.cameras:
            cam_name = normalize_camera_name(cam.spec.name)
            mkv_path = os.path.join(self.recording_dir, f"kinect_{cam_name}.mkv")
            ts_csv_path = os.path.splitext(mkv_path)[0] + ".save_timestamps.csv"

            record = PyK4ARecord(path=mkv_path, config=cam.config, device=cam.device)
            record.create()

            ts_fh = open(ts_csv_path, "w", newline="", encoding="utf-8")
            ts_writer = csv.writer(ts_fh)
            ts_writer.writerow(
                [
                    "frame_idx",
                    "host_capture_start_ns",
                    "save_timestamp_ns",
                    "host_write_complete_ns",
                    "color_timestamp_usec",
                    "color_system_timestamp_nsec",
                    "depth_timestamp_usec",
                    "depth_system_timestamp_nsec",
                    "color_present",
                    "depth_enabled",
                ]
            )

            cam.record = record
            cam.ts_csv_path = ts_csv_path
            cam.ts_fh = ts_fh
            cam.ts_writer = ts_writer

    def capture_once_all(self) -> None:
        for cam in self.cameras:
            host_capture_start_ns = time.time_ns()
            cap = cam.device.get_capture(timeout=self.capture_timeout_ms)

            if cap.color is None:
                print(f"skipping frame on {cam.spec.name} (device_id={cam.spec.device_id}) because color is missing")
                continue

            is_master = cam.spec.mode == WiredSyncMode.MASTER
            has_depth = is_master and (cap.depth is not None)
            if is_master and not has_depth:
                print(f"skipping frame on master {cam.spec.name} because depth is missing")
                continue

            save_timestamp_ns = time.time_ns()
            color_ts_usec = cap.color_timestamp_usec if cap.color is not None else ""
            color_sys_ts_ns = cap.color_system_timestamp_nsec if cap.color is not None else ""
            depth_ts_usec = cap.depth_timestamp_usec if has_depth else ""
            depth_sys_ts_ns = cap.depth_system_timestamp_nsec if has_depth else ""

            cam.record.write_capture(cap)
            host_write_complete_ns = time.time_ns()

            cam.ts_writer.writerow(
                [
                    cam.frame_idx,
                    host_capture_start_ns,
                    save_timestamp_ns,
                    host_write_complete_ns,
                    color_ts_usec,
                    color_sys_ts_ns,
                    depth_ts_usec,
                    depth_sys_ts_ns,
                    int(cap.color is not None),
                    int(has_depth),
                ]
            )
            cam.frame_idx += 1

    def close_all(self) -> None:
        # Flush/close record files first.
        for cam in self.cameras:
            if cam.record is not None:
                cam.record.flush()
                cam.record.close()

            if cam.ts_fh is not None:
                cam.ts_fh.flush()
                cam.ts_fh.close()

        # Stop devices in timing-safe order (master first, then subs).
        for cam in self._stop_order():
            try:
                cam.device.stop()
            except Exception as e:
                print(f"warning: failed stopping {cam.spec.name}: {e}")

        for cam in self.cameras:
            capture_count = cam.record.captures_count if cam.record is not None else 0
            print(
                f"saved camera={cam.spec.name} frames={capture_count} "
                f"timestamps={cam.ts_csv_path}"
            )


# -----------------------------------------------------------
# 5) ARGPARSE
# -----------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Kinect streams with configurable output directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recordings",
        help="Base output directory for recording folders (default: recordings).",
    )
    parser.add_argument(
        "--stop-file",
        type=str,
        default=None,
        help="If provided, recording stops when this file path exists.",
    )
    parser.add_argument(
        "--capture-timeout-ms",
        type=int,
        default=10000,
        help="Capture timeout per camera in milliseconds.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enter ipdb before capture loop starts.",
    )
    return parser.parse_args()


# -----------------------------------------------------------
# 6) MAIN
# -----------------------------------------------------------
def main() -> None:
    args = parse_args()

    recording_dir, timestamp_string = make_recording_dir(args.output_dir)
    print(f"Recording directory: {recording_dir}")
    print(f"Timestamp: {timestamp_string}")

    rig_config = LocalRigProfiles.five_camera_profile()

    available_devices = connected_device_count()
    if available_devices < len(rig_config.devices):
        raise RuntimeError(f"Expected at least {len(rig_config.devices)} devices, found {available_devices}")

    manager = KinectRigManager(
        rig_config=rig_config,
        recording_dir=recording_dir,
        capture_timeout_ms=args.capture_timeout_ms,
    )
    manager.initialize_cameras()
    manager.start_all_devices()
    manager.open_all_recordings()

    if args.debug:
        pdb.set_trace()

    try:
        print("Recording... Press CTRL-C to stop.")
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                print(f"Stop file found: {args.stop_file}")
                break
            manager.capture_once_all()
    except KeyboardInterrupt:
        print("Stopping recording...")
    finally:
        manager.close_all()


# -----------------------------------------------------------
# 7) ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
