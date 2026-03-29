import datetime
import time
import csv
import os
from pathlib import Path
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


# define the local devices and their configurations for the Kinect Rig
# Current rig mapping from `k4arecorder.exe --list`:
LOCAL_DEVICES = [
    {"device_id": 1, "name": "master", "mode": WiredSyncMode.MASTER, "sub_delay_usec": 0},
    {"device_id": 2, "name": "subordinate_1", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 200},
    {"device_id": 4, "name": "subordinate_2", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 400},
    {"device_id": 3, "name": "subordinate_3", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 600},
    {"device_id": 0, "name": "subordinate_4", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 800},
]

MASTER_BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_3072P,
    depth_mode=DepthMode.WFOV_2X2BINNED,
    camera_fps=FPS.FPS_15,
    synchronized_images_only=True,
)

SUBORDINATE_BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_3072P,
    depth_mode=DepthMode.OFF,
    camera_fps=FPS.FPS_15,
    synchronized_images_only=False,
)


def build_config(mode: WiredSyncMode, sub_delay_usec: int) -> Config:
    if mode == WiredSyncMode.MASTER:
        return Config(**MASTER_BASE_CONFIG, wired_sync_mode=WiredSyncMode.MASTER)
    return Config(
        **SUBORDINATE_BASE_CONFIG,
        wired_sync_mode=WiredSyncMode.SUBORDINATE,
        subordinate_delay_off_master_usec=sub_delay_usec,
    )

def make_recording_dir() -> tuple[Path, str]:
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"recordings/recording_{now_str}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, now_str

def main():

    # make the recording directory
    out_dir, now_str = make_recording_dir()

    # check if the number of connected devices matches the expected number of local devices
    num_connected = connected_device_count()
    if num_connected != len(LOCAL_DEVICES):
        print(f"Expected {len(LOCAL_DEVICES)} devices, but found {num_connected} connected. Exiting.")
        return RuntimeError(f"Expected {len(LOCAL_DEVICES)} devices, but found {num_connected} connected.")
    
