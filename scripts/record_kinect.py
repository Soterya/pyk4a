"""
Running Instructions:

python scripts/record_five_camera_pc_master_depth_only_csv.py
"""

import csv
import os
import time
from pathlib import Path
from datetime import datetime

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

# Local setup: 5 devices total (1 MASTER + 4 SUBORDINATES).
# MASTER records RGB + depth. SUBORDINATES record RGB only.
# Current rig mapping from `k4arecorder.exe --list`:
# index 1 serial ...794512 = master
# index 2 serial ...1812   = subordinate_1
# index 4 serial ...5012   = subordinate_2
# index 3 serial ...4912   = subordinate_3
# index 0 serial ...4512   = subordinate_4
# IMPORTANT: subordinate delays must be globally unique across the full rig.
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
    timestamp_str = os.environ.get("RECORD_SESSION_TIMESTAMP")
    if not timestamp_str:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_output_dir = os.environ.get("RECORD_SESSION_DIR")
    if env_output_dir:
        output_dir = Path(env_output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent / "recordings" / f"session_{timestamp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, timestamp_str


def main() -> None:
    out_dir, timestamp_str = make_recording_dir()
    stop_file_path = os.environ.get("RECORD_STOP_FILE")
    available = connected_device_count()
    if available < len(LOCAL_DEVICES):
        raise RuntimeError(f"Expected at least {len(LOCAL_DEVICES)} devices, found {available}")

    cameras = []
    for spec in LOCAL_DEVICES:
        cfg = build_config(spec["mode"], spec["sub_delay_usec"])
        dev = PyK4A(config=cfg, device_id=spec["device_id"])
        cameras.append({"spec": spec, "device": dev, "config": cfg})

    # Start local subordinates first, then the local master.
    start_order = [c for c in cameras if c["spec"]["mode"] == WiredSyncMode.SUBORDINATE]
    start_order += [c for c in cameras if c["spec"]["mode"] == WiredSyncMode.MASTER]

    for c in start_order:
        c["device"].start()
        print(
            f"started device_id={c['spec']['device_id']} serial={c['device'].serial} "
            f"name={c['spec']['name']} mode={c['spec']['mode'].name} "
            f"sub_delay_usec={c['spec']['sub_delay_usec']} sync_jack={c['device'].sync_jack_status}"
        )

    outputs = []
    for c in cameras:
        normalized_name = c["spec"]["name"].replace("_", "")
        mkv_path = out_dir / f"kinect_{normalized_name}_{timestamp_str}.mkv"
        ts_csv_path = mkv_path.with_suffix(".save_timestamps.csv")

        record = PyK4ARecord(path=mkv_path, config=c["config"], device=c["device"])
        record.create()

        ts_fh = open(ts_csv_path, "w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_fh)
        ts_writer.writerow(
            [
                "frame_idx",
                "save_timestamp_ns",
                "color_timestamp_usec",
                "depth_timestamp_usec",
                "depth_enabled",
            ]
        )

        outputs.append(
            {
                "camera": c,
                "record": record,
                "mkv_path": mkv_path,
                "ts_csv_path": ts_csv_path,
                "ts_fh": ts_fh,
                "ts_writer": ts_writer,
                "frame_idx": 0,
            }
        )

    try:
        print("Recording MKV on local 5-camera rig (master depth ON, subordinates depth OFF)...")
        print("Press CTRL-C to stop.")
        while True:
            if stop_file_path and os.path.exists(stop_file_path):
                break
            for out in outputs:
                spec = out["camera"]["spec"]
                cap = out["camera"]["device"].get_capture(timeout=10000)

                if cap.color is None:
                    print(f"skipping frame on device {spec['device_id']} because color is missing")
                    continue

                is_master = spec["mode"] == WiredSyncMode.MASTER
                if is_master and cap.depth is None:
                    print(f"skipping frame on device {spec['device_id']} because depth is missing on master")
                    continue

                # Host timestamp taken right before persisting this capture.
                save_timestamp_ns = time.time_ns()
                depth_ts_usec = cap.depth_timestamp_usec if cap.depth is not None else ""
                out["ts_writer"].writerow(
                    [
                        out["frame_idx"],
                        save_timestamp_ns,
                        cap.color_timestamp_usec,
                        depth_ts_usec,
                        int(is_master),
                    ]
                )
                out["record"].write_capture(cap)
                out["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping 5-camera recording")
    finally:
        for out in outputs:
            out["record"].flush()
            out["record"].close()
            out["ts_fh"].flush()
            out["ts_fh"].close()
            out["camera"]["device"].stop()
            print(
                f"saved {out['mkv_path']} frames={out['record'].captures_count} "
                f"timestamps={out['ts_csv_path']}"
            )


if __name__ == "__main__":
    main()
