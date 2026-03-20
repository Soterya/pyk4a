"""
Running Instructions:

python scripts/record_five_camera_pc_csv_depth_off.py
"""

import csv
import time
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

# Local setup: 5 devices total (1 MASTER + 4 SUBORDINATES), depth disabled.
# IMPORTANT: subordinate delays must be globally unique across the full rig.
# Adjust device_id values to match the order on this machine.
LOCAL_DEVICES = [
    {"device_id": 0, "name": "master", "mode": WiredSyncMode.MASTER, "sub_delay_usec": 0},
    {"device_id": 1, "name": "subordinate_1", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 200},
    {"device_id": 2, "name": "subordinate_2", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 400},
    {"device_id": 3, "name": "subordinate_3", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 600},
    {"device_id": 4, "name": "subordinate_4", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 800},
]

OUT_DIR = Path("multi_mkv/five_camera_pc_depth_off")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_3072P,
    depth_mode=DepthMode.OFF,
    camera_fps=FPS.FPS_15,
    synchronized_images_only=False,
)


def build_config(mode: WiredSyncMode, sub_delay_usec: int) -> Config:
    if mode == WiredSyncMode.MASTER:
        return Config(**BASE_CONFIG, wired_sync_mode=WiredSyncMode.MASTER)
    return Config(
        **BASE_CONFIG,
        wired_sync_mode=WiredSyncMode.SUBORDINATE,
        subordinate_delay_off_master_usec=sub_delay_usec,
    )


def main() -> None:
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
        serial = c["device"].serial
        mkv_path = OUT_DIR / f"{c['spec']['name']}_dev{c['spec']['device_id']}_{serial}.mkv"
        ts_csv_path = mkv_path.with_suffix(".save_timestamps.csv")

        record = PyK4ARecord(path=mkv_path, config=c["config"], device=c["device"])
        record.create()

        ts_fh = open(ts_csv_path, "w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_fh)
        ts_writer.writerow(["frame_idx", "save_timestamp_ns", "color_timestamp_usec"])

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
        print("Recording color-only MKV on local 5-camera rig (depth OFF)...")
        print("Press CTRL-C to stop.")
        while True:
            for out in outputs:
                cap = out["camera"]["device"].get_capture(timeout=10000)
                if cap.color is None:
                    print(f"skipping frame on device {out['camera']['spec']['device_id']} because color is missing")
                    continue

                # Host timestamp taken right before persisting this capture.
                save_timestamp_ns = time.time_ns()
                out["ts_writer"].writerow([out["frame_idx"], save_timestamp_ns, cap.color_timestamp_usec])
                out["record"].write_capture(cap)
                out["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping 5-camera recording (depth OFF)")
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
