"""
Running Instructions:

python scripts/record_subordinate_pc_csv.py
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

# Subordinate PC setup: 2 devices, both SUBORDINATE.
# IMPORTANT: delays must be globally unique across all subordinates in the full rig.
LOCAL_DEVICES = [
    {"device_id": 1, "name": "master",      "mode": WiredSyncMode.MASTER,       "sub_delay_usec": 0},
    {"device_id": 0, "name": "subordinate", "mode": WiredSyncMode.SUBORDINATE,  "sub_delay_usec": 200},
]

OUT_DIR = Path("multi_mkv/subordinate_pc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_3072P,
    depth_mode=DepthMode.WFOV_2X2BINNED,
    camera_fps=FPS.FPS_15,
    synchronized_images_only=True,
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
        raise RuntimeError(f"Expected at least {len(LOCAL_DEVICES)} devices on subordinate PC, found {available}")

    cameras = []
    for spec in LOCAL_DEVICES:
        cfg = build_config(spec["mode"], spec["sub_delay_usec"])
        dev = PyK4A(config=cfg, device_id=spec["device_id"])
        cameras.append({"spec": spec, "device": dev, "config": cfg})

    for c in cameras:
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
        ts_writer.writerow(["frame_idx", "post_write_timestamp_ns", "depth_timestamp_usec"])

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
        print("Recording MKV on subordinate PC with sidecar save timestamps...")
        print("Press CTRL-C to stop.")
        while True:
            for out in outputs:
                cap = out["camera"]["device"].get_capture(timeout=10000)
                if cap.color is None or cap.depth is None:
                    print(
                        f"skipping frame on device {out['camera']['spec']['device_id']} "
                        "because color/depth is missing"
                    )
                    continue

                out["record"].write_capture(cap)
                # Host timestamp taken immediately after persisting this capture.
                post_write_timestamp_ns = time.time_ns()
                out["ts_writer"].writerow([out["frame_idx"], post_write_timestamp_ns, cap.depth_timestamp_usec])
                out["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping subordinate PC recording")
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
