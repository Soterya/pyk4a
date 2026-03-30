"""
Running Instructions:

python scripts/capture_femto_bolt.py
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
from pyk4a.errors import K4ATimeoutException

# Local setup: 2 Femto Bolt devices total (1 MASTER + 1 SUBORDINATE).
# Update the device_id values if `k4arecorder.exe --list` reports a different order.
# IMPORTANT: subordinate delays must be globally unique across the full rig.
LOCAL_DEVICES = [
    {"device_id": 0, "name": "master", "mode": WiredSyncMode.MASTER, "sub_delay_usec": 0},
    {"device_id": 1, "name": "subordinate_1", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 200},
]

OUT_DIR = Path("multi_mkv/femto_bolt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_1080P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    camera_fps=FPS.FPS_15,
    synchronized_images_only=True,
)


def build_config(mode: WiredSyncMode, sub_delay_usec: int) -> Config:
    if mode == WiredSyncMode.MASTER:
        return Config(**BASE_CONFIG, wired_sync_mode=WiredSyncMode.MASTER)
    if mode == WiredSyncMode.SUBORDINATE:
        return Config(
            **BASE_CONFIG,
            wired_sync_mode=WiredSyncMode.SUBORDINATE,
            subordinate_delay_off_master_usec=sub_delay_usec,
        )
    return Config(**BASE_CONFIG, wired_sync_mode=WiredSyncMode.STANDALONE)


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
        sync_jack = c["device"].sync_jack_status
        print(
            f"started device_id={c['spec']['device_id']} serial={c['device'].serial} "
            f"name={c['spec']['name']} mode={c['spec']['mode'].name} "
            f"sub_delay_usec={c['spec']['sub_delay_usec']} sync_jack={sync_jack}"
        )
        if sync_jack == (False, False):
            print(
                f"warning: device {c['spec']['device_id']} reports no sync cables detected; "
                "master/subordinate capture may timeout"
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
        ts_writer.writerow(["frame_idx", "save_timestamp_ns", "color_timestamp_usec", "depth_timestamp_usec"])

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
        print("Recording MKV on local Femto Bolt rig with sidecar save timestamps...")
        print("Press CTRL-C to stop.")
        while True:
            for out in outputs:
                try:
                    cap = out["camera"]["device"].get_capture(timeout=10000)
                except K4ATimeoutException:
                    print(
                        f"timed out waiting for frame on device {out['camera']['spec']['device_id']} "
                        f"name={out['camera']['spec']['name']}"
                    )
                    continue
                if cap.color is None or cap.depth is None:
                    print(
                        f"skipping frame on device {out['camera']['spec']['device_id']} "
                        "because color/depth is missing"
                    )
                    continue

                save_timestamp_ns = time.time_ns()
                out["ts_writer"].writerow(
                    [out["frame_idx"], save_timestamp_ns, cap.color_timestamp_usec, cap.depth_timestamp_usec]
                )
                out["record"].write_capture(cap)
                out["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping Femto Bolt recording")
    finally:
        for out in outputs:
            if out["record"].captures_count > 0:
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
