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
    WiredSyncMode,
    connected_device_count,
)

# Subordinate PC setup: 2 devices, both SUBORDINATE.
# IMPORTANT: delays must be globally unique across all subordinates in the full rig.
LOCAL_DEVICES = [
    # {"device_id": 1, "name": "master", "mode": WiredSyncMode.MASTER, "sub_delay_usec": 0}, # Use when both master and subordinate are connected to the same PC. 
    {"device_id": 1, "name": "sub_remote_1", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 200},
    {"device_id": 0, "name": "sub_remote_2", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 400},
]

OUT_DIR = Path("multi_csv/subordinate_pc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_BGRA32,
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


def flatten_to_csv_field(frame_array) -> str:
    # Keep each row to three columns by serializing the flattened arrays into one string field.
    return " ".join(map(str, frame_array.reshape(-1).tolist()))


def main() -> None:
    available = connected_device_count()
    if available < len(LOCAL_DEVICES):
        raise RuntimeError(f"Expected at least {len(LOCAL_DEVICES)} devices on subordinate PC, found {available}")

    cameras = []
    for spec in LOCAL_DEVICES:
        cfg = build_config(spec["mode"], spec["sub_delay_usec"])
        dev = PyK4A(config=cfg, device_id=spec["device_id"])
        cameras.append({"spec": spec, "device": dev})

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
        csv_path = OUT_DIR / f"{c['spec']['name']}_dev{c['spec']['device_id']}_{serial}.csv"
        fh = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "rgb_data", "depth_data"])
        outputs.append(
            {
                "camera": c,
                "serial": serial,
                "csv_path": csv_path,
                "fh": fh,
                "writer": writer,
                "frame_idx": 0,
            }
        )

    try:
        print("Recording flattened frames to CSV on subordinate PC...")
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

                timestamp_ns = time.time_ns()
                rgb_data = flatten_to_csv_field(cap.color)
                depth_data = flatten_to_csv_field(cap.depth)
                out["writer"].writerow([timestamp_ns, rgb_data, depth_data])
                out["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping subordinate PC CSV recording")
    finally:
        for out in outputs:
            out["fh"].flush()
            out["fh"].close()
            out["camera"]["device"].stop()
            print(f"saved {out['csv_path']} rows={out['frame_idx']}")


if __name__ == "__main__":
    main()
