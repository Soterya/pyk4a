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

# Local single-device capture target:
# index 1 serial 000214794512 = master camera
TARGET_DEVICE = {
    "device_id": 1,
    "name": "master",
    "mode": WiredSyncMode.MASTER,
    "sub_delay_usec": 0,
    "expected_serial": "000214794512",
}

OUT_DIR = Path("multi_mkv/master_pc")
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
    if available <= TARGET_DEVICE["device_id"]:
        raise RuntimeError(
            f"Expected device index {TARGET_DEVICE['device_id']} to exist, found {available} connected devices"
        )

    cfg = build_config(TARGET_DEVICE["mode"], TARGET_DEVICE["sub_delay_usec"])
    dev = PyK4A(config=cfg, device_id=TARGET_DEVICE["device_id"])
    dev.start()
    if dev.serial != TARGET_DEVICE["expected_serial"]:
        dev.stop()
        raise RuntimeError(
            f"Device at index {TARGET_DEVICE['device_id']} has serial {dev.serial}, "
            f"expected {TARGET_DEVICE['expected_serial']}"
        )

    print(
        f"started device_id={TARGET_DEVICE['device_id']} serial={dev.serial} "
        f"name={TARGET_DEVICE['name']} mode={TARGET_DEVICE['mode'].name} "
        f"sub_delay_usec={TARGET_DEVICE['sub_delay_usec']} sync_jack={dev.sync_jack_status}"
    )

    mkv_path = OUT_DIR / f"{TARGET_DEVICE['name']}_dev{TARGET_DEVICE['device_id']}_{dev.serial}.mkv"
    sidecar_path = mkv_path.with_suffix(".save_timestamps.csv")
    rec = PyK4ARecord(path=mkv_path, config=cfg, device=dev)
    rec.create()
    sidecar_fh = open(sidecar_path, "w", newline="", encoding="utf-8")
    sidecar_writer = csv.writer(sidecar_fh)
    sidecar_writer.writerow(
        [
            "frame_idx",
            "save_timestamp_ns",
            "color_timestamp_usec",
            "depth_timestamp_usec",
            "depth_enabled",
        ]
    )

    record = {
        "device": dev,
        "record": rec,
        "mkv_path": mkv_path,
        "sidecar_path": sidecar_path,
        "sidecar_fh": sidecar_fh,
        "sidecar_writer": sidecar_writer,
        "frame_idx": 0,
    }

    try:
        print(f"Recording only from serial {dev.serial}... Press CTRL-C to stop.")
        while True:
            host_unix_before_get_ns = time.time_ns()
            cap = dev.get_capture(timeout=1000)
            host_unix_after_get_ns = time.time_ns()
            host_unix_mid_get_ns = (host_unix_before_get_ns + host_unix_after_get_ns) // 2

            record["sidecar_writer"].writerow(
                [
                    record["frame_idx"],
                    host_unix_mid_get_ns,
                    cap.color_timestamp_usec if cap.color is not None else "",
                    cap.depth_timestamp_usec if cap.depth is not None else "",
                    int(cap.depth is not None),
                ]
            )

            record["record"].write_capture(cap)
            record["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping master PC recording")
    finally:
        record["record"].flush()
        record["record"].close()
        record["sidecar_fh"].flush()
        record["sidecar_fh"].close()
        dev.stop()
        print(
            f"saved {record['mkv_path']} frames={record['record'].captures_count} sidecar={record['sidecar_path']}"
        )


if __name__ == "__main__":
    main()
