"""
Kinect capture worker. Launched by capture.py with RECORD_MASTER_CONFIG and session env set.
"""

import csv
import json
import os
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


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set")
    return value


def load_kinect_config() -> dict:
    config_path = Path(require_env("RECORD_MASTER_CONFIG")).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    kinect = root.get("kinect")
    if not isinstance(kinect, dict):
        raise RuntimeError(f'{config_path} must contain a top-level "kinect" object')
    return kinect


def parse_enum(enum_type, value: str):
    if not isinstance(value, str):
        return value
    return getattr(enum_type, value.upper())


def build_runtime_settings(kinect: dict):
    if "devices" not in kinect or not kinect["devices"]:
        raise RuntimeError('kinect.devices must be a non-empty list in RECORD_MASTER_CONFIG')
    if "master_base_config" not in kinect:
        raise RuntimeError('kinect.master_base_config is required in RECORD_MASTER_CONFIG')
    if "subordinate_base_config" not in kinect:
        raise RuntimeError('kinect.subordinate_base_config is required in RECORD_MASTER_CONFIG')

    normalized_devices = []
    for device in kinect["devices"]:
        normalized_devices.append(
            {
                "device_id": int(device["device_id"]),
                "name": str(device["name"]),
                "mode": parse_enum(WiredSyncMode, device["mode"]),
                "sub_delay_usec": int(device.get("sub_delay_usec", 0)),
            }
        )

    master_base = {}
    for key, value in kinect["master_base_config"].items():
        if key == "color_format":
            master_base[key] = parse_enum(ImageFormat, value)
        elif key == "color_resolution":
            master_base[key] = parse_enum(ColorResolution, value)
        elif key == "depth_mode":
            master_base[key] = parse_enum(DepthMode, value)
        elif key == "camera_fps":
            master_base[key] = parse_enum(FPS, value)
        else:
            master_base[key] = value

    subordinate_base = {}
    for key, value in kinect["subordinate_base_config"].items():
        if key == "color_format":
            subordinate_base[key] = parse_enum(ImageFormat, value)
        elif key == "color_resolution":
            subordinate_base[key] = parse_enum(ColorResolution, value)
        elif key == "depth_mode":
            subordinate_base[key] = parse_enum(DepthMode, value)
        elif key == "camera_fps":
            subordinate_base[key] = parse_enum(FPS, value)
        else:
            subordinate_base[key] = value

    return normalized_devices, master_base, subordinate_base


def build_config(
    mode: WiredSyncMode,
    sub_delay_usec: int,
    master_base_config: dict,
    subordinate_base_config: dict,
) -> Config:
    if mode == WiredSyncMode.MASTER:
        return Config(**master_base_config, wired_sync_mode=WiredSyncMode.MASTER)
    return Config(
        **subordinate_base_config,
        wired_sync_mode=WiredSyncMode.SUBORDINATE,
        subordinate_delay_off_master_usec=sub_delay_usec,
    )


def make_recording_dir() -> Path:
    require_env("RECORD_SESSION_TIMESTAMP")
    output_dir = Path(require_env("RECORD_SESSION_DIR")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    kinect = load_kinect_config()
    local_devices, master_base_config, subordinate_base_config = build_runtime_settings(kinect)
    out_dir = make_recording_dir()
    stop_file_path = os.environ.get("RECORD_STOP_FILE")
    available = connected_device_count()
    if available < len(local_devices):
        raise RuntimeError(f"Expected at least {len(local_devices)} devices, found {available}")

    cameras = []
    for spec in local_devices:
        cfg = build_config(
            spec["mode"],
            spec["sub_delay_usec"],
            master_base_config,
            subordinate_base_config,
        )
        dev = PyK4A(config=cfg, device_id=spec["device_id"])
        cameras.append({"spec": spec, "device": dev, "config": cfg})

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
        mkv_path = out_dir / f"kinect_{normalized_name}.mkv"
        ts_csv_path = mkv_path.with_suffix(".save_timestamps.csv")

        record = PyK4ARecord(path=mkv_path, config=c["config"], device=c["device"])
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
            stop_after_round = False
            for out in outputs:
                if stop_file_path and os.path.exists(stop_file_path):
                    stop_after_round = True
                    break
                spec = out["camera"]["spec"]
                host_capture_start_ns = time.time_ns()
                cap = out["camera"]["device"].get_capture(timeout=10000)

                if cap.color is None:
                    print(f"skipping frame on device {spec['device_id']} because color is missing")
                    continue

                is_master = spec["mode"] == WiredSyncMode.MASTER
                if is_master and cap.depth is None:
                    print(f"skipping frame on device {spec['device_id']} because depth is missing on master")
                    continue
                has_depth = is_master and cap.depth is not None

                save_timestamp_ns = time.time_ns()
                color_ts_usec = cap.color_timestamp_usec if cap.color is not None else ""
                color_system_ts_ns = (
                    cap.color_system_timestamp_nsec if cap.color is not None else ""
                )
                depth_ts_usec = cap.depth_timestamp_usec if has_depth else ""
                depth_system_ts_ns = (
                    cap.depth_system_timestamp_nsec if has_depth else ""
                )
                out["record"].write_capture(cap)
                host_write_complete_ns = time.time_ns()
                out["ts_writer"].writerow(
                    [
                        out["frame_idx"],
                        host_capture_start_ns,
                        save_timestamp_ns,
                        host_write_complete_ns,
                        color_ts_usec,
                        color_system_ts_ns,
                        depth_ts_usec,
                        depth_system_ts_ns,
                        int(cap.color is not None),
                        int(has_depth),
                    ]
                )
                out["frame_idx"] += 1
            if stop_after_round:
                break
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
