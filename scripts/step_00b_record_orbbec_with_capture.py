import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import List

from pyorbbecsdk import (
    Context,
    Device,
    FrameSet,
    OBError,
    OBFormat,
    OBSensorType,
    OBMultiDeviceSyncMode,
    Pipeline,
    VideoStreamProfile,
    Config,
    RecordDevice,
)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set")
    return value


def load_shared_orbbec_section() -> dict:
    config_path = Path(require_env("RECORD_MASTER_CONFIG")).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    orbbec = root.get("orbbec")
    if not isinstance(orbbec, dict):
        raise RuntimeError(f'{config_path} must contain a top-level "orbbec" object')
    return orbbec


def require_orbbec_keys(orbbec: dict, config_path: str) -> None:
    required = (
        "max_devices",
        "wfov_binned_depth_width",
        "wfov_binned_depth_height",
        "record_fps",
        "multi_device_sync",
    )
    for key in required:
        if key not in orbbec:
            raise RuntimeError(f'orbbec.{key} is required in {config_path}')
    mds = orbbec["multi_device_sync"]
    if not isinstance(mds, dict):
        raise RuntimeError(f'orbbec.multi_device_sync must be an object in {config_path}')


MAX_DEVICES = 0
WFOV_BINNED_DEPTH_WIDTH = 0
WFOV_BINNED_DEPTH_HEIGHT = 0
RECORD_FPS = 0

multi_device_sync_config: dict = {}
recorders: List[RecordDevice | None] = []
timestamp_files: List = []
timestamp_writers: List = []
timestamp_locks: List[threading.Lock] = []
frame_indices: List[int] = []
stop_file_path: str | None = None


def apply_runtime_from_config(orbbec: dict) -> None:
    global MAX_DEVICES, WFOV_BINNED_DEPTH_WIDTH, WFOV_BINNED_DEPTH_HEIGHT
    global RECORD_FPS, recorders, timestamp_files, timestamp_writers
    global timestamp_locks, frame_indices, stop_file_path

    MAX_DEVICES = int(orbbec["max_devices"])
    WFOV_BINNED_DEPTH_WIDTH = int(orbbec["wfov_binned_depth_width"])
    WFOV_BINNED_DEPTH_HEIGHT = int(orbbec["wfov_binned_depth_height"])
    RECORD_FPS = int(orbbec["record_fps"])

    recorders = [None for _ in range(MAX_DEVICES)]
    timestamp_files = [None for _ in range(MAX_DEVICES)]
    timestamp_writers = [None for _ in range(MAX_DEVICES)]
    timestamp_locks = [threading.Lock() for _ in range(MAX_DEVICES)]
    frame_indices = [0 for _ in range(MAX_DEVICES)]
    stop_file_path = os.environ.get("RECORD_STOP_FILE")


def sync_mode_from_str(sync_mode_str: str) -> OBMultiDeviceSyncMode:
    sync_mode_str = sync_mode_str.upper()
    if sync_mode_str == "FREE_RUN":
        return OBMultiDeviceSyncMode.FREE_RUN
    if sync_mode_str == "STANDALONE":
        return OBMultiDeviceSyncMode.STANDALONE
    if sync_mode_str == "PRIMARY":
        return OBMultiDeviceSyncMode.PRIMARY
    if sync_mode_str == "SECONDARY":
        return OBMultiDeviceSyncMode.SECONDARY
    if sync_mode_str == "SECONDARY_SYNCED":
        return OBMultiDeviceSyncMode.SECONDARY_SYNCED
    if sync_mode_str == "SOFTWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
    if sync_mode_str == "HARDWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.HARDWARE_TRIGGERING
    raise ValueError(f"Invalid sync mode: {sync_mode_str}")


def read_multi_device_sync(orbbec: dict) -> None:
    global multi_device_sync_config
    multi_device_sync_config = {}
    mds = orbbec["multi_device_sync"]
    for device in mds.get("devices", []):
        multi_device_sync_config[device["serial_number"]] = device


def default_sync_config_for_index(index: int):
    mode = "PRIMARY" if index == 0 else "SECONDARY"
    return {
        "serial_number": "",
        "config": {
            "mode": mode,
            "depth_delay_us": 0,
            "color_delay_us": 0,
            "trigger_to_image_delay_us": 0,
            "trigger_out_enable": index == 0,
            "trigger_out_delay_us": 0,
            "frames_per_trigger": 1,
        },
    }


def make_recording_dir() -> str:
    require_env("RECORD_SESSION_TIMESTAMP")
    output_dir = Path(require_env("RECORD_SESSION_DIR")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def stop_recorders():
    global recorders
    for i in range(len(recorders)):
        recorders[i] = None


def stop_timestamp_files():
    global timestamp_files, timestamp_writers, frame_indices
    for i in range(len(timestamp_files)):
        if timestamp_files[i] is not None:
            timestamp_files[i].flush()
            timestamp_files[i].close()
        timestamp_files[i] = None
        timestamp_writers[i] = None
        frame_indices[i] = 0


def select_wfov_binned_depth_profile(pipeline: Pipeline) -> VideoStreamProfile:
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    return depth_profiles.get_video_stream_profile(
        WFOV_BINNED_DEPTH_WIDTH,
        WFOV_BINNED_DEPTH_HEIGHT,
        OBFormat.Y16,
        RECORD_FPS,
    )


def enable_default_streams(pipeline: Pipeline, config: Config, serial_number: str):
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        default_color_profile: VideoStreamProfile = (
            color_profiles.get_default_video_stream_profile()
        )
        color_profile = color_profiles.get_video_stream_profile(
            default_color_profile.get_width(),
            default_color_profile.get_height(),
            default_color_profile.get_format(),
            RECORD_FPS,
        )
        config.enable_stream(color_profile)
        print(
            f"{serial_number}: color "
            f"{color_profile.get_width()}x{color_profile.get_height()} "
            f"{color_profile.get_format()} @{color_profile.get_fps()}fps"
        )
    except OBError:
        pass

    try:
        depth_profile = select_wfov_binned_depth_profile(pipeline)
        print(
            f"{serial_number}: using WFOV binned depth "
            f"{depth_profile.get_width()}x{depth_profile.get_height()} "
            f"{depth_profile.get_format()} @{depth_profile.get_fps()}fps"
        )
    except OBError:
        raise RuntimeError(
            f"{serial_number}: required depth profile "
            f"{WFOV_BINNED_DEPTH_WIDTH}x{WFOV_BINNED_DEPTH_HEIGHT} "
            f"{OBFormat.Y16} @{RECORD_FPS}fps is unavailable"
        )
    config.enable_stream(depth_profile)


def configure_device(device: Device, index: int):
    serial_number = device.get_device_info().get_serial_number()
    sync_config_json = multi_device_sync_config.get(serial_number)
    if sync_config_json is None:
        sync_config_json = default_sync_config_for_index(index)
        sync_config_json["serial_number"] = serial_number

    sync_config = device.get_multi_device_sync_config()
    sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
    sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
    sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
    sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
    sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
    sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
    device.set_multi_device_sync_config(sync_config)
    role_name = "master" if sync_config.mode == OBMultiDeviceSyncMode.PRIMARY else "subordinate"
    return serial_number, role_name


def on_new_frame_callback(frames: FrameSet, index: int):
    if frames is None or timestamp_writers[index] is None:
        return

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is None and depth_frame is None:
        return

    host_callback_start_ns = time.time_ns()
    save_timestamp_ns = time.time_ns()
    color_timestamp_us = color_frame.get_timestamp_us() if color_frame is not None else ""
    color_system_timestamp_us = (
        color_frame.get_system_timestamp_us() if color_frame is not None else ""
    )
    depth_timestamp_us = depth_frame.get_timestamp_us() if depth_frame is not None else ""
    depth_system_timestamp_us = (
        depth_frame.get_system_timestamp_us() if depth_frame is not None else ""
    )
    with timestamp_locks[index]:
        timestamp_writers[index].writerow(
            [
                frame_indices[index],
                host_callback_start_ns,
                save_timestamp_ns,
                color_timestamp_us,
                color_system_timestamp_us,
                depth_timestamp_us,
                depth_system_timestamp_us,
                int(color_frame is not None),
                int(depth_frame is not None),
            ]
        )
        frame_indices[index] += 1


def main():
    config_path = str(Path(require_env("RECORD_MASTER_CONFIG")).expanduser().resolve())
    orbbec = load_shared_orbbec_section()
    require_orbbec_keys(orbbec, config_path)
    apply_runtime_from_config(orbbec)
    read_multi_device_sync(orbbec)
    recording_dir = make_recording_dir()

    ctx = Context()
    device_list = ctx.query_devices()
    device_count = device_list.get_count()

    if device_count == 0:
        print("No device connected")
        return
    if device_count > MAX_DEVICES:
        print("Too many devices connected")
        return

    pipelines: List[Pipeline] = []
    configs: List[Config] = []

    for i in range(device_count):
        device = device_list.get_device_by_index(i)
        serial_number, role_name = configure_device(device, i)

        pipeline = Pipeline(device)
        config = Config()
        enable_default_streams(pipeline, config, serial_number)

        bag_path = os.path.join(recording_dir, f"orbbec_{role_name}.bag")
        recorders[i] = RecordDevice(device, bag_path)
        csv_path = os.path.splitext(bag_path)[0] + ".save_timestamps.csv"
        timestamp_files[i] = open(csv_path, "w", newline="", encoding="utf-8")
        timestamp_writers[i] = csv.writer(timestamp_files[i])
        timestamp_writers[i].writerow(
            [
                "frame_idx",
                "host_callback_start_ns",
                "save_timestamp_ns",
                "color_timestamp_usec",
                "color_system_timestamp_usec",
                "depth_timestamp_usec",
                "depth_system_timestamp_usec",
                "color_present",
                "depth_enabled",
            ]
        )

        pipelines.append(pipeline)
        configs.append(config)

    index = 0
    for pipeline, config in zip(pipelines, configs):
        pipeline.start(
            config,
            lambda frame_set, curr_index=index: on_new_frame_callback(
                frame_set, curr_index
            ),
        )
        try:
            pipeline.enable_frame_sync()
        except OBError:
            pass
        index += 1

    ctx.enable_multi_device_sync(60000)

    print("Recording. Press Ctrl+C to stop.")
    try:
        while True:
            if stop_file_path and os.path.exists(stop_file_path):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        stop_timestamp_files()
        stop_recorders()


if __name__ == "__main__":
    main()
