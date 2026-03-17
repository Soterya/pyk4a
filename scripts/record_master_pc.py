import json
import time
from pathlib import Path
from typing import Dict, TextIO

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

# Master PC setup: 3 devices total (1 MASTER + 2 SUBORDINATES)
# Adjust device_id values if local ordering is different.
LOCAL_DEVICES = [
    {"device_id": 0, "name": "master", "mode": WiredSyncMode.MASTER, "sub_delay_usec": 0},
    {"device_id": 1, "name": "sub_local_1", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 200},
    {"device_id": 2, "name": "sub_local_2", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 400},
]

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


def open_sidecar_json(path: Path, metadata: Dict) -> TextIO:
    fh = open(path, "w", encoding="utf-8")
    fh.write("{\"metadata\":")
    json.dump(metadata, fh, separators=(",", ":"))
    fh.write(",\"frames\":[")
    return fh


def append_json_frame(fh: TextIO, frame_data: Dict, is_first: bool) -> bool:
    if not is_first:
        fh.write(",")
    json.dump(frame_data, fh, separators=(",", ":"))
    return False


def close_sidecar_json(fh: TextIO) -> None:
    fh.write("]}\n")
    fh.flush()
    fh.close()


def main() -> None:
    available = connected_device_count()
    if available < len(LOCAL_DEVICES):
        raise RuntimeError(f"Expected at least {len(LOCAL_DEVICES)} devices on master PC, found {available}")

    cameras = []
    for spec in LOCAL_DEVICES:
        cfg = build_config(spec["mode"], spec["sub_delay_usec"])
        dev = PyK4A(config=cfg, device_id=spec["device_id"])
        cameras.append({"spec": spec, "device": dev, "config": cfg})

    # Start local subordinates first, master last.
    start_order = [c for c in cameras if c["spec"]["mode"] == WiredSyncMode.SUBORDINATE]
    start_order += [c for c in cameras if c["spec"]["mode"] == WiredSyncMode.MASTER]

    for c in start_order:
        c["device"].start()
        print(
            f"started device_id={c['spec']['device_id']} serial={c['device'].serial} "
            f"name={c['spec']['name']} mode={c['spec']['mode'].name} "
            f"sub_delay_usec={c['spec']['sub_delay_usec']} sync_jack={c['device'].sync_jack_status}"
        )

    records = []
    for c in cameras:
        serial = c["device"].serial
        mkv_path = OUT_DIR / f"{c['spec']['name']}_dev{c['spec']['device_id']}_{serial}.mkv"
        sidecar_path = mkv_path.with_suffix(".timestamps.json")
        rec = PyK4ARecord(path=mkv_path, config=c["config"], device=c["device"])
        rec.create()

        metadata = {
            "name": c["spec"]["name"],
            "device_id": c["spec"]["device_id"],
            "serial": serial,
            "wired_sync_mode": c["spec"]["mode"].name,
            "subordinate_delay_off_master_usec": c["spec"]["sub_delay_usec"],
            "record_path": str(mkv_path),
            "started_unix_ns": time.time_ns(),
        }
        sidecar_fh = open_sidecar_json(sidecar_path, metadata)

        records.append(
            {
                "camera": c,
                "serial": serial,
                "record": rec,
                "mkv_path": mkv_path,
                "sidecar_path": sidecar_path,
                "sidecar_fh": sidecar_fh,
                "sidecar_first_frame": True,
                "frame_idx": 0,
            }
        )

    try:
        print("Recording on master PC... Press CTRL-C to stop.")
        while True:
            captures = []
            for r in records:
                host_unix_before_get_ns = time.time_ns()
                cap = r["camera"]["device"].get_capture(timeout=1000)
                host_unix_after_get_ns = time.time_ns()
                host_unix_mid_get_ns = (host_unix_before_get_ns + host_unix_after_get_ns) // 2
                captures.append(
                    (
                        r,
                        cap,
                        host_unix_before_get_ns,
                        host_unix_after_get_ns,
                        host_unix_mid_get_ns,
                    )
                )

            ts = [cap.depth_timestamp_usec for _, cap, _, _, _ in captures]
            skew_usec = max(ts) - min(ts)
            if skew_usec > 1000:
                print(f"local timestamp skew={skew_usec} usec")

            for r, cap, host_before, host_after, host_mid in captures:
                depth_ts_usec = cap.depth_timestamp_usec

                frame_data = {
                    "frame_idx": r["frame_idx"],
                    "host_unix_before_get_ns": host_before,
                    "host_unix_after_get_ns": host_after,
                    "host_unix_mid_get_ns": host_mid,
                    "depth_timestamp_usec": depth_ts_usec,
                }
                r["sidecar_first_frame"] = append_json_frame(
                    r["sidecar_fh"], frame_data, r["sidecar_first_frame"]
                )

                r["record"].write_capture(cap)
                r["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping master PC recording")
    finally:
        for r in records:
            r["record"].flush()
            r["record"].close()
            close_sidecar_json(r["sidecar_fh"])
            r["camera"]["device"].stop()
            print(
                f"saved {r['mkv_path']} frames={r['record'].captures_count} sidecar={r['sidecar_path']}"
            )


if __name__ == "__main__":
    main()
