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

# Master PC local setup: 3 devices total (1 MASTER + 2 SUBORDINATES)
# Adjust device_id values if your local device ordering is different.
LOCAL_DEVICES = [
    {"device_id": 0, "name": "master", "mode": WiredSyncMode.MASTER, "sub_delay_usec": 0},
    {"device_id": 1, "name": "sub_local_1", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 200},
    {"device_id": 2, "name": "sub_local_2", "mode": WiredSyncMode.SUBORDINATE, "sub_delay_usec": 400},
]

OUT_DIR = Path("multi_mkv/master_pc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_720P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    camera_fps=FPS.FPS_30,
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
        path = OUT_DIR / f"{c['spec']['name']}_dev{c['spec']['device_id']}_{serial}.mkv"
        sidecar_path = path.with_suffix(".timestamps.csv")
        rec = PyK4ARecord(path=path, config=c["config"], device=c["device"])
        rec.create()
        sidecar_fh = open(sidecar_path, "w", newline="")
        sidecar_writer = csv.writer(sidecar_fh)
        sidecar_writer.writerow(
            [
                "frame_idx",
                "device_id",
                "serial",
                "name",
                "mode",
                "sub_delay_usec",
                "host_unix_before_get_ns",
                "host_unix_after_get_ns",
                "host_unix_mid_get_ns",
                "host_monotonic_after_get_ns",
                "depth_timestamp_usec",
                "depth_system_timestamp_nsec",
                "color_timestamp_usec",
                "color_system_timestamp_nsec",
            ]
        )
        records.append(
            {
                "camera": c,
                "record": rec,
                "path": path,
                "serial": serial,
                "sidecar_path": sidecar_path,
                "sidecar_fh": sidecar_fh,
                "sidecar_writer": sidecar_writer,
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
                host_monotonic_after_get_ns = time.perf_counter_ns()
                captures.append(
                    (
                        r,
                        cap,
                        host_unix_before_get_ns,
                        host_unix_after_get_ns,
                        host_unix_mid_get_ns,
                        host_monotonic_after_get_ns,
                    )
                )

            ts = [cap.depth_timestamp_usec for _, cap, _, _, _, _ in captures]
            skew_usec = max(ts) - min(ts)
            if skew_usec > 1000:
                print(f"local timestamp skew={skew_usec} usec")

            for r, cap, host_before, host_after, host_mid, host_mono_after in captures:
                depth_ts_usec = cap.depth_timestamp_usec
                depth_sys_ts_ns = cap.depth_system_timestamp_nsec
                color_ts_usec = cap.color_timestamp_usec
                color_sys_ts_ns = cap.color_system_timestamp_nsec

                r["sidecar_writer"].writerow(
                    [
                        r["frame_idx"],
                        r["camera"]["spec"]["device_id"],
                        r["serial"],
                        r["camera"]["spec"]["name"],
                        r["camera"]["spec"]["mode"].name,
                        r["camera"]["spec"]["sub_delay_usec"],
                        host_before,
                        host_after,
                        host_mid,
                        host_mono_after,
                        depth_ts_usec,
                        depth_sys_ts_ns,
                        color_ts_usec,
                        color_sys_ts_ns,
                    ]
                )
                r["record"].write_capture(cap)
                r["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping master PC recording")
    finally:
        for r in records:
            r["record"].flush()
            r["record"].close()
            r["sidecar_fh"].flush()
            r["sidecar_fh"].close()
            r["camera"]["device"].stop()
            print(f"saved {r['path']} frames={r['record'].captures_count} sidecar={r['sidecar_path']}")


if __name__ == "__main__":
    main()
