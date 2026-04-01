# multi_record.py
from pathlib import Path
from pyk4a import (
    PyK4A, PyK4ARecord, connected_device_count,
    Config, FPS, ImageFormat, ColorResolution, DepthMode, WiredSyncMode
)

OUT_DIR = Path("multi_mkv")
OUT_DIR.mkdir(exist_ok=True)

master_id = 0
delay_step_usec = 200  # common starting value for subordinates

# Same imaging settings on all cameras
base_kwargs = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_720P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    camera_fps=FPS.FPS_30,
    synchronized_images_only=True,
)

n = connected_device_count()
if n < 2:
    raise RuntimeError(f"Need >=2 devices, found {n}")

devices = []
for device_id in range(n):
    if device_id == master_id:
        cfg = Config(**base_kwargs, wired_sync_mode=WiredSyncMode.MASTER)
    else:
        sub_idx = len([d for d in devices if d["mode"] == "sub"]) + 1
        cfg = Config(
            **base_kwargs,
            wired_sync_mode=WiredSyncMode.SUBORDINATE,
            subordinate_delay_off_master_usec=sub_idx * delay_step_usec,
        )
    dev = PyK4A(config=cfg, device_id=device_id)
    devices.append({"id": device_id, "dev": dev, "cfg": cfg, "mode": "master" if device_id == master_id else "sub"})

# Start subordinates first, master last
for d in [x for x in devices if x["mode"] == "sub"] + [x for x in devices if x["mode"] == "master"]:
    d["dev"].start()
    print(f"started device {d['id']} serial={d['dev'].serial} mode={d['mode']} sync_jack={d['dev'].sync_jack_status}")

records = []
for d in devices:
    path = OUT_DIR / f"device{d['id']}_{d['dev'].serial}.mkv"
    rec = PyK4ARecord(path=path, config=d["cfg"], device=d["dev"])
    rec.create()
    records.append({"rec": rec, "dev": d["dev"], "id": d["id"], "path": path})

try:
    print("Recording... Ctrl-C to stop")
    while True:
        captures = []
        for r in records:
            cap = r["dev"].get_capture(timeout=1000)
            captures.append((r, cap))

        # Optional sync sanity check
        ts = [cap.depth_timestamp_usec for _, cap in captures]
        skew = max(ts) - min(ts)
        if skew > 1000:  # 1 ms
            print(f"timestamp skew: {skew} usec")

        for r, cap in captures:
            r["rec"].write_capture(cap)

except KeyboardInterrupt:
    pass
finally:
    for r in records:
        r["rec"].flush()
        r["rec"].close()
        r["dev"].stop()
        print(f"saved {r['path']} frames={r['rec'].captures_count}")
