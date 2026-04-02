# Data Workflow Reference

## Data Capture

### 1) Main capture entrypoint
Run the unified capture launcher:

```bash
python scripts/capture.py
```

What it does:
- Reads `scripts/config/config.json`.
- Launches `scripts/record_kinect_final_with_capture.py` when `launch.run_kinect=true`.
- Launches `scripts/record_orbbec_final_with_capture.py` when `launch.run_orbbec=true`.
- Creates a session folder under `recordings/` by default.
- Stops automatically after configured duration, or manually with `Ctrl+C`.

### 2) Related capture scripts
- `scripts/record_kinect_final_with_capture.py`: records Kinect MKVs + per-frame timestamp CSV sidecars.
- `scripts/record_orbbec_final_with_capture.py`: records Orbbec BAGs + per-frame timestamp CSV sidecars.
- `scripts/config/config.json`: shared capture configuration used by `capture.py`.
- `scripts/config/multi_device_sync_config.json`: optional Orbbec serial-specific sync config reference.

### 3) Tuning parameters in `scripts/config/config.json`
Session control (`session`):
- `duration_seconds`: total capture duration.
- `recordings_root`: base output folder (default `recordings`).
- `session_prefix`: session folder prefix.
- `graceful_shutdown_seconds`: wait time after stop signal before forced termination.
- `stop_grace_seconds`: force-kill grace period.

Launch toggles (`launch`):
- `run_kinect`: enable/disable Kinect recorder.
- `run_orbbec`: enable/disable Orbbec recorder.

Kinect tuning (`kinect`):
- `devices[]`: per-device mapping (`device_id`, `name`, `mode`, `sub_delay_usec`).
- `master_base_config`: master camera settings (`color_format`, `color_resolution`, `depth_mode`, `camera_fps`, `synchronized_images_only`).
- `subordinate_base_config`: subordinate settings (typically depth off).

Orbbec tuning (`orbbec`):
- `max_devices`: expected Orbbec device count.
- `wfov_binned_depth_width`, `wfov_binned_depth_height`: required depth profile.
- `record_fps`: recording FPS.
- `multi_device_sync.devices[]`: per-serial sync mode and timing fields.

### 4) Input directory expected by processing
Processing scripts expect data under:

```text
data/person_x/session_x/data_collection/
```

Recommended layout:

```text
data/
└── person_x/
    └── session_x/
        ├── rgb_calibration/
        ├── depth_calibration/
        └── data_collection/
            ├── rgb_depth_data/
            │   ├── kinect_master.mkv
            │   ├── kinect_master.save_timestamps.csv
            │   ├── kinect_subordinate1.mkv
            │   ├── kinect_subordinate1.save_timestamps.csv
            │   ├── kinect_subordinate2.mkv
            │   ├── kinect_subordinate2.save_timestamps.csv
            │   ├── kinect_subordinate3.mkv
            │   ├── kinect_subordinate3.save_timestamps.csv
            │   ├── kinect_subordinate4.mkv
            │   ├── kinect_subordinate4.save_timestamps.csv
            │   ├── orbbec_master.bag
            │   ├── orbbec_master.save_timestamps.csv
            │   ├── orbbec_subordinate.bag
            │   └── orbbec_subordinate.save_timestamps.csv
            └── pressure_data/
                └── pressure_map_save_*.csv
```

Typical contents inside `data_collection/`:
- `rgb_depth_data/` (Kinect MKVs + sidecars, Orbbec BAGs + sidecars)
- `pressure_data/` (pressure CSV files)

## Data Processing

All commands below are run from repo root:

```bash
cd /home/rutwik/korus-ml-devel/devel/pyk4a
```

Use your target person/session in place of `person_1/session_1`.

### Run order
1. Export Kinect synced data.
2. Export Orbbec synced data.
3. Export Pressure synced data.
4. Export fused depth maps from Orbbec synced depth.
5. Build Kinect-Orbbec timestamp mapping (and optional plots).
6. Build Pressure-Kinect-Orbbec timestamp mapping (and optional plots).
7. Trim synced CSVs to overlap window.
8. Plot from trimmed CSVs.

### Commands
1. Kinect export:

```bash
python scripts/export_kinect_synced_by_device_ts_data.py data/person_1/session_1/data_collection
```

2. Orbbec export:

```bash
python scripts/export_orbbec_synced_by_device_ts_data.py data/person_1/session_1/data_collection
```

3. Pressure export:

```bash
python scripts/export_pressure_synced_by_device_ts_data.py data/person_1/session_1/data_collection
```

4. Fused depth export:

```bash
python scripts/export_fused_depth_maps.py \
  --calib_json outputs/person_1/session_1/depth_calibration/depth_calibration.json \
  --sync_csv outputs/person_1/session_1/orbbec/orbbec_synced_data_companion.csv \
  --cam1_depth_dir outputs/person_1/session_1/orbbec/orbbec_depth_master \
  --cam2_depth_dir outputs/person_1/session_1/orbbec/orbbec_depth_subordinate \
  --max_frames 0 \
  --save_npy
```

5. Kinect-Orbbec mapping (CSV only):

```bash
python scripts/plot_kinect_orbbec_from_exported_data.py outputs/person_1/session_1
```

6. Pressure-Kinect-Orbbec mapping (CSV only):

```bash
python scripts/plot_kinect_orbbec_pressure_from_exported_data.py outputs/person_1/session_1
```

If you also want plot images for steps 5 and 6, add `--plot`.

7. Trim both synced CSVs to their valid overlap windows:

```bash
python scripts/trim_synced_overlap_csvs.py \
  --pressure-rgb-depth-csv outputs/person_1/session_1/synced_data_from_pressure_kinect_orbbec/synced_data_from_pressure_kinect_orbbec.csv \
  --rgb-depth-csv outputs/person_1/session_1/synced_data_from_kinect_orbbec/synced_data_from_kinect_orbbec.csv
```

8. Plot from trimmed CSVs:

```bash
python scripts/plot_kinect_orbbec_from_trimmed_csv.py \
  --trimmed-csv outputs/person_1/session_1/person_1_session_1_rgb_depth.csv
```

```bash
python scripts/plot_kinect_orbbec_pressure_from_trimmed_csv.py \
  --trimmed-csv outputs/person_1/session_1/person_1_session_1_pressure_rgb_depth.csv
```

### Output layout
Exports and synced outputs are saved under:

```text
outputs/person_x/session_x/
```

Main subfolders:
- `kinect/`
- `orbbec/`
- `pressure/`
- `fused_depth_maps/`
- `synced_data_from_kinect_orbbec/`
- `synced_data_from_pressure_kinect_orbbec/`

Trimmed CSVs are written to:
- `outputs/person_x/session_x/person_x_session_x_rgb_depth.csv`
- `outputs/person_x/session_x/person_x_session_x_pressure_rgb_depth.csv`
