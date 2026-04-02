import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_session(config: dict, repo_root: Path) -> tuple[str, Path, int, int, int]:
    session_cfg = config.get("session", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_prefix = session_cfg.get("session_prefix", "session")
    recordings_root = session_cfg.get("recordings_root", "recordings")
    output_dir = repo_root / recordings_root / f"{session_prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_seconds = int(session_cfg.get("duration_seconds", 300))
    graceful_shutdown_seconds = int(session_cfg.get("graceful_shutdown_seconds", 90))
    stop_grace_seconds = int(session_cfg.get("stop_grace_seconds", 10))
    return (
        timestamp,
        output_dir,
        duration_seconds,
        graceful_shutdown_seconds,
        stop_grace_seconds,
    )


def request_graceful_stop(stop_file: Path) -> None:
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("", encoding="utf-8")


def wait_for_processes(processes: list[subprocess.Popen], timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(p.poll() is not None for p in processes):
            return
        time.sleep(0.2)


def terminate_processes(processes: list[subprocess.Popen], grace_seconds: int) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + grace_seconds
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            process.kill()


def graceful_then_force_stop(
    processes: list[subprocess.Popen],
    stop_file: Path,
    graceful_wait_seconds: float,
    kill_grace_seconds: int,
) -> None:
    if all(p.poll() is not None for p in processes):
        return
    print("Requesting graceful shutdown (stop file)...")
    request_graceful_stop(stop_file)
    wait_for_processes(processes, graceful_wait_seconds)
    if any(p.poll() is None for p in processes):
        print("Some recorders still running; sending terminate/kill.")
        terminate_processes(processes, kill_grace_seconds)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_path = script_dir / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    (
        timestamp,
        output_dir,
        duration_seconds,
        graceful_shutdown_seconds,
        stop_grace_seconds,
    ) = build_session(config, repo_root)

    stop_file = output_dir / ".stop_recording"

    env = os.environ.copy()
    env["RECORD_MASTER_CONFIG"] = str(config_path)
    env["RECORD_SESSION_TIMESTAMP"] = timestamp
    env["RECORD_SESSION_DIR"] = str(output_dir)
    env["RECORD_STOP_FILE"] = str(stop_file.resolve())

    launch_cfg = config.get("launch", {})
    run_kinect = bool(launch_cfg.get("run_kinect", True))
    run_orbbec = bool(launch_cfg.get("run_orbbec", True))

    processes: list[subprocess.Popen] = []
    if run_kinect:
        processes.append(
            subprocess.Popen(
                [sys.executable, str(script_dir / "step_00a_record_kinect_with_capture.py")],
                env=env,
            )
        )
    if run_orbbec:
        processes.append(
            subprocess.Popen(
                [sys.executable, str(script_dir / "step_00b_record_orbbec_with_capture.py")],
                env=env,
            )
        )
    if not processes:
        raise RuntimeError("No recorder enabled. Set launch.run_kinect or launch.run_orbbec to true.")

    print(f"Recording started. Session dir: {output_dir}")
    print(f"Configured duration: {duration_seconds} seconds")

    start_time = time.time()
    stop_reason = "timeout"
    try:
        while True:
            if any(process.poll() is not None for process in processes):
                stop_reason = "a child process exited"
                break
            if time.time() - start_time >= duration_seconds:
                stop_reason = "session duration elapsed"
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_reason = "manual Ctrl+C"

    print(f"Stopping recorders: {stop_reason}")
    graceful_then_force_stop(
        processes,
        stop_file,
        float(graceful_shutdown_seconds),
        stop_grace_seconds,
    )

    exit_codes = [process.poll() for process in processes]
    print(f"Recorder exit codes: {exit_codes}")


if __name__ == "__main__":
    main()
