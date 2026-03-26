"""
Run Kinect and Orbbec recording scripts in parallel and force them to share
the same session folder and timestamp.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

KINECT_SCRIPT = SCRIPT_DIR / "record_kinect.py"
ORBBEC_SCRIPT = SCRIPT_DIR / "record_orbbec.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Kinect and Orbbec recorders in parallel."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Recording duration in seconds before automatic shutdown. Default: 15",
    )
    return parser.parse_args()


def make_session_env():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = ROOT_DIR / "recordings" / f"session_{timestamp_str}"
    session_dir.mkdir(parents=True, exist_ok=True)
    stop_file = session_dir / ".stop"

    env = os.environ.copy()
    env["RECORD_SESSION_TIMESTAMP"] = timestamp_str
    env["RECORD_SESSION_DIR"] = str(session_dir)
    env["RECORD_STOP_FILE"] = str(stop_file)
    return env, session_dir, stop_file


def terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    except Exception:
        proc.terminate()


def force_stop_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def main():
    args = parse_args()
    env, session_dir, stop_file = make_session_env()
    print(f"Session directory: {session_dir}", flush=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    processes = [
        subprocess.Popen(
            [sys.executable, "-u", str(KINECT_SCRIPT)],
            cwd=str(ROOT_DIR),
            env=env,
            creationflags=creationflags,
        ),
        subprocess.Popen(
            [sys.executable, "-u", str(ORBBEC_SCRIPT)],
            cwd=str(ROOT_DIR),
            env=env,
            creationflags=creationflags,
        ),
    ]

    try:
        deadline = time.time() + args.duration
        while True:
            if any(proc.poll() is not None for proc in processes):
                exit_codes = [proc.wait() for proc in processes]
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                print(
                    f"\nDuration reached ({args.duration:g}s). Stopping child recorders...",
                    flush=True,
                )
                stop_file.touch()
                for proc in processes:
                    terminate_process(proc)
                exit_codes = []
                for proc in processes:
                    try:
                        exit_codes.append(proc.wait(timeout=8))
                    except subprocess.TimeoutExpired:
                        force_stop_process(proc)
                        exit_codes.append(proc.wait())
                break
            time.sleep(min(0.5, remaining))
    except KeyboardInterrupt:
        print("\nStopping child recorders...", flush=True)
        stop_file.touch()
        for proc in processes:
            terminate_process(proc)
        exit_codes = []
        for proc in processes:
            try:
                exit_codes.append(proc.wait(timeout=8))
            except subprocess.TimeoutExpired:
                force_stop_process(proc)
                exit_codes.append(proc.wait())

    if any(code != 0 for code in exit_codes):
        raise SystemExit(max(code for code in exit_codes if code is not None))


if __name__ == "__main__":
    main()
