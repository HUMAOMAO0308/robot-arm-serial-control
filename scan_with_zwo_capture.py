from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from PIL import Image
import serial
import numpy as np

from main import MotionConfig, joint3_sequence, send_command
from zwo_camera import ZwoCamera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep robot joint 3 and capture one ZWO frame during each pause."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port, for example /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial timeout in seconds")
    parser.add_argument("--home-wait", type=float, default=5.0, help="Wait time after !START")
    parser.add_argument(
        "--settle-wait",
        type=float,
        default=3.0,
        help="Wait time after moving back to the initial pose",
    )
    parser.add_argument(
        "--pause-time",
        type=float,
        default=1.0,
        help="Total pause time per step, including image capture time",
    )
    parser.add_argument(
        "--capture-delay",
        type=float,
        default=0.3,
        help="Delay after each move before grabbing an image",
    )
    parser.add_argument("--total-steps", type=int, default=200, help="Number of sweep steps")
    parser.add_argument(
        "--step-angle",
        type=float,
        default=0.1,
        help="Angle delta applied to joint 3 at each step",
    )
    parser.add_argument("--j1", type=float, default=0.0, help="Initial joint 1 angle")
    parser.add_argument("--j2", type=float, default=-64.0, help="Initial joint 2 angle")
    parser.add_argument("--j3", type=float, default=155.0, help="Initial joint 3 angle")
    parser.add_argument("--j4", type=float, default=-1.0, help="Initial joint 4 angle")
    parser.add_argument("--j5", type=float, default=0.0, help="Initial joint 5 angle")
    parser.add_argument("--j6", type=float, default=0.0, help="Initial joint 6 angle")
    parser.add_argument("--speed", type=int, default=50, help="Motion speed")
    parser.add_argument(
        "--disable",
        dest="disable_on_exit",
        action="store_true",
        default=False,
        help="Send !DISABLE before closing the serial port",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="ZWO camera index to open.",
    )
    parser.add_argument(
        "--exposure-us",
        type=int,
        default=20000,
        help="Exposure in microseconds for each captured frame.",
    )
    parser.add_argument("--gain", type=int, default=50, help="Gain value for frame capture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures"),
        help="Directory where per-step images are saved.",
    )
    args = parser.parse_args()

    if args.total_steps <= 0:
        parser.error("--total-steps must be greater than 0")
    if args.step_angle <= 0:
        parser.error("--step-angle must be greater than 0")
    if args.pause_time < 0 or args.home_wait < 0 or args.settle_wait < 0 or args.capture_delay < 0:
        parser.error("wait values must be non-negative")
    if args.speed <= 0:
        parser.error("--speed must be greater than 0")
    if args.exposure_us <= 0:
        parser.error("--exposure-us must be greater than 0")

    motion = MotionConfig(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        home_wait=args.home_wait,
        settle_wait=args.settle_wait,
        pause_time=args.pause_time,
        total_steps=args.total_steps,
        step_angle=args.step_angle,
        initial_j1=args.j1,
        initial_j2=args.j2,
        initial_j3=args.j3,
        initial_j4=args.j4,
        initial_j5=args.j5,
        initial_j6=args.j6,
        speed=args.speed,
        disable_on_exit=args.disable_on_exit,
    )
    return args, motion


class ZwoCaptureSession:
    def __init__(
        self,
        camera_index: int,
        exposure_us: int,
        gain: int,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        self.camera_index = camera_index
        self.exposure_us = exposure_us
        self.gain = gain
        self.width = width
        self.height = height
        self.output_dir = output_dir.resolve()
        self.cam: ZwoCamera | None = None

    def open(self) -> None:
        self.cam = ZwoCamera()
        self.cam.open(self.camera_index, self.width, self.height, self.exposure_us, self.gain)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_step(self, step: int, joint3_angle: float) -> Path:
        if self.cam is None:
            raise RuntimeError("Camera session is not open.")
        frame = self.cam.grab_frame()
        image_path = self.output_dir / f"step_{step:04d}_j3_{joint3_angle:06.1f}.png"
        Image.fromarray(frame).save(image_path)
        return image_path

    def close(self) -> None:
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None


def iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def append_capture_log(csv_path: Path, record: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "timestamp",
        "step",
        "total_steps",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "speed",
        "pause_time",
        "capture_delay",
        "image_path",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def write_summary_json(json_path: Path, payload: dict[str, object]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_scan_with_capture(args, motion: MotionConfig) -> int:
    ser: serial.Serial | None = None
    camera_session = ZwoCaptureSession(
        camera_index=args.camera_index,
        exposure_us=args.exposure_us,
        gain=args.gain,
        width=640,
        height=480,
        output_dir=args.output_dir,
    )
    log_csv_path = args.output_dir / "captures_log.csv"
    summary_json_path = args.output_dir / "captures_summary.json"
    capture_records: list[dict[str, object]] = []

    try:
        camera_session.open()
        print(f"[OK] ZWO camera ready. Images saved to: {camera_session.output_dir}")

        ser = serial.Serial(motion.port, motion.baudrate, timeout=motion.timeout)
        print(f"[OK] Serial connected: {motion.port} @ {motion.baudrate}")

        send_command(ser, b"!START\n", "Home")
        print(f"[INFO] Waiting {motion.home_wait:.1f}s for homing")
        time.sleep(motion.home_wait)

        send_command(ser, motion.init_command, "Initial pose")
        print(f"[INFO] Waiting {motion.settle_wait:.1f}s for initial pose")
        time.sleep(motion.settle_wait)

        for step, current_j3 in joint3_sequence(motion):
            command = motion.format_pose(
                motion.initial_j1,
                motion.initial_j2,
                current_j3,
                motion.initial_j4,
                motion.initial_j5,
                motion.initial_j6,
            )
            send_command(ser, command, f"Step {step}/{motion.total_steps}")

            if args.capture_delay > 0:
                time.sleep(args.capture_delay)

            image_path = camera_session.capture_step(step, current_j3)
            record = {
                "timestamp": iso_timestamp(),
                "step": step,
                "total_steps": motion.total_steps,
                "joint1": motion.initial_j1,
                "joint2": motion.initial_j2,
                "joint3": round(current_j3, 4),
                "joint4": motion.initial_j4,
                "joint5": motion.initial_j5,
                "joint6": motion.initial_j6,
                "speed": motion.speed,
                "pause_time": motion.pause_time,
                "capture_delay": args.capture_delay,
                "image_path": str(image_path),
            }
            append_capture_log(log_csv_path, record)
            capture_records.append(record)
            print(
                f"[CAPTURE] Step {step}/{motion.total_steps} | "
                f"Joint 3: {current_j3:.1f} deg | Saved: {image_path}"
            )

            remaining_pause = motion.pause_time - args.capture_delay
            if remaining_pause > 0:
                time.sleep(remaining_pause)

        write_summary_json(
            summary_json_path,
            {
                "generated_at": iso_timestamp(),
                "motion": asdict(motion),
                "camera": {
                    "camera_index": args.camera_index,
                    "exposure_us": args.exposure_us,
                    "gain": args.gain,
                    "output_dir": str(camera_session.output_dir),
                },
                "files": {
                    "log_csv": str(log_csv_path.resolve()),
                    "summary_json": str(summary_json_path.resolve()),
                },
                "captures": capture_records,
            },
        )
        print(f"[OK] Capture log saved: {log_csv_path.resolve()}")
        print(f"[OK] Capture summary saved: {summary_json_path.resolve()}")
        print("[DONE] Sweep and capture completed")
        return 0
    except KeyboardInterrupt:
        print("[WARN] Motion interrupted by user")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        camera_session.close()

        if ser is not None:
            try:
                send_command(ser, motion.init_command, "Return to safe pose")
                time.sleep(motion.settle_wait)
                if motion.disable_on_exit:
                    send_command(ser, b"!DISABLE\n", "Disable")
            except Exception as exc:
                print(f"[WARN] Failed to return to safe pose: {exc}", file=sys.stderr)

            try:
                ser.close()
                print("[OK] Serial port closed")
            except Exception as exc:
                print(f"[WARN] Failed to close serial port: {exc}", file=sys.stderr)


def main() -> int:
    args, motion = build_parser()
    return run_scan_with_capture(args, motion)


if __name__ == "__main__":
    rc = main()
    os._exit(rc)