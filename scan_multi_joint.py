from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import serial
from PIL import Image

from main import MotionConfig, send_command
from robot_kinematics import forward_kinematics, pose_to_rt, pose_to_rpy
from zwo_camera import ZwoCamera


# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------
@dataclass
class ScanConfig:
    name: str
    description: str = ""
    waypoints: List[List[float]] = field(default_factory=list)  # [[j1..j6], ...]
    speeds: List[int] = field(default_factory=list)             # per-waypoint speed
    waypoint_labels: List[str] = field(default_factory=list)    # human-readable labels


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def generate_arc_scan(
    base_joints: List[float],
    joint_index: int,        # which joint to vary (0-based)
    start_angle: float,
    end_angle: float,
    steps: int,
    speed: int = 50,
) -> ScanConfig:
    """Sweep a single joint through an angle range, keeping others fixed."""
    waypoints: List[List[float]] = []
    labels: List[str] = []
    angles = [start_angle + (end_angle - start_angle) * i / max(steps - 1, 1)
              for i in range(steps)]

    joint_name = f"J{joint_index + 1}"
    for i, angle in enumerate(angles):
        joints = list(base_joints)
        joints[joint_index] = round(angle, 2)
        waypoints.append(joints)
        labels.append(f"{joint_name}={angle:.1f}°")
    return ScanConfig(
        name=f"arc_{joint_name}",
        description=f"Sweep {joint_name} from {start_angle}° to {end_angle}° "
                    f"in {steps} steps",
        waypoints=waypoints,
        speeds=[speed] * steps,
        waypoint_labels=labels,
    )


def generate_hemisphere_scan(
    base_joints: List[float],
    j1_range: tuple = (-60, 60),
    j1_steps: int = 5,
    j2_range: tuple = (-70, -55),
    j3_range: tuple = (140, 160),
    elevation_steps: int = 4,
    speed: int = 50,
) -> ScanConfig:
    """Hemisphere scan: sweep J1 in an arc at multiple elevations.

    Varies J2 and J3 together to change camera elevation, and J1 for
    horizontal coverage around the plant. Starting from the farthest
    (lowest elevation, smaller J3), sweeps J1 in full arcs, then
    steps to the next elevation.

    Returns waypoints that proceed systematically in rows.
    """
    waypoints: List[List[float]] = []
    labels: List[str] = []
    speeds: List[int] = []

    j2_levels = [j2_range[0] + (j2_range[1] - j2_range[0]) * i / max(elevation_steps - 1, 1)
                 for i in range(elevation_steps)]
    j3_levels = [j3_range[0] + (j3_range[1] - j3_range[0]) * i / max(elevation_steps - 1, 1)
                 for i in range(elevation_steps)]

    # Odd rows: J1 goes left → right,  even rows: right → left  (zigzag)
    for row in range(elevation_steps):
        row_j1_start = j1_range[0] if row % 2 == 0 else j1_range[1]
        row_j1_end = j1_range[1] if row % 2 == 0 else j1_range[0]
        j1_angles = np.linspace(row_j1_start, row_j1_end, j1_steps)

        for j1 in j1_angles:
            joints = list(base_joints)
            joints[0] = round(float(j1), 2)
            joints[1] = round(j2_levels[row], 2)
            joints[2] = round(j3_levels[row], 2)
            waypoints.append(joints)
            labels.append(f"J1={j1:.1f}° J2={j2_levels[row]:.1f}° J3={j3_levels[row]:.1f}°")
            speeds.append(speed)

    return ScanConfig(
        name="hemisphere",
        description=(f"Hemisphere scan: {elevation_steps} elevations × "
                     f"{j1_steps} horizontal steps per row "
                     f"(J1={j1_range}, J2={j2_range}, J3={j3_range})"),
        waypoints=waypoints,
        speeds=speeds,
        waypoint_labels=labels,
    )


def load_waypoints_from_file(path: Path) -> ScanConfig:
    """Load waypoints from CSV or JSON.

    CSV format: j1,j2,j3,j4,j5,j6,speed[,label]
    JSON format: {"waypoints": [[j1..j6], ...], "speeds": [...], "labels": [...]}
    """
    if path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
        return ScanConfig(
            name=path.stem,
            description=f"Loaded from {path.name}",
            waypoints=data["waypoints"],
            speeds=data.get("speeds", [50] * len(data["waypoints"])),
            waypoint_labels=data.get("labels",
                                     [f"wp_{i}" for i in range(len(data["waypoints"]))]),
        )
    elif path.suffix.lower() == ".csv":
        waypoints, speeds, labels = [], [], []
        with path.open() as f:
            for i, row in enumerate(csv.reader(f)):
                if not row or row[0].startswith("#"):
                    continue
                waypoints.append([float(v) for v in row[:6]])
                speeds.append(int(row[6]) if len(row) > 6 else 50)
                labels.append(row[7] if len(row) > 7 else f"wp_{i}")
        return ScanConfig(
            name=path.stem,
            description=f"Loaded from {path.name}",
            waypoints=waypoints,
            speeds=speeds,
            waypoint_labels=labels,
        )
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def format_move(joints: List[float], speed: int) -> bytes:
    return (f"&{joints[0]:.1f},{joints[1]:.1f},{joints[2]:.1f},"
            f"{joints[3]:.1f},{joints[4]:.1f},{joints[5]:.1f},{speed};\n"
            ).encode("utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-joint robot arm scan with ZWO camera capture.",
    )

    g = p.add_argument_group("robot connection")
    g.add_argument("--port", default="/dev/ttyACM0")
    g.add_argument("--baudrate", type=int, default=115200)
    g.add_argument("--timeout", type=float, default=1.0)

    g = p.add_argument_group("scan mode")
    g.add_argument("--mode", choices=["arc", "elevation", "hemisphere", "file"],
                   default="arc", help="Scan pattern mode")
    g.add_argument("--waypoints", type=Path, default=None,
                   help="JSON/CSV file of waypoints (for --mode file)")

    g = p.add_argument_group("trajectory (arc / elevation / hemisphere)")
    g.add_argument("--joint", type=int, default=3,
                   help="Joint to sweep for arc/elevation mode (1-6)")
    g.add_argument("--start-angle", type=float, default=135.0,
                   help="Sweep start angle (degrees)")
    g.add_argument("--end-angle", type=float, default=175.0,
                   help="Sweep end angle (degrees)")
    g.add_argument("--steps", type=int, default=40,
                   help="Number of steps in the sweep")
    g.add_argument("--j1-range", type=str, default="-60,60",
                   help="J1 range for hemisphere: start,end")
    g.add_argument("--j2-range", type=str, default="-70,-55",
                   help="J2 range for hemisphere")
    g.add_argument("--j3-range", type=str, default="140,160",
                   help="J3 range for hemisphere")
    g.add_argument("--j1-steps", type=int, default=5,
                   help="Horizontal steps per row in hemisphere")
    g.add_argument("--elevation-steps", type=int, default=4,
                   help="Number of elevation levels")

    g = p.add_argument_group("base pose")
    g.add_argument("--j1", type=float, default=0.0)
    g.add_argument("--j2", type=float, default=-64.0)
    g.add_argument("--j3", type=float, default=155.0)
    g.add_argument("--j4", type=float, default=-1.0)
    g.add_argument("--j5", type=float, default=0.0)
    g.add_argument("--j6", type=float, default=0.0)
    g.add_argument("--speed", type=int, default=50)

    g = p.add_argument_group("timing")
    g.add_argument("--home-wait", type=float, default=5.0)
    g.add_argument("--settle-wait", type=float, default=2.0)
    g.add_argument("--capture-delay", type=float, default=0.3)
    g.add_argument("--pause-time", type=float, default=1.0,
                   help="Total time per waypoint (includes capture)")

    g = p.add_argument_group("camera")
    g.add_argument("--camera-id", type=int, default=0)
    g.add_argument("--camera-width", type=int, default=1920)
    g.add_argument("--camera-height", type=int, default=1080)
    g.add_argument("--exposure", type=int, default=50000)
    g.add_argument("--gain", type=int, default=50)

    g.add_argument("--output-dir", type=Path, default=Path("scans"),
                   help="Output directory")
    g.add_argument("--compute-fk", action="store_true", default=False,
                   help="Also compute & log camera poses via FK")
    g.add_argument("--disable", action="store_true", default=False,
                   help="Send !DISABLE on exit")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    base_joints = [args.j1, args.j2, args.j3, args.j4, args.j5, args.j6]

    # --- Build scan ---
    if args.mode == "file":
        if args.waypoints is None:
            print("[ERROR] --waypoints required for --mode file")
            return 1
        scan = load_waypoints_from_file(args.waypoints)
    elif args.mode == "arc":
        scan = generate_arc_scan(
            base_joints, args.joint - 1,
            args.start_angle, args.end_angle,
            args.steps, args.speed,
        )
    elif args.mode == "elevation":
        scan = generate_arc_scan(
            base_joints, args.joint - 1,
            args.start_angle, args.end_angle,
            args.steps, args.speed,
        )
    else:  # hemisphere
        scan = generate_hemisphere_scan(
            base_joints,
            j1_range=tuple(float(v) for v in args.j1_range.split(",")),
            j1_steps=args.j1_steps,
            j2_range=tuple(float(v) for v in args.j2_range.split(",")),
            j3_range=tuple(float(v) for v in args.j3_range.split(",")),
            elevation_steps=args.elevation_steps,
            speed=args.speed,
        )

    n_waypoints = len(scan.waypoints)
    if n_waypoints == 0:
        print("[ERROR] No waypoints generated")
        return 1

    # --- Setup output ---
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    log_csv = output_dir / "captures_log.csv"
    summary_json = output_dir / "captures_summary.json"

    # --- Connect robot ---
    ser: serial.Serial | None = None
    cam: ZwoCamera | None = None

    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=args.timeout)
        print(f"[OK] Serial: {args.port} @ {args.baudrate}")
    except serial.SerialException as e:
        print(f"[ERROR] Serial: {e}")
        return 1

    # --- Home ---
    send_command(ser, b"!START\n", "Home")
    print(f"[INFO] Waiting {args.home_wait:.1f}s for homing")
    time.sleep(args.home_wait)

    # --- Open camera ---
    try:
        cam = ZwoCamera()
        cam.open(args.camera_id, args.camera_width, args.camera_height,
                 args.exposure, args.gain)
        print(f"[OK] ZWO: {cam.camera_info.name} "
              f"{cam.width}x{cam.height} exp={cam.exposure_us}us gain={cam.gain}")
    except RuntimeError as e:
        print(f"[ERROR] Camera: {e}")
        ser.close()
        return 1

    # --- Scan ---
    print(f"\n{'='*55}")
    print(f"  SCAN: {scan.name}")
    print(f"  {scan.description}")
    print(f"  Waypoints: {n_waypoints}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*55}\n")

    capture_records: list[dict] = []

    try:
        for i, (joints, speed) in enumerate(zip(scan.waypoints, scan.speeds)):
            label = scan.waypoint_labels[i] if i < len(scan.waypoint_labels) else f"wp_{i}"
            print(f"[{i+1}/{n_waypoints}] {label}")
            print(f"         Joints: [{', '.join(f'{j:.1f}' for j in joints)}]")

            # Move
            cmd = format_move(joints, speed)
            send_command(ser, cmd, f"Move")
            time.sleep(args.capture_delay)

            # Capture
            frame = cam.grab_frame()
            img_path = images_dir / f"scan_{i:04d}.png"
            Image.fromarray(frame).save(img_path)

            # Record
            record = {
                "timestamp": iso_timestamp(),
                "step": i + 1,
                "total_steps": n_waypoints,
                "label": label,
                "joint1": joints[0], "joint2": joints[1], "joint3": joints[2],
                "joint4": joints[3], "joint5": joints[4], "joint6": joints[5],
                "speed": speed,
                "image_path": str(img_path),
            }

            # Optional FK
            if args.compute_fk:
                T_ee = forward_kinematics(joints)
                _R, t = pose_to_rt(T_ee)
                r, p, y = pose_to_rpy(T_ee)
                record.update({
                    "ee_x": round(float(t[0]) * 1000, 2),
                    "ee_y": round(float(t[1]) * 1000, 2),
                    "ee_z": round(float(t[2]) * 1000, 2),
                    "ee_roll_deg": round(float(np.degrees(r)), 2),
                    "ee_pitch_deg": round(float(np.degrees(p)), 2),
                    "ee_yaw_deg": round(float(np.degrees(y)), 2),
                })

            capture_records.append(record)

            remaining = max(0, args.pause_time - args.capture_delay)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("[WARN] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        # --- Close camera ---
        if cam:
            cam.close()

        # --- Return to safe pose ---
        if ser:
            try:
                send_command(ser,
                             format_move(base_joints, args.speed),
                             "Return to base pose")
                time.sleep(args.settle_wait)
                if args.disable:
                    send_command(ser, b"!DISABLE\n", "Disable")
            except Exception as e:
                print(f"[WARN] Failed to return: {e}")
            ser.close()

    # --- Write logs ---
    summary = {
        "generated_at": iso_timestamp(),
        "scan": {
            "name": scan.name,
            "description": scan.description,
            "mode": args.mode,
            "num_waypoints": n_waypoints,
        },
        "robot": {
            "port": args.port,
            "base_joints": base_joints,
        },
        "camera": {
            "camera_id": args.camera_id,
            "width": args.camera_width,
            "height": args.camera_height,
            "exposure_us": args.exposure,
            "gain": args.gain,
        },
        "captures": capture_records,
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV log
    fieldnames = [
        "timestamp", "step", "total_steps", "label",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "speed", "image_path",
    ]
    if args.compute_fk:
        fieldnames += ["ee_x", "ee_y", "ee_z",
                       "ee_roll_deg", "ee_pitch_deg", "ee_yaw_deg"]

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(capture_records)

    print(f"[SAVE]  {summary_json}")
    print(f"[SAVE]  {log_csv}")
    print(f"[DONE]  {n_waypoints} frames saved to {images_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
