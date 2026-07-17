from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import serial


@dataclass(frozen=True)
class MotionConfig:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    timeout: float = 1.0
    home_wait: float = 5.0
    settle_wait: float = 3.0
    pause_time: float = 0.5
    total_steps: int = 200
    step_angle: float = 0.1
    initial_j1: float = 0.0
    initial_j2: float = -64.0
    initial_j3: float = 155.0
    initial_j4: float = -1.0
    initial_j5: float = 0.0
    initial_j6: float = 0.0
    speed: int = 50
    disable_on_exit: bool = False

    @property
    def init_command(self) -> bytes:
        return self.format_pose(
            self.initial_j1,
            self.initial_j2,
            self.initial_j3,
            self.initial_j4,
            self.initial_j5,
            self.initial_j6,
        )

    def format_pose(
        self,
        j1: float,
        j2: float,
        j3: float,
        j4: float,
        j5: float,
        j6: float,
    ) -> bytes:
        return (
            f"&{j1:.1f},{j2:.1f},{j3:.1f},{j4:.1f},{j5:.1f},{j6:.1f},{self.speed};\n"
        ).encode("utf-8")


def parse_args() -> MotionConfig:
    parser = argparse.ArgumentParser(
        description="Control the robot arm over serial and sweep joint 3 upward."
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
    parser.add_argument("--pause-time", type=float, default=0.5, help="Pause between steps")
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
    args = parser.parse_args()

    if args.total_steps <= 0:
        parser.error("--total-steps must be greater than 0")
    if args.step_angle <= 0:
        parser.error("--step-angle must be greater than 0")
    if args.pause_time < 0 or args.home_wait < 0 or args.settle_wait < 0:
        parser.error("wait values must be non-negative")
    if args.speed <= 0:
        parser.error("--speed must be greater than 0")

    return MotionConfig(
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


def send_command(ser: serial.Serial, command: bytes, label: str) -> None:
    ser.write(command)
    ser.flush()
    print(f"[SEND] {label}: {command.decode('utf-8').strip()}")


def joint3_sequence(config: MotionConfig) -> Iterable[tuple[int, float]]:
    current_j3 = config.initial_j3
    for step in range(1, config.total_steps + 1):
        current_j3 -= config.step_angle
        yield step, current_j3


def run_motion(config: MotionConfig) -> int:
    ser: serial.Serial | None = None

    try:
        ser = serial.Serial(config.port, config.baudrate, timeout=config.timeout)
        print(f"[OK] Serial connected: {config.port} @ {config.baudrate}")

        send_command(ser, b"!START\n", "Home")
        print(f"[INFO] Waiting {config.home_wait:.1f}s for homing")
        time.sleep(config.home_wait)

        send_command(ser, config.init_command, "Initial pose")
        print(f"[INFO] Waiting {config.settle_wait:.1f}s for initial pose")
        time.sleep(config.settle_wait)

        for step, current_j3 in joint3_sequence(config):
            command = config.format_pose(
                config.initial_j1,
                config.initial_j2,
                current_j3,
                config.initial_j4,
                config.initial_j5,
                config.initial_j6,
            )
            send_command(ser, command, f"Step {step}/{config.total_steps}")
            print(f"[INFO] Joint 3 angle: {current_j3:.1f} deg")
            time.sleep(config.pause_time)

        print("[DONE] Sweep completed")
        return 0
    except KeyboardInterrupt:
        print("[WARN] Motion interrupted by user")
        return 130
    except serial.SerialException as exc:
        print(f"[ERROR] Serial error: {exc}", file=sys.stderr)
        return 1
    finally:
        if ser is not None:
            try:
                send_command(ser, config.init_command, "Return to safe pose")
                time.sleep(config.settle_wait)
                if config.disable_on_exit:
                    send_command(ser, b"!DISABLE\n", "Disable")
            except Exception as exc:
                print(f"[WARN] Failed to return to safe pose: {exc}", file=sys.stderr)

            try:
                ser.close()
                print("[OK] Serial port closed")
            except Exception as exc:
                print(f"[WARN] Failed to close serial port: {exc}", file=sys.stderr)


def main() -> int:
    config = parse_args()
    return run_motion(config)


if __name__ == "__main__":
    raise SystemExit(main())