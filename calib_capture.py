from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from zwo_camera import ZwoCamera

CHESSBOARD_SIZE = (9, 6)  # internal corners (cols, rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture chessboard images from ZWO camera for intrinsic calibration.",
    )
    p.add_argument("--camera-id", type=int, default=0, help="ZWO camera index")
    p.add_argument("--width", type=int, default=1920, help="ROI width")
    p.add_argument("--height", type=int, default=1080, help="ROI height")
    p.add_argument("--exposure", type=int, default=50000, help="Initial exposure (us)")
    p.add_argument("--gain", type=int, default=50, help="Initial gain")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calib_images"),
        help="Directory to save captured images",
    )
    return p


def draw_ui(frame: np.ndarray, count: int, exposure: int, gain: int) -> np.ndarray:
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    lines = [
        f"Captured: {count} | Exposure: {exposure}us | Gain: {gain}",
        "[SPACE] Capture  [Q/ESC] Quit  [+/-] Exposure  [g/G] Gain",
        "Aim for 20-30 images, varied angles, chessboard fully visible",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            display, line, (10, 25 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0) if i == 0 else (200, 200, 200), 1, cv2.LINE_AA,
        )
    return display


def main() -> int:
    args = build_parser().parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cam = ZwoCamera()
        cam.open(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            exposure_us=args.exposure,
            gain=args.gain,
        )
        print(f"[INFO] Camera: {cam.camera_info.name}")
    except RuntimeError as e:
        print(f"[ERROR] Failed to open camera: {e}")
        return 1

    exposure = args.exposure
    gain = args.gain
    captured = 0

    print("=" * 55)
    print("  Chessboard Capture Tool")
    print("  Prepare a 10x7 chessboard printed on rigid flat surface")
    print(f"  Internal corners: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}")
    print("  Keys: [SPACE] Capture  [Q] Quit  [+/-] Exp  [g/G] Gain")
    print("=" * 55)

    cv2.namedWindow("Chessboard Capture", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.grab_frame()

            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            ret_corners, corners = cv2.findChessboardCorners(
                gray, CHESSBOARD_SIZE, None,
            )

            display = draw_ui(gray, captured, cam.exposure_us, cam.gain)

            if ret_corners:
                cv2.drawChessboardCorners(
                    display, CHESSBOARD_SIZE, corners, ret_corners,
                )

            cv2.imshow("Chessboard Capture", display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                print("[INFO] Quit")
                break

            if key == ord(" "):
                if ret_corners:
                    fname = output_dir / f"calib_{captured:04d}.png"
                    cv2.imwrite(str(fname), gray)
                    captured += 1
                    print(f"[SAVE] {fname}  (total: {captured})")
                else:
                    print("[WARN] No chessboard detected — frame NOT saved")

            if key in (ord("+"), ord("=")):
                exposure = min(exposure + 10000, 5000000)
                cam.set_exposure(exposure)
            if key == ord("-"):
                exposure = max(exposure - 10000, 1000)
                cam.set_exposure(exposure)
            if key == ord("g"):
                gain = min(gain + 10, 500)
                cam.set_gain(gain)
            if key == ord("G"):
                gain = max(gain - 10, 0)
                cam.set_gain(gain)

    except KeyboardInterrupt:
        print("[INFO] Interrupted")
    finally:
        cam.close()
        cv2.destroyAllWindows()

    print(f"[DONE] {captured} images saved to {output_dir}")
    if captured < 10:
        print("[WARN] Fewer than 10 images — calibration may be inaccurate. Aim for 20-30.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
