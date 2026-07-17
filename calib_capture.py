from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# ZWO ASI SDK constants
# ---------------------------------------------------------------------------
ASI_SUCCESS = 0
ASI_GAIN = 0
ASI_EXPOSURE = 1
ASI_FALSE = 0
ASI_IMG_RAW8 = 0

# ---------------------------------------------------------------------------
# ZWO ASI SDK wrapper
# ---------------------------------------------------------------------------


def _check(ret: int, label: str) -> None:
    if ret != ASI_SUCCESS:
        raise RuntimeError(f"[ZWO] {label} failed (code={ret})")


class ZwoASI:
    """Thin ctypes wrapper around ZWO ASI SDK v1.41."""

    def __init__(self, sdk_path: str):
        self.lib = ctypes.CDLL(sdk_path)
        self.camera_id: int = -1
        self._opened = False

    def get_num_cameras(self) -> int:
        self.lib.ASIGetNumOfConnectedCameras.restype = ctypes.c_int
        return self.lib.ASIGetNumOfConnectedCameras()

    def open(self, camera_id: int, width: int, height: int) -> None:
        self.camera_id = camera_id

        self.lib.ASIOpenCamera.restype = ctypes.c_int
        self.lib.ASIOpenCamera.argtypes = [ctypes.c_int]

        self.lib.ASIInitCamera.restype = ctypes.c_int
        self.lib.ASIInitCamera.argtypes = [ctypes.c_int]

        self.lib.ASISetControlValue.restype = ctypes.c_int
        self.lib.ASISetControlValue.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_int,
        ]

        self.lib.ASISetROIFormat.restype = ctypes.c_int
        self.lib.ASISetROIFormat.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.ASIStartVideoCapture.restype = ctypes.c_int
        self.lib.ASIStartVideoCapture.argtypes = [ctypes.c_int]

        self.lib.ASIGetVideoData.restype = ctypes.c_int
        self.lib.ASIGetVideoData.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
        ]

        self.lib.ASIStopVideoCapture.restype = ctypes.c_int
        self.lib.ASIStopVideoCapture.argtypes = [ctypes.c_int]

        self.lib.ASICloseCamera.restype = ctypes.c_int
        self.lib.ASICloseCamera.argtypes = [ctypes.c_int]

        _check(self.lib.ASIOpenCamera(camera_id), "ASIOpenCamera")
        _check(self.lib.ASIInitCamera(camera_id), "ASIInitCamera")
        _check(
            self.lib.ASISetROIFormat(camera_id, width, height, 1, ASI_IMG_RAW8),
            "ASISetROIFormat",
        )
        _check(self.lib.ASIStartVideoCapture(camera_id), "ASIStartVideoCapture")
        self._opened = True

    def set_exposure(self, exposure_us: int) -> None:
        _check(
            self.lib.ASISetControlValue(self.camera_id, ASI_EXPOSURE, exposure_us, ASI_FALSE),
            "ASISetControlValue(EXPOSURE)",
        )

    def set_gain(self, gain: int) -> None:
        _check(
            self.lib.ASISetControlValue(self.camera_id, ASI_GAIN, gain, ASI_FALSE),
            "ASISetControlValue(GAIN)",
        )

    def grab_frame(self, width: int, height: int) -> np.ndarray:
        buf_size = width * height
        buf = ctypes.create_string_buffer(buf_size)
        pbuf = ctypes.cast(buf, ctypes.c_void_p)
        ret = self.lib.ASIGetVideoData(self.camera_id, pbuf, buf_size, 1000)
        _check(ret, "ASIGetVideoData")
        return np.frombuffer(buf.raw, dtype=np.uint8).reshape(height, width)

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self.lib.ASIStopVideoCapture(self.camera_id)
            self.lib.ASICloseCamera(self.camera_id)
        except Exception:
            pass
        self._opened = False


# ---------------------------------------------------------------------------
# Chessboard capture UI
# ---------------------------------------------------------------------------

CHESSBOARD_SIZE = (9, 6)  # internal corners (cols, rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture chessboard images from ZWO camera for intrinsic calibration.",
    )
    p.add_argument(
        "--sdk",
        default="/home/hu/桌面/ASI_linux_mac_SDK_V1.41/lib/x64/libASICamera2.so",
        help="Path to libASICamera2.so",
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
    """Overlay status text on the frame."""
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    lines = [
        f"Captured: {count} | Exposure: {exposure}us | Gain: {gain}",
        "[SPACE] Capture  [Q/ESC] Quit  [+/-] Exposure  [g/G] Gain  [r] Reset corners",
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
    sdk_path = os.path.realpath(args.sdk)
    if not os.path.isfile(sdk_path):
        print(f"[ERROR] SDK not found: {sdk_path}")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Open camera ---
    cam = ZwoASI(sdk_path)
    try:
        num = cam.get_num_cameras()
        if num <= 0:
            print("[ERROR] No ZWO camera detected")
            return 1
        print(f"[INFO] {num} camera(s) connected")

        cam.open(args.camera_id, args.width, args.height)
        cam.set_exposure(args.exposure)
        cam.set_gain(args.gain)
        print(f"[INFO] Camera opened: {args.width}x{args.height}")
    except Exception as e:
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
            frame = cam.grab_frame(args.width, args.height)
            corners_found = False

            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            ret_corners, corners = cv2.findChessboardCorners(
                gray, CHESSBOARD_SIZE, None,
            )
            corners_found = ret_corners

            display = draw_ui(gray, captured, exposure, gain)

            if corners_found:
                cv2.drawChessboardCorners(
                    display, CHESSBOARD_SIZE, corners, ret_corners,
                )

            cv2.imshow("Chessboard Capture", display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):  # Q or ESC
                print("[INFO] Quit")
                break

            if key == ord(" "):
                if corners_found:
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
