from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from zwo_camera import ZwoCamera

CHESSBOARD_SIZE = (9, 6)  # internal corners (cols, rows)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        data = json.load(f)
    K = data["camera_matrix"]
    mtx = np.array([[K["fx"], 0, K["cx"]],
                    [0, K["fy"], K["cy"]],
                    [0, 0, 1]], dtype=np.float64)
    dist = np.array(data["distortion_coefficients"]["vector"], dtype=np.float64)
    return mtx, dist


# ---------------------------------------------------------------------------
# --mode capture
# ---------------------------------------------------------------------------

def draw_ui(frame: np.ndarray, count: int, exposure: int, gain: int) -> np.ndarray:
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    lines = [
        f"Captured: {count} | Exposure: {exposure}us | Gain: {gain}",
        "[SPACE] Capture  [Q/ESC] Quit  [+/-] Exposure  [g/G] Gain",
        "Aim for 20-30 varied-angle images, chessboard fully visible",
    ]
    for i, line in enumerate(lines):
        cv2.putText(display, line, (10, 25 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if i == 0 else (200, 200, 200), 1, cv2.LINE_AA)
    return display


def run_capture(args: argparse.Namespace) -> int:
    output_dir = (args.output_dir or Path("calib_images")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cam = ZwoCamera()
        cam.open(args.camera_id, args.width, args.height, args.exposure, args.gain)
        print(f"[INFO] Camera: {cam.camera_info.name}  {cam.width}x{cam.height}")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1

    exposure = args.exposure
    gain = args.gain
    captured = 0

    print("=" * 55)
    print("  Chessboard Capture — 10×7 grid / 9×6 internal corners")
    print("  [SPACE] Save  [Q] Quit  [+/-] Exposure  [g/G] Gain")
    print("=" * 55)

    cv2.namedWindow("Chessboard Capture", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = cam.grab_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            display = draw_ui(gray, captured, cam.exposure_us, cam.gain)
            if ret_corners:
                cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret_corners)

            cv2.imshow("Chessboard Capture", display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                print("[INFO] Quit"); break
            if key == ord(" "):
                if ret_corners:
                    fname = output_dir / f"calib_{captured:04d}.png"
                    cv2.imwrite(str(fname), gray)
                    captured += 1; print(f"[SAVE] {fname}  (total: {captured})")
                else:
                    print("[WARN] No chessboard — not saved")
            if key in (ord("+"), ord("=")):
                exposure = min(exposure + 10000, 5000000); cam.set_exposure(exposure)
            if key == ord("-"):
                exposure = max(exposure - 10000, 1000); cam.set_exposure(exposure)
            if key == ord("g"):
                gain = min(gain + 10, 500); cam.set_gain(gain)
            if key == ord("G"):
                gain = max(gain - 10, 0); cam.set_gain(gain)
    except KeyboardInterrupt:
        print("[INFO] Interrupted")
    finally:
        cam.close(); cv2.destroyAllWindows()

    print(f"[DONE] {captured} images → {output_dir}")
    if captured < 10:
        print("[WARN] Fewer than 10 — aim for 20-30.")
    return 0


# ---------------------------------------------------------------------------
# --mode compute
# ---------------------------------------------------------------------------

def detect_corners(image_paths: list[Path], pattern_size: tuple[int, int],
                   show: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    obj_points, img_points = [], []
    for path in sorted(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[SKIP] Cannot read: {path}"); continue
        ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
        if not ret:
            print(f"[SKIP] No chessboard: {path}"); continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp); img_points.append(refined)
        print(f"[OK]   {path.name}")
        if show:
            d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Corners", cv2.drawChessboardCorners(d, pattern_size, refined, True))
            cv2.waitKey(500)
    if show: cv2.destroyWindow("Corners")
    print(f"\n[INFO] Detected: {len(obj_points)} / {len(image_paths)} images")
    return obj_points, img_points


def run_compute(args: argparse.Namespace) -> int:
    input_dir = (args.input_dir or Path("calib_images")).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] Not found: {input_dir}"); return 1

    image_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    if not image_paths:
        print(f"[ERROR] No images in {input_dir}"); return 1

    print(f"[INFO] {len(image_paths)} images, square={args.square_size}mm")
    obj_points, img_points = detect_corners(image_paths, CHESSBOARD_SIZE, args.show_corners)

    if len(obj_points) < 8:
        print("[ERROR] Need ≥8 usable images."); return 1

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    first = first or cv2.imread(str(image_paths[1]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        print("[ERROR] Cannot determine image size."); return 1
    h, w = first.shape

    obj_scaled = [p * args.square_size for p in obj_points]
    print("\nRunning calibration...")
    ok, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_scaled, img_points, (w, h), None, None)
    if not ok:
        print("[ERROR] Calibration failed."); return 1

    total, npts = 0.0, 0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_scaled[i], rvecs[i], tvecs[i], mtx, dist)
        total += cv2.norm(img_points[i], proj, cv2.NORM_L2) ** 2
        npts += len(img_points[i])
    me = np.sqrt(total / npts)

    print("\n" + "=" * 55)
    print("  CALIBRATION RESULTS")
    print("=" * 55)
    print(f"\n  fx={mtx[0,0]:.3f}  fy={mtx[1,1]:.3f}  cx={mtx[0,2]:.3f}  cy={mtx[1,2]:.3f}")
    print(f"  dist: [{', '.join(f'{v:.6f}' for v in dist.ravel())}]")
    print(f"  Reprojection error: {me:.4f} px  ({len(obj_points)} images)")
    level = ("excellent" if me < 0.3 else "good" if me < 0.5 else
             "acceptable" if me < 1.0 else "poor")
    print(f"  Quality: {level}")

    print("\n  Per-image errors:")
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_scaled[i], rvecs[i], tvecs[i], mtx, dist)
        e = cv2.norm(img_points[i], proj, cv2.NORM_L2)
        flag = "  <-- outlier" if e > me * 1.5 else ""
        print(f"    {i:2d}: {e:.4f} px{flag}")

    out = {
        "calibration_info": {
            "camera": "ZWO ASI", "image_width": w, "image_height": h,
            "chessboard_corners": list(CHESSBOARD_SIZE),
            "square_size_mm": args.square_size, "num_images": len(obj_points),
            "mean_reprojection_error_px": round(me, 4), "quality": level,
        },
        "camera_matrix": {
            "fx": round(float(mtx[0,0]), 3), "fy": round(float(mtx[1,1]), 3),
            "cx": round(float(mtx[0,2]), 3), "cy": round(float(mtx[1,2]), 3),
            "matrix": [[round(float(v), 3) for v in row] for row in mtx.tolist()],
        },
        "distortion_coefficients": {
            "k1": round(float(dist[0,0]), 6), "k2": round(float(dist[0,1]), 6),
            "p1": round(float(dist[0,2]), 6), "p2": round(float(dist[0,3]), 6),
            "k3": round(float(dist[0,4]), 6),
            "vector": [round(float(v), 6) for v in dist.ravel().tolist()],
        },
    }
    output_path = (args.output or Path("camera_intrinsics.json")).resolve()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {output_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Camera intrinsic calibration: capture + compute.")
    sub = p.add_subparsers(dest="mode", required=True)

    # capture
    cap = sub.add_parser("capture", help="Capture chessboard images from ZWO camera")
    cap.add_argument("--camera-id", type=int, default=0)
    cap.add_argument("--width", type=int, default=1920)
    cap.add_argument("--height", type=int, default=1080)
    cap.add_argument("--exposure", type=int, default=50000)
    cap.add_argument("--gain", type=int, default=50)
    cap.add_argument("--output-dir", type=Path, default=Path("calib_images"))

    # compute
    com = sub.add_parser("compute", help="Compute intrinsics from captured images")
    com.add_argument("--input-dir", type=Path, default=Path("calib_images"))
    com.add_argument("--square-size", type=float, default=25.0)
    com.add_argument("--output", type=Path, default=Path("camera_intrinsics.json"))
    com.add_argument("--show-corners", action="store_true", default=False)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "capture":
        return run_capture(args)
    return run_compute(args)


if __name__ == "__main__":
    raise SystemExit(main())
