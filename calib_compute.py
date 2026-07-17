from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CHESSBOARD_SIZE = (9, 6)  # (cols, rows) internal corners


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute camera intrinsics from chessboard images.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("calib_images"),
        help="Directory containing calibration images",
    )
    p.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Physical size of one chessboard square in mm",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("camera_intrinsics.json"),
        help="Output JSON file for calibration results",
    )
    p.add_argument(
        "--show-corners",
        action="store_true",
        default=False,
        help="Show corner detection result for each image",
    )
    return p


def detect_corners(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    show: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns: (object_points, image_points)
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    used_paths: list[Path] = []

    for path in sorted(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[SKIP] Could not read: {path}")
            continue

        ret, corners = cv2.findChessboardCorners(img, pattern_size, None)

        if not ret:
            print(f"[SKIP] No chessboard found: {path}")
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners_refined)
        used_paths.append(path)
        print(f"[OK]   {path.name}")

        if show:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            display = cv2.drawChessboardCorners(display, pattern_size, corners_refined, True)
            cv2.imshow("Corners", display)
            cv2.waitKey(500)

    if show:
        cv2.destroyWindow("Corners")

    print(f"\n[INFO] Detected corners in {len(obj_points)} / {len(image_paths)} images")
    return obj_points, img_points


def calibrate(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    """
    Returns: (camera_matrix, dist_coeffs, rvecs, tvecs, mean_error)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None,
    )
    if not ret:
        raise RuntimeError("Calibration failed to converge")

    total_error = 0.0
    total_points = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(img_points[i])
    mean_error = np.sqrt(total_error / total_points)

    return mtx, dist, rvecs, tvecs, mean_error


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Directory not found: {input_dir}")
        return 1

    image_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}")
        return 1

    print(f"[INFO] Found {len(image_paths)} images in {input_dir}")
    print(f"[INFO] Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} corners, "
          f"square size = {args.square_size}mm")
    print()

    # --- Detect corners ---
    print("Detecting chessboard corners...")
    obj_points, img_points = detect_corners(image_paths, CHESSBOARD_SIZE, args.show_corners)

    if len(obj_points) < 8:
        print("[ERROR] Need at least 8 usable images for calibration.")
        return 1

    # --- First image size ---
    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        first_img = cv2.imread(str(image_paths[1]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print("[ERROR] Cannot read any image to determine size")
        return 1
    h, w = first_img.shape
    print(f"[INFO] Image size: {w}x{h}")

    # --- Scale object points ---
    obj_points_scaled = [pts * args.square_size for pts in obj_points]

    # --- Calibrate ---
    print("\nRunning calibration...")
    mtx, dist, rvecs, tvecs, mean_error = calibrate(obj_points_scaled, img_points, (w, h))

    # --- Results ---
    print("\n" + "=" * 55)
    print("  CALIBRATION RESULTS")
    print("=" * 55)
    print(f"\n  Camera Matrix (K):")
    print(f"    fx = {mtx[0, 0]:.3f}")
    print(f"    fy = {mtx[1, 1]:.3f}")
    print(f"    cx = {mtx[0, 2]:.3f}")
    print(f"    cy = {mtx[1, 2]:.3f}")
    print(f"\n  Matrix:")
    print(f"    [[{mtx[0,0]:.3f}, {mtx[0,1]:.3f}, {mtx[0,2]:.3f}],")
    print(f"     [{mtx[1,0]:.3f}, {mtx[1,1]:.3f}, {mtx[1,2]:.3f}],")
    print(f"     [{mtx[2,0]:.3f}, {mtx[2,1]:.3f}, {mtx[2,2]:.3f}]]")
    print(f"\n  Distortion (k1, k2, p1, p2, k3):")
    print(f"    [{', '.join(f'{v:.6f}' for v in dist.ravel())}]")
    print(f"\n  Mean reprojection error: {mean_error:.4f} pixels")
    print(f"  Images used: {len(obj_points)}")
    print(f"  Square size: {args.square_size} mm")

    # --- Quality assessment ---
    error_level = "excellent" if mean_error < 0.3 else \
                  "good"      if mean_error < 0.5 else \
                  "acceptable" if mean_error < 1.0 else "poor"
    print(f"\n  Quality: {error_level}")

    if mean_error >= 1.0:
        print("  => Consider re-capturing with more varied angles and sharper focus.")
    elif mean_error >= 0.5:
        print("  => Acceptable. Remove images with high individual error and re-run.")

    # --- Show individual errors ---
    print("\n  Per-image reprojection errors:")
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points_scaled[i], rvecs[i], tvecs[i], mtx, dist,
        )
        per_img_error = cv2.norm(img_points[i], projected, cv2.NORM_L2)
        flag = "" if per_img_error < mean_error * 1.5 else "  <-- outlier, consider removing"
        print(f"    Image {i:2d}: {per_img_error:.4f} px{flag}")

    # --- Save ---
    output = {
        "calibration_info": {
            "camera": "ZWO ASI",
            "image_width": w,
            "image_height": h,
            "chessboard_corners": list(CHESSBOARD_SIZE),
            "square_size_mm": args.square_size,
            "num_images": len(obj_points),
            "mean_reprojection_error_px": round(mean_error, 4),
            "quality": error_level,
        },
        "camera_matrix": {
            "fx": round(float(mtx[0, 0]), 3),
            "fy": round(float(mtx[1, 1]), 3),
            "cx": round(float(mtx[0, 2]), 3),
            "cy": round(float(mtx[1, 2]), 3),
            "matrix": [[round(float(v), 3) for v in row] for row in mtx.tolist()],
        },
        "distortion_coefficients": {
            "k1": round(float(dist[0, 0]), 6),
            "k2": round(float(dist[0, 1]), 6),
            "p1": round(float(dist[0, 2]), 6),
            "p2": round(float(dist[0, 3]), 6),
            "k3": round(float(dist[0, 4]), 6),
            "vector": [round(float(v), 6) for v in dist.ravel().tolist()],
        },
    }

    output_path = args.output.resolve()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
