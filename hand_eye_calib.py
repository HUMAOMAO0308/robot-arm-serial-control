from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import serial

from robot_kinematics import forward_kinematics, pose_to_rt
from zwo_camera import ZwoCamera

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHESSBOARD_SIZE = (9, 6)  # internal corners
JOINT_COUNT = 6


# ---------------------------------------------------------------------------
# Hand-eye calibration pose set
# ---------------------------------------------------------------------------
# Varying viewpoints around the scanning pose while keeping chessboard in view.
# Format: [j1, j2, j3, j4, j5, j6] in degrees
HAND_EYE_POSES = [
    [ 0.0, -64.0, 155.0, -1.0,  0.0,  0.0],
    [20.0, -64.0, 155.0, -1.0,  0.0,  0.0],
    [-20.0, -64.0, 155.0, -1.0,  0.0,  0.0],
    [ 0.0, -55.0, 155.0, -1.0,  0.0,  0.0],
    [ 0.0, -70.0, 150.0, -1.0,  0.0,  0.0],
    [20.0, -60.0, 150.0,  5.0,  0.0,  0.0],
    [-20.0, -60.0, 150.0, -5.0,  0.0,  0.0],
    [ 0.0, -64.0, 155.0,  0.0, 10.0,  0.0],
    [ 0.0, -64.0, 155.0,  0.0,-10.0,  0.0],
    [15.0, -60.0, 145.0, -3.0,  5.0,  0.0],
    [-15.0, -60.0, 145.0,  3.0, -5.0,  0.0],
    [ 0.0, -68.0, 155.0,  0.0,  0.0,  5.0],
    [ 0.0, -60.0, 155.0,  0.0,  0.0, -5.0],
    [25.0, -62.0, 150.0, -2.0, -3.0,  0.0],
    [-25.0, -62.0, 150.0,  2.0,  3.0,  0.0],
]


# ---------------------------------------------------------------------------
# Archival type
# ---------------------------------------------------------------------------
@dataclass
class PoseRecord:
    index: int
    joints: list[float]
    R_base_to_ee: list[list[float]]
    t_base_to_ee: list[float]
    R_board_to_cam: list[list[float]]
    t_board_to_cam: list[float]
    reprojection_error: float
    image_path: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hand-eye calibration for ZWO camera on dummy robot arm.",
    )
    p.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--intrinsics", type=Path, required=True,
                   help="Path to camera_intrinsics.json from intrinsic calibration")
    p.add_argument("--square-size", type=float, default=25.0,
                   help="Chessboard square size in mm")
    p.add_argument("--output-dir", type=Path, default=Path("hand_eye_calib"),
                   help="Output directory for images and results")
    p.add_argument("--speed", type=int, default=50, help="Robot move speed")
    p.add_argument("--settle-time", type=float, default=2.0,
                   help="Seconds to wait after move before capturing")
    p.add_argument("--camera-width", type=int, default=1920)
    p.add_argument("--camera-height", type=int, default=1080)
    p.add_argument("--exposure", type=int, default=50000)
    p.add_argument("--gain", type=int, default=50)
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--method", choices=["tsai", "park", "horaud", "andreff", "danillidis"],
                   default="tsai", help="Hand-eye calibration method")
    return p


# ---------------------------------------------------------------------------
# Serial helpers (same protocol as main.py)
# ---------------------------------------------------------------------------
def send_command(ser: serial.Serial, command: bytes, label: str) -> None:
    ser.write(command)
    ser.flush()
    print(f"  [SEND] {label}: {command.decode('utf-8').strip()}")

def format_move(joints: list[float], speed: int) -> bytes:
    return f"&{joints[0]:.1f},{joints[1]:.1f},{joints[2]:.1f},{joints[3]:.1f},{joints[4]:.1f},{joints[5]:.1f},{speed};\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()

    # --- Load intrinsics ---
    if not args.intrinsics.is_file():
        print(f"[ERROR] Intrinsics file not found: {args.intrinsics}")
        return 1
    with args.intrinsics.open() as f:
        calib_data = json.load(f)
    K_data = calib_data["camera_matrix"]
    camera_matrix = np.array([
        [K_data["fx"], 0,             K_data["cx"]],
        [0,            K_data["fy"],  K_data["cy"]],
        [0,            0,             1],
    ], dtype=np.float64)
    dist_coeffs = np.array(calib_data["distortion_coefficients"]["vector"], dtype=np.float64)

    image_width = calib_data["calibration_info"]["image_width"]
    image_height = calib_data["calibration_info"]["image_height"]

    print(f"[INFO] Loaded intrinsics: fx={K_data['fx']:.1f}, fy={K_data['fy']:.1f}, "
          f"cx={K_data['cx']:.1f}, cy={K_data['cy']:.1f}")

    # --- Setup output ---
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # --- Open serial ---
    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=1.0)
        print(f"[INFO] Serial connected: {args.port} @ {args.baudrate}")
    except serial.SerialException as e:
        print(f"[ERROR] Serial: {e}")
        return 1

    # --- Home robot ---
    send_command(ser, b"!START\n", "Home")
    print("[INFO] Waiting for homing (5s)...")
    time.sleep(5.0)

    # --- Open camera ---
    try:
        cam = ZwoCamera()
        cam.open(args.camera_id, args.camera_width, args.camera_height,
                 args.exposure, args.gain)
    except RuntimeError as e:
        print(f"[ERROR] Camera: {e}")
        ser.close()
        return 1

    # Check resolution matches
    if image_width != cam.width or image_height != cam.height:
        print(f"[WARN] Intrinsics calibrated at {image_width}x{image_height}, "
              f"but camera set to {cam.width}x{cam.height}.")
        print(f"       Calibration accuracy may be degraded. "
              f"Recommend using the same resolution as the intrinsics.")

    # --- Prepare chessboard object points ---
    square_size = args.square_size
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:CHESSBOARD_SIZE[0],
                            0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * square_size)

    # --- Collect poses ---
    R_base_to_ee_list = []
    t_base_to_ee_list = []
    R_board_to_cam_list = []
    t_board_to_cam_list = []
    records: list[PoseRecord] = []

    n_poses = len(HAND_EYE_POSES)
    print(f"\n{'='*55}")
    print(f"  Starting hand-eye calibration with {n_poses} poses")
    print(f"  Place chessboard FIXED on the table, in view of the camera")
    print(f"  Method: {args.method}")
    print(f"{'='*55}\n")

    scan_error = False
    try:
        for idx, joints in enumerate(HAND_EYE_POSES):
            print(f"--- Pose {idx + 1}/{n_poses} ---")
            print(f"  Joints: [{', '.join(f'{j:.1f}' for j in joints)}]")

            # Move robot
            cmd = format_move(joints, args.speed)
            send_command(ser, cmd, f"Move to pose {idx+1}")
            print(f"  Settling for {args.settle_time:.1f}s...")
            time.sleep(args.settle_time)

            # Capture frame
            frame = cam.grab_frame()
            img_path = images_dir / f"pose_{idx:02d}.png"
            cv2.imwrite(str(img_path), frame)

            # Detect chessboard
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if not ret:
                print(f"  [FAIL] Chessboard not detected in pose {idx+1} -- skipped")
                continue

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # solvePnP: board in camera frame
            ok, rvec, tvec = cv2.solvePnP(
                objp, corners_refined, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                print(f"  [FAIL] solvePnP failed for pose {idx+1} -- skipped")
                continue

            R_board2cam, _ = cv2.Rodrigues(rvec)
            t_board2cam = tvec.reshape(3)

            # Compute reprojection error
            projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            reproj_err = cv2.norm(corners_refined, projected, cv2.NORM_L2) / np.sqrt(len(corners_refined))

            # FK: base to ee
            T_ee_in_base = forward_kinematics(joints)
            R_b2e, t_b2e = pose_to_rt(T_ee_in_base)

            R_base_to_ee_list.append(R_b2e)
            t_base_to_ee_list.append(t_b2e)
            R_board_to_cam_list.append(R_board2cam)
            t_board_to_cam_list.append(t_board2cam)

            record = PoseRecord(
                index=idx,
                joints=joints,
                R_base_to_ee=R_b2e.tolist(),
                t_base_to_ee=t_b2e.tolist(),
                R_board_to_cam=R_board2cam.tolist(),
                t_board_to_cam=t_board2cam.tolist(),
                reprojection_error=round(reproj_err, 4),
                image_path=str(img_path),
            )
            records.append(record)
            print(f"  [OK] Chessboard detected, reprojection error: {reproj_err:.4f} px")
    except Exception as e:
        print(f"[ERROR] Calibration loop failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        scan_error = True

    print(f"\n[INFO] Valid poses collected: {len(R_base_to_ee_list)} / {n_poses}")

    if len(R_base_to_ee_list) < 8:
        print("[ERROR] Need at least 8 valid poses for hand-eye calibration")
        can_solve = False
    else:
        can_solve = not scan_error

    # --- Solve hand-eye ---
    if can_solve:
        method_map = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "danillidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        print(f"\n[INFO] Solving hand-eye calibration ({args.method})...")
        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            R_base_to_ee_list, t_base_to_ee_list,
            R_board_to_cam_list, t_board_to_cam_list,
            method=method_map[args.method],
        )

        T_cam_to_ee = np.eye(4)
        T_cam_to_ee[:3, :3] = R_cam2ee
        T_cam_to_ee[:3, 3] = t_cam2ee.ravel()

        print("\n" + "=" * 55)
        print("  HAND-EYE CALIBRATION RESULT")
        print("  Camera → End-Effector transform (T_cam_to_ee):")
        print("=" * 55)
        print()
        for row in T_cam_to_ee:
            print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}]")
        print()
        print(f"  Translation: ({t_cam2ee[0,0]:.4f}, {t_cam2ee[1,0]:.4f}, {t_cam2ee[2,0]:.4f}) meters")
        print(f"  (i.e., camera is {t_cam2ee[0,0]*1000:.1f}, {t_cam2ee[1,0]*1000:.1f}, "
              f"{t_cam2ee[2,0]*1000:.1f} mm from end-effector flange)")

        # --- Save results ---
        result = {
            "method": args.method,
            "num_poses_used": len(R_base_to_ee_list),
            "num_poses_attempted": n_poses,
            "T_cam_to_ee": {
                "rotation": R_cam2ee.tolist(),
                "translation": t_cam2ee.ravel().tolist(),
                "matrix_4x4": T_cam_to_ee.tolist(),
            },
            "pose_records": [asdict(r) for r in records],
            "camera_intrinsics_source": str(args.intrinsics.resolve()),
            "chessboard": {
                "internal_corners": list(CHESSBOARD_SIZE),
                "square_size_mm": square_size,
            },
        }

        result_path = output_dir / "hand_eye_result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] {result_path}")

    # --- Cleanup ---
    # Return to safe pose
    try:
        send_command(ser, format_move([0.0, -64.0, 155.0, -1.0, 0.0, 0.0], args.speed),
                     "Return to safe pose")
        time.sleep(2.0)
    except Exception:
        pass

    ser.close()
    cam.close()
    print("[DONE] Hand-eye calibration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
