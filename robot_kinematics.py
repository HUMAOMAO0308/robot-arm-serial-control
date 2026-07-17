from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# URDF-derived kinematic parameters for the dummy-ros2 6-DOF robot arm
# ---------------------------------------------------------------------------

JOINT_COUNT = 6
LINK_NAMES = [
    "base_link",
    "link1_1_1",
    "link2_1_1",
    "link3_1_1",
    "link4_1_1",
    "link5_1_1",
    "link6_1_1",
]
JOINT_NAMES = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]

# Joint origins in the parent link frame (meters), from URDF <origin xyz=...>
JOINT_ORIGINS: np.ndarray = np.array([
    [-0.019225, -0.000523,  0.100684],
    [-0.015500,  0.035000,  0.028500],
    [ 0.036650,  0.000000,  0.146000],
    [-0.018550, -0.010000,  0.052000],
    [ 0.016800,  0.127000,  0.000000],
    [-0.018506,  0.077000, -0.000106],
], dtype=np.float64)

# Joint rotation axes in the parent link frame, from URDF <axis xyz=...>
JOINT_AXES: np.ndarray = np.array([
    [ 0.0,  0.0, -1.0],
    [-1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
], dtype=np.float64)

# The world_joint (fixed) has rpy = (pi, pi, 0) and origin (0,0,0)
T_WORLD_TO_BASE: np.ndarray = np.array([
    [-1,  0,  0, 0],
    [ 0, -1,  0, 0],
    [ 0,  0,  1, 0],
    [ 0,  0,  0, 1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix for rotation around *axis* by *angle_rad*."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    v = 1.0 - c
    x, y, z = axis
    return np.array([
        [c + v*x*x,    v*x*y - s*z,  v*x*z + s*y],
        [v*y*x + s*z,  c + v*y*y,    v*y*z - s*x],
        [v*z*x - s*y,  v*z*y + s*x,  c + v*z*z],
    ], dtype=np.float64)


def _pose_4x4(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from 3x3 rotation and 3x1 translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Kinematic results
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """A 6-DOF pose with rotation matrix and translation vector."""
    R: np.ndarray   # 3x3
    t: np.ndarray   # (3,)

    def matrix(self) -> np.ndarray:
        """Return 4x4 homogeneous transform."""
        return _pose_4x4(self.R, self.t)

    @property
    def x(self) -> float: return self.t[0]
    @property
    def y(self) -> float: return self.t[1]
    @property
    def z(self) -> float: return self.t[2]

    def __repr__(self) -> str:
        return (
            f"Pose(x={self.t[0]:.4f}, y={self.t[1]:.4f}, z={self.t[2]:.4f})"
        )


@dataclass
class KinematicChain:
    """Poses of every link in the chain for one joint configuration."""
    joint_angles_deg: List[float]
    link_poses: List[Pose]   # 7 poses: base_link through link6_1_1

    @property
    def ee_pose(self) -> Pose:
        """End-effector (link6_1_1) pose."""
        return self.link_poses[-1]

    def cam_pose(self, hand_eye_result_path: Optional[Path] = None,
                 T_cam_to_ee: Optional[np.ndarray] = None) -> Pose:
        """Camera pose in world frame, using hand-eye transform.

        Provide either a path to hand_eye_result.json or a 4x4 T_cam_to_ee matrix.
        """
        if T_cam_to_ee is None and hand_eye_result_path is not None:
            with hand_eye_result_path.open() as f:
                data = json.load(f)
            T_cam_to_ee = np.array(data["T_cam_to_ee"]["matrix_4x4"], dtype=np.float64)
        if T_cam_to_ee is None:
            raise ValueError("One of hand_eye_result_path or T_cam_to_ee must be provided")

        T_world_to_cam = self.ee_pose.matrix() @ T_cam_to_ee
        return Pose(R=T_world_to_cam[:3, :3].copy(), t=T_world_to_cam[:3, 3].copy())

    def __repr__(self) -> str:
        return f"KinematicChain(joints={self.joint_angles_deg}, ee={self.ee_pose})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward_kinematics(
    joint_angles: List[float],
    degrees: bool = True,
) -> np.ndarray:
    """Compute end-effector 4x4 homogeneous transform.

    Args:
        joint_angles: [j1, j2, j3, j4, j5, j6] in degrees (default) or radians.
        degrees: if True, input is in degrees; otherwise radians.

    Returns:
        4x4 T_world_to_ee matrix.
    """
    T = T_WORLD_TO_BASE.copy()
    for i in range(JOINT_COUNT):
        angle = math.radians(joint_angles[i]) if degrees else float(joint_angles[i])
        R = _rodrigues(JOINT_AXES[i], angle)
        T_i = _pose_4x4(R, JOINT_ORIGINS[i])
        T = T @ T_i
    return T


def forward_kinematics_chain(
    joint_angles: List[float],
    degrees: bool = True,
) -> KinematicChain:
    """Compute the pose of EVERY link along the kinematic chain.

    Returns a KinematicChain with 7 link poses (base_link through link6_1_1).
    """
    link_poses: List[Pose] = []
    T = T_WORLD_TO_BASE.copy()

    # world → base_link
    link_poses.append(Pose(R=T[:3, :3].copy(), t=T[:3, 3].copy()))

    for i in range(JOINT_COUNT):
        angle = math.radians(joint_angles[i]) if degrees else float(joint_angles[i])
        R = _rodrigues(JOINT_AXES[i], angle)
        T_i = _pose_4x4(R, JOINT_ORIGINS[i])
        T = T @ T_i
        link_poses.append(Pose(R=T[:3, :3].copy(), t=T[:3, 3].copy()))

    return KinematicChain(
        joint_angles_deg=list(joint_angles) if degrees
        else [math.degrees(a) for a in joint_angles],
        link_poses=link_poses,
    )


def pose_to_rt(pose_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 matrix into (3x3 R, 3x1 t)."""
    return pose_4x4[:3, :3].copy(), pose_4x4[:3, 3].copy()


def pose_to_rpy(pose_4x4: np.ndarray) -> Tuple[float, float, float]:
    """Extract roll-pitch-yaw (XYZ fixed angles) from a 4x4 matrix.

    Returns (roll, pitch, yaw) in radians.
    """
    R = pose_4x4[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# IK (placeholder) — Pseudo-Inverse Jacobian method
# ---------------------------------------------------------------------------

def inverse_kinematics(
    target_pose: np.ndarray,
    initial_joints: Optional[List[float]] = None,
    max_iters: int = 200,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-4,
    alpha: float = 0.1,
) -> Optional[List[float]]:
    """Numerical IK using pseudo-inverse Jacobian.

    Args:
        target_pose: 4x4 target end-effector transform.
        initial_joints: starting joint angles in degrees (default: all zero).
        max_iters: maximum iterations.
        tol_pos: position tolerance in meters.
        tol_rot: rotation tolerance in radians.
        alpha: step size.

    Returns:
        Joint angles in degrees, or None if it fails to converge.
    """
    joints = np.array(initial_joints or [0.0] * JOINT_COUNT, dtype=np.float64)
    target_R = target_pose[:3, :3]
    target_t = target_pose[:3, 3]

    for iteration in range(max_iters):
        T = forward_kinematics(joints.tolist())
        current_R = T[:3, :3]
        current_t = T[:3, 3]

        pos_err = target_t - current_t
        # Orientation error via skew-symmetric
        R_err = target_R @ current_R.T
        rot_err = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) * 0.5

        if np.linalg.norm(pos_err) < tol_pos and np.linalg.norm(rot_err) < tol_rot:
            return joints.tolist()

        J = _compute_jacobian(joints)
        error = np.concatenate([pos_err, rot_err])
        try:
            J_pinv = np.linalg.pinv(J)
            delta = alpha * (J_pinv @ error)
        except np.linalg.LinAlgError:
            return None

        joints += delta

    return None


def _compute_jacobian(joints: np.ndarray, delta: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian (position + orientation, 6x6)."""
    J = np.zeros((6, JOINT_COUNT), dtype=np.float64)
    T0 = forward_kinematics(joints.tolist())
    p0 = T0[:3, 3]

    for i in range(JOINT_COUNT):
        joints_perturbed = joints.copy()
        joints_perturbed[i] += delta
        T1 = forward_kinematics(joints_perturbed.tolist())
        p1 = T1[:3, 3]
        J[:3, i] = (p1 - p0) / delta
        Rn = T1[:3, :3] @ T0[:3, :3].T
        J[3:, i] = np.array([
            Rn[2, 1] - Rn[1, 2],
            Rn[0, 2] - Rn[2, 0],
            Rn[1, 0] - Rn[0, 1],
        ]) * 0.5 / delta

    return J


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test FK at several poses
    test_poses = [
        [0, 0, 0, 0, 0, 0],
        [0, -64, 155, -1, 0, 0],
        [90, -64, 155, -1, 0, 0],
        [-90, -64, 155, -1, 0, 0],
        [0, -90, 90, 0, 0, 0],
    ]

    print("=" * 65)
    print("  DUMMY-ROS2 FORWARD KINEMATICS TEST")
    print("=" * 65)

    for joints in test_poses:
        T = forward_kinematics(joints)
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        roll, pitch, yaw = pose_to_rpy(T)
        print(f"\n  Joints: {joints}")
        print(f"  EE Position:  x={x*1000:.1f}  y={y*1000:.1f}  z={z*1000:.1f} mm")
        print(f"  EE RPY (rad): r={roll:.4f}  p={pitch:.4f}  y={yaw:.4f}")
        print(f"  EE RPY (deg): r={math.degrees(roll):.1f}  p={math.degrees(pitch):.1f}  "
              f"y={math.degrees(yaw):.1f}")

    # Test full chain
    chain = forward_kinematics_chain([0, -64, 155, -1, 0, 0])
    print(f"\n  Chain poses ({len(chain.link_poses)} links):")
    for name, pose in zip(LINK_NAMES, chain.link_poses):
        print(f"    {name:12s} → ({pose.x*1000:.1f}, {pose.y*1000:.1f}, {pose.z*1000:.1f}) mm")
