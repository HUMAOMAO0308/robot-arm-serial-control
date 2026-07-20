from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import time

import numpy as np


JOINT_COUNT = 6
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, JOINT_COUNT + 1))


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.a, self.b, self.c]


@dataclass(frozen=True)
class JointPositions:
    values: List[float]

    def __post_init__(self) -> None:
        if len(self.values) != JOINT_COUNT:
            raise ValueError(f"Expected {JOINT_COUNT} joints, got {len(self.values)}")

    def as_list(self) -> List[float]:
        return list(self.values)

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(JOINT_NAMES, self.values))


@dataclass(frozen=True)
class SensorFrame:
    name: str
    timestamp: float
    data: Dict[str, Any]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: List[float] = field(default_factory=list)
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def camera_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist_coeffs": list(self.dist_coeffs),
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


@dataclass(frozen=True)
class Plane3D:
    normal_xyz: List[float]
    offset: float

    def __post_init__(self) -> None:
        if len(self.normal_xyz) != 3:
            raise ValueError("Plane normal must contain exactly 3 elements")

    def normal_vector(self) -> np.ndarray:
        vector = np.asarray(self.normal_xyz, dtype=np.float64)
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            raise ValueError("Plane normal must not be zero")
        return vector / magnitude

    def as_dict(self) -> Dict[str, Any]:
        return {"normal_xyz": list(self.normal_xyz), "offset": self.offset}


@dataclass(frozen=True)
class RigidTransform:
    rotation: List[List[float]]
    translation_xyz: List[float]
    from_frame: str = ""
    to_frame: str = ""

    def __post_init__(self) -> None:
        if len(self.rotation) != 3 or any(len(row) != 3 for row in self.rotation):
            raise ValueError("Rotation must be a 3x3 matrix")
        if len(self.translation_xyz) != 3:
            raise ValueError("Translation must contain exactly 3 elements")

    def rotation_matrix(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float64)

    def translation_vector(self) -> np.ndarray:
        return np.asarray(self.translation_xyz, dtype=np.float64)

    def matrix4x4(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation_matrix()
        matrix[:3, 3] = self.translation_vector()
        return matrix

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "translation_xyz": list(self.translation_xyz),
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
        }


@dataclass(frozen=True)
class CameraFrame:
    timestamp: float
    image: np.ndarray
    frame_id: str = "camera"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "shape": list(self.image.shape),
            "dtype": str(self.image.dtype),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SyncedCameraFrame:
    frame: CameraFrame
    pose_before: "RobotStateSnapshot"
    pose_after: "RobotStateSnapshot"
    pose_midpoint: "RobotStateSnapshot"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame.as_dict(),
            "pose_before": self.pose_before.as_dict(),
            "pose_after": self.pose_after.as_dict(),
            "pose_midpoint": self.pose_midpoint.as_dict(),
            "sync_latency_s": self.pose_after.timestamp - self.pose_before.timestamp,
        }


@dataclass(frozen=True)
class LaserLineObservation:
    timestamp: float
    uv_points: List[List[float]]
    intensities: Optional[List[float]] = None
    image_shape: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "uv_points": self.uv_points,
            "point_count": len(self.uv_points),
            "image_shape": self.image_shape,
        }
        if self.intensities is not None:
            payload["intensities"] = self.intensities
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class ReconstructedPointCloudFrame:
    timestamp: float
    points_xyz: List[List[float]]
    source_observation: LaserLineObservation
    frame_id: str = "camera"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "points_xyz": self.points_xyz,
            "point_count": len(self.points_xyz),
            "frame_id": self.frame_id,
            "source_observation": self.source_observation.as_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RobotStateSnapshot:
    timestamp: float
    joints: JointPositions
    tool_pose: Optional[Pose6D] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "joint_positions": self.joints.as_list(),
            "joint_positions_by_name": self.joints.as_dict(),
        }
        if self.tool_pose is not None:
            payload["tool_pose"] = self.tool_pose.as_list()
        return payload


@dataclass
class RobotObservation:
    timestamp: float = field(default_factory=time.time)
    joints: Optional[JointPositions] = None
    tool_pose: Optional[Pose6D] = None
    sensors: Dict[str, SensorFrame] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"timestamp": self.timestamp}
        if self.joints is not None:
            payload["joint_positions"] = self.joints.as_list()
            payload["joint_positions_by_name"] = self.joints.as_dict()
        if self.tool_pose is not None:
            payload["tool_pose"] = self.tool_pose.as_list()
        if self.sensors:
            payload["sensors"] = {
                name: {"timestamp": frame.timestamp, "data": frame.data}
                for name, frame in self.sensors.items()
            }
        return payload


@dataclass(frozen=True)
class RobotAction:
    joint_positions: Optional[List[float]] = None
    tool_pose: Optional[Pose6D] = None
    speed: Optional[float] = None

    def __post_init__(self) -> None:
        if self.joint_positions is not None and len(self.joint_positions) != JOINT_COUNT:
            raise ValueError(f"Expected {JOINT_COUNT} joint targets, got {len(self.joint_positions)}")
        if self.joint_positions is not None and self.tool_pose is not None:
            raise ValueError("Specify either joint_positions or tool_pose, not both")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
