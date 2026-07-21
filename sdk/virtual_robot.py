from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
import sys
from typing import List, Optional, Sequence
import time

import numpy as np

from .exceptions import DummyRobotCommandError
from .types import JOINT_COUNT, JointPositions, Pose6D

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import robot_kinematics  # noqa: E402


class VirtualDummyRobot:
    """In-memory digital twin for UI development without physical hardware."""

    def __init__(self, port: str = "VIRTUAL", baudrate: int = 115200, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._connected = False
        self._enabled = False
        self._joints = [0.0] * JOINT_COUNT
        self._last_response = "ok VIRTUAL_READY"
        self._updated_at = time.time()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def list_serial_ports() -> List[str]:
        return []

    def connect(self) -> "VirtualDummyRobot":
        self._connected = True
        self._last_response = "ok VIRTUAL_CONNECTED"
        self._updated_at = time.time()
        return self

    def disconnect(self) -> None:
        self._connected = False
        self._enabled = False
        self._last_response = "ok VIRTUAL_DISCONNECTED"
        self._updated_at = time.time()

    def enable(self) -> str:
        self._require_connected()
        self._enabled = True
        return self._respond("ok !START")

    def disable(self) -> str:
        self._require_connected()
        self._enabled = False
        return self._respond("ok !DISABLE")

    def stop(self) -> str:
        self._require_connected()
        self._enabled = False
        return self._respond("ok !STOP")

    def home(self) -> str:
        self._require_connected()
        self._joints = [0.0] * JOINT_COUNT
        return self._respond("ok !HOME")

    def reset(self) -> str:
        self._require_connected()
        self._enabled = False
        self._joints = [0.0] * JOINT_COUNT
        return self._respond("ok !RESET")

    def get_joint_positions(self) -> JointPositions:
        self._require_connected()
        return JointPositions(values=list(self._joints))

    def get_tool_pose(self) -> Pose6D:
        self._require_connected()
        return self._forward_kinematics(self._joints)

    def move_to_pose(
        self, x: float, y: float, z: float, a: float, b: float, c: float, speed: Optional[float] = None
    ) -> str:
        self._require_connected()
        roll, pitch, yaw = radians(a), radians(b), radians(c)
        Rx = np.array([
            [1, 0, 0],
            [0, cos(roll), -sin(roll)],
            [0, sin(roll), cos(roll)],
        ])
        Ry = np.array([
            [cos(pitch), 0, sin(pitch)],
            [0, 1, 0],
            [-sin(pitch), 0, cos(pitch)],
        ])
        Rz = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw), cos(yaw), 0],
            [0, 0, 1],
        ])
        R_target = Rz @ Ry @ Rx
        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]

        ik_result = robot_kinematics.inverse_kinematics(
            T_target, initial_joints=list(self._joints),
            max_iters=200, alpha=0.2,
        )
        if ik_result is None:
            raise DummyRobotCommandError(
                f"Virtual IK failed to converge for pose ({x:.1f},{y:.1f},{z:.1f})"
            )

        self._joints = [max(-180.0, min(180.0, float(v))) for v in ik_result]
        self._updated_at = time.time()
        return self._respond(f"ok @{x:.4f},{y:.4f},{z:.4f},{a:.4f},{b:.4f},{c:.4f}")

    def move_to_joints(
        self,
        j1: float,
        j2: float,
        j3: float,
        j4: float,
        j5: float,
        j6: float,
        speed: Optional[float] = None,
        sequential: bool = False,
    ) -> str:
        return self.move_joints([j1, j2, j3, j4, j5, j6], speed=speed, sequential=sequential)

    def move_joints(self, joints: Sequence[float], speed: Optional[float] = None, sequential: bool = False) -> str:
        self._require_connected()
        if len(joints) != JOINT_COUNT:
            raise ValueError(f"Expected {JOINT_COUNT} joint targets, got {len(joints)}")
        self._joints = [max(-180.0, min(180.0, float(value))) for value in joints]
        prefix = ">" if sequential else "&"
        payload = ",".join(format(value, "g") for value in self._joints)
        return self._respond(f"ok {prefix}{payload}")

    def move_single_joint(
        self,
        joint_index: int,
        target: Optional[float] = None,
        delta: Optional[float] = None,
        speed: Optional[float] = None,
        sequential: bool = False,
    ) -> str:
        self._require_connected()
        if not 1 <= joint_index <= JOINT_COUNT:
            raise ValueError(f"joint_index must be between 1 and {JOINT_COUNT}")
        if (target is None and delta is None) or (target is not None and delta is not None):
            raise ValueError("Specify exactly one of target or delta")
        joints = list(self._joints)
        joints[joint_index - 1] = float(target) if target is not None else joints[joint_index - 1] + float(delta)
        return self.move_joints(joints, speed=speed, sequential=sequential)

    def send_raw(self, command: str, expect_response: bool = True) -> Optional[str]:
        self._require_connected()
        command = command.strip()
        if not expect_response:
            self._updated_at = time.time()
            return None
        if command == "#GETJPOS":
            return self._respond("ok " + " ".join(format(value, "g") for value in self._joints))
        if command == "#GETLPOS":
            pose = self.get_tool_pose().as_list()
            return self._respond("ok " + " ".join(format(value, "g") for value in pose))
        if command == "!START":
            return self.enable()
        if command == "!DISABLE":
            return self.disable()
        if command == "!STOP":
            return self.stop()
        if command == "!HOME":
            return self.home()
        if command == "!RESET":
            return self.reset()
        if command.startswith("&") or command.startswith(">"):
            values = [float(part) for part in command[1:].split(",")[:JOINT_COUNT]]
            return self.move_joints(values, sequential=command.startswith(">"))
        if command.startswith("@"):
            values = [float(part) for part in command[1:].split(",")[:JOINT_COUNT]]
            if len(values) != JOINT_COUNT:
                raise DummyRobotCommandError(f"Invalid virtual pose command: {command}")
            return self.move_to_pose(*values)
        return self._respond(f"ok {command}")

    def _forward_kinematics(self, joints: Sequence[float]) -> Pose6D:
        base_height = 142.0
        link_1 = 148.0
        link_2 = 122.0
        tool_len = 86.0

        j1, j2, j3, j4, j5, j6 = [radians(value) for value in joints]
        shoulder = j2
        elbow = j2 + j3
        wrist = elbow + j5

        radial = link_1 * cos(shoulder) + link_2 * cos(elbow) + tool_len * cos(wrist)
        x = radial * cos(j1)
        y = radial * sin(j1)
        z = base_height + link_1 * sin(shoulder) + link_2 * sin(elbow) + tool_len * sin(wrist)
        a = joints[0]
        b = joints[1] + joints[2] + joints[4]
        c = joints[3] + joints[5]
        return Pose6D(x=round(x, 3), y=round(y, 3), z=round(z, 3), a=round(a, 3), b=round(b, 3), c=round(c, 3))

    def _require_connected(self) -> None:
        if not self._connected:
            raise DummyRobotCommandError("Virtual robot is not connected")

    def _respond(self, response: str) -> str:
        self._last_response = response
        self._updated_at = time.time()
        return response
