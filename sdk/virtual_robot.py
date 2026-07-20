from __future__ import annotations

from math import cos, radians, sin
from typing import List, Optional, Sequence
import time

from .exceptions import DummyRobotCommandError
from .types import JOINT_COUNT, JointPositions, Pose6D


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
        self._joints[0] = max(-180.0, min(180.0, a))
        self._joints[1] = max(-180.0, min(180.0, b * 0.5))
        self._joints[2] = max(-180.0, min(180.0, c * 0.5))
        self._joints[3] = max(-180.0, min(180.0, a * 0.25))
        self._joints[4] = max(-180.0, min(180.0, b * 0.5))
        self._joints[5] = max(-180.0, min(180.0, c * 0.5))
        self._updated_at = time.time()
        return self._respond(f"ok @{x:g},{y:g},{z:g},{a:g},{b:g},{c:g}")

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
