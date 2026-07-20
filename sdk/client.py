from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import logging
import time

from .camera import CameraInterface, CameraPoseSynchronizer
from .exceptions import DummyRobotCommandError, DummyRobotResponseError
from .structured_light_tasks import StructuredLightScanResult, StructuredLightScanTask
from .sensors import SensorInterface
from .transport import SerialTransport
from .types import JOINT_COUNT, JointPositions, Pose6D, RobotAction, RobotObservation, RobotStateSnapshot, SyncedCameraFrame


logger = logging.getLogger(__name__)


class DummyRobot:
    """High-level Python client for the Dummy robot ASCII protocol."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0) -> None:
        self.transport = SerialTransport(port=port, baudrate=baudrate, timeout=timeout)
        self._sensors: Dict[str, SensorInterface] = {}
        self._camera_synchronizer = CameraPoseSynchronizer()

    @property
    def port(self) -> str:
        return self.transport.port

    @property
    def is_connected(self) -> bool:
        return self.transport.is_connected

    def connect(self) -> "DummyRobot":
        self.transport.connect()
        return self

    def disconnect(self) -> None:
        self.transport.disconnect()

    def __enter__(self) -> "DummyRobot":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    @staticmethod
    def list_serial_ports() -> List[str]:
        return SerialTransport.list_ports()

    def send_raw(self, command: str, expect_response: bool = True) -> Optional[str]:
        return self.transport.send(command, expect_response=expect_response)

    def register_sensor(self, sensor: SensorInterface) -> None:
        self._sensors[sensor.name] = sensor

    def register_camera(self, camera: CameraInterface) -> None:
        self.register_sensor(camera)

    def unregister_sensor(self, name: str) -> None:
        self._sensors.pop(name, None)

    def list_sensors(self) -> List[str]:
        return sorted(self._sensors.keys())

    def enable(self) -> str:
        return self._require_ok("!START")

    def stop(self) -> str:
        return self._require_ok("!STOP")

    def disable(self) -> str:
        return self._require_ok("!DISABLE")

    def home(self) -> str:
        return self._require_ok("!HOME")

    def reset(self) -> str:
        return self._require_ok("!RESET")

    def calibrate(self) -> str:
        return self._require_ok("!CALIBRATION")

    def set_kp(self, motor_id: int, value: Union[int, float]) -> str:
        self._validate_motor_id(motor_id)
        return self._require_ok(f"#SET_DCE_KP {motor_id} {value}")

    def set_ki(self, motor_id: int, value: Union[int, float]) -> str:
        self._validate_motor_id(motor_id)
        return self._require_ok(f"#SET_DCE_KI {motor_id} {value}")

    def set_kd(self, motor_id: int, value: Union[int, float]) -> str:
        self._validate_motor_id(motor_id)
        return self._require_ok(f"#SET_DCE_KD {motor_id} {value}")

    def reboot_motor(self, motor_id: int) -> str:
        self._validate_motor_id(motor_id)
        return self._require_ok(f"#REBOOT {motor_id}")

    def set_command_mode(self, mode: int) -> str:
        return self._require_ok(f"#CMDMODE {mode}")

    def get_joint_positions(self) -> JointPositions:
        response = self._require_prefix("#GETJPOS", prefix="ok")
        values = self._parse_float_list(response, expected=JOINT_COUNT)
        return JointPositions(values=values)

    def get_tool_pose(self) -> Pose6D:
        response = self._require_prefix("#GETLPOS", prefix="ok")
        values = self._parse_float_list(response, expected=JOINT_COUNT)
        return Pose6D(*values)

    def move_to_pose(
        self, x: float, y: float, z: float, a: float, b: float, c: float, speed: Optional[float] = None
    ) -> str:
        values = [x, y, z, a, b, c]
        if speed is not None:
            values.append(speed)
        command = "@" + ",".join(self._format_number(value) for value in values)
        return self._send_motion(command)

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
        values = [j1, j2, j3, j4, j5, j6]
        if speed is not None:
            values.append(speed)
        prefix = ">" if sequential else "&"
        command = prefix + ",".join(self._format_number(value) for value in values)
        return self._send_motion(command)

    def move_joints(self, joints: Sequence[float], speed: Optional[float] = None, sequential: bool = False) -> str:
        if len(joints) != JOINT_COUNT:
            raise ValueError(f"Expected {JOINT_COUNT} joint targets, got {len(joints)}")
        return self.move_to_joints(*joints, speed=speed, sequential=sequential)

    def move_single_joint(
        self,
        joint_index: int,
        target: Optional[float] = None,
        delta: Optional[float] = None,
        speed: Optional[float] = None,
        sequential: bool = False,
    ) -> str:
        if not 1 <= joint_index <= JOINT_COUNT:
            raise ValueError(f"joint_index must be between 1 and {JOINT_COUNT}")
        if (target is None and delta is None) or (target is not None and delta is not None):
            raise ValueError("Specify exactly one of target or delta")
        joints = self.get_joint_positions().as_list()
        joints[joint_index - 1] = float(target) if target is not None else joints[joint_index - 1] + float(delta)
        return self.move_joints(joints, speed=speed, sequential=sequential)

    def send_gcode(self, gcode: str) -> Optional[str]:
        return self.send_raw(gcode)

    def get_observation(self, include_pose: bool = True, include_sensors: bool = True) -> RobotObservation:
        observation = RobotObservation(timestamp=time.time(), joints=self.get_joint_positions())
        if include_pose:
            observation.tool_pose = self.get_tool_pose()
        if include_sensors:
            for name, sensor in self._sensors.items():
                frame = sensor.read()
                if frame is not None:
                    observation.sensors[name] = frame
        return observation

    def get_lerobot_observation(self, include_pose: bool = True) -> Dict[str, Any]:
        observation = self.get_observation(include_pose=include_pose, include_sensors=False)
        return observation.as_dict()

    def get_state_snapshot(self, include_pose: bool = True) -> RobotStateSnapshot:
        joints = self.get_joint_positions()
        pose = self.get_tool_pose() if include_pose else None
        return RobotStateSnapshot(timestamp=time.time(), joints=joints, tool_pose=pose)

    def capture_camera_frame(self, camera_name: str) -> SyncedCameraFrame:
        camera = self._get_camera_sensor(camera_name)
        return self._camera_synchronizer.sync_frame(self, camera)

    def execute_structured_light_scan_task(
        self,
        task: StructuredLightScanTask,
        writer: Optional[Any] = None,
    ) -> StructuredLightScanResult:
        return task.execute(self, writer=writer)

    def execute_action(self, action: Union[RobotAction, Dict[str, Any], Sequence[float]]) -> str:
        normalized = self._normalize_action(action)
        if normalized.joint_positions is not None:
            return self.move_joints(normalized.joint_positions, speed=normalized.speed)
        if normalized.tool_pose is not None:
            pose = normalized.tool_pose
            return self.move_to_pose(*pose.as_list(), speed=normalized.speed)
        raise ValueError("Action must include joint_positions or tool_pose")

    def _normalize_action(self, action: Union[RobotAction, Dict[str, Any], Sequence[float]]) -> RobotAction:
        if isinstance(action, RobotAction):
            return action
        if isinstance(action, dict):
            tool_pose = action.get("tool_pose")
            if tool_pose is not None and not isinstance(tool_pose, Pose6D):
                tool_pose = Pose6D(*tool_pose)
            return RobotAction(
                joint_positions=action.get("joint_positions"),
                tool_pose=tool_pose,
                speed=action.get("speed"),
            )
        if isinstance(action, Sequence):
            return RobotAction(joint_positions=list(action))
        raise TypeError(f"Unsupported action type: {type(action)!r}")

    def _get_camera_sensor(self, name: str) -> CameraInterface:
        sensor = self._sensors.get(name)
        if sensor is None:
            raise KeyError(f"No registered sensor named '{name}'")
        if not isinstance(sensor, CameraInterface):
            raise TypeError(f"Sensor '{name}' is not a camera sensor")
        return sensor

    def _send_motion(self, command: str) -> str:
        response = self.send_raw(command)
        if response is None:
            raise DummyRobotCommandError(f"No response for motion command: {command}")
        return response

    def _require_ok(self, command: str) -> str:
        response = self.send_raw(command)
        if response is None:
            raise DummyRobotCommandError(f"No response for command: {command}")
        lowered = response.lower()
        if lowered.startswith("ok") or "ok" in lowered:
            return response
        raise DummyRobotCommandError(f"Command failed: {command} -> {response}")

    def _require_prefix(self, command: str, prefix: str) -> str:
        response = self.send_raw(command)
        if response is None:
            raise DummyRobotCommandError(f"No response for command: {command}")
        if not response.startswith(prefix):
            raise DummyRobotCommandError(f"Unexpected response for {command}: {response}")
        return response

    def _parse_float_list(self, response: str, expected: int) -> List[float]:
        try:
            values = [float(item) for item in response.split()[1:]]
        except ValueError as exc:
            raise DummyRobotResponseError(f"Failed to parse numeric response: {response}") from exc
        if len(values) != expected:
            raise DummyRobotResponseError(
                f"Expected {expected} numeric values, got {len(values)} from response: {response}"
            )
        return values

    @staticmethod
    def _validate_motor_id(motor_id: int) -> None:
        if not 1 <= motor_id <= JOINT_COUNT:
            raise ValueError(f"motor_id must be between 1 and {JOINT_COUNT}")

    @staticmethod
    def _format_number(value: Union[int, float]) -> str:
        return format(float(value), "g")
