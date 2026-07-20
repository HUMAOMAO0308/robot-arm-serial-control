from __future__ import annotations

from typing import List, Optional
import logging

import serial
import serial.tools.list_ports

from .exceptions import DummyRobotConnectionError


logger = logging.getLogger(__name__)


class SerialTransport:
    """Thin transport wrapper around the robot ASCII protocol."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        except serial.SerialException as exc:
            raise DummyRobotConnectionError(
                f"Failed to open serial port {self.port} at {self.baudrate} baud"
            ) from exc
        logger.info("Connected to %s at %s baud", self.port, self.baudrate)

    def disconnect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            logger.info("Disconnected from %s", self.port)
        self._serial = None

    def send(self, command: str, expect_response: bool = True) -> Optional[str]:
        if not self.is_connected or self._serial is None:
            raise DummyRobotConnectionError("Device not connected. Call connect() first.")

        payload = f"{command}\n".encode("utf-8")
        self._serial.write(payload)
        logger.debug("TX: %s", command)

        if not expect_response:
            return None

        try:
            response = self._serial.readline().decode("utf-8", errors="replace").strip()
        except serial.SerialException as exc:
            raise DummyRobotConnectionError("Failed to read response from device") from exc

        logger.debug("RX: %s", response)
        return response

    @staticmethod
    def list_ports() -> List[str]:
        return [port.device for port in serial.tools.list_ports.comports()]
