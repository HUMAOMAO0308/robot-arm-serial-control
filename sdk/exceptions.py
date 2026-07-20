class DummyRobotError(Exception):
    """Base exception for the DummyRobot SDK."""


class DummyRobotConnectionError(DummyRobotError):
    """Raised when the serial connection is unavailable or fails."""


class DummyRobotCommandError(DummyRobotError):
    """Raised when the device rejects a command or returns an unexpected response."""


class DummyRobotResponseError(DummyRobotError):
    """Raised when a device response cannot be parsed."""
