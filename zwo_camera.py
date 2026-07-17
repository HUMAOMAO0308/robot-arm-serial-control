from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from types import TracebackType
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# ASI SDK constants
# ---------------------------------------------------------------------------
ASI_SUCCESS = 0
ASI_GAIN = 0
ASI_EXPOSURE = 1
ASI_FALSE = 0
ASI_IMG_RAW8 = 0

DEFAULT_SDK_PATH = os.path.realpath(
    "/home/hu/桌面/ASI_linux_mac_SDK_V1.41/lib/x64/libASICamera2.so"
)


# ---------------------------------------------------------------------------
# Camera info data class
# ---------------------------------------------------------------------------
@dataclass
class CameraInfo:
    name: str
    camera_id: int
    max_width: int
    max_height: int
    is_color: bool
    pixel_size_um: float
    bit_depth: int
    is_usb3: bool
    is_cooled: bool


# ---------------------------------------------------------------------------
# ZWO Camera class
# ---------------------------------------------------------------------------
class ZwoCamera:
    """High-level interface for ZWO ASI cameras via the ASI SDK (v1.41+).

    Usage:
        cam = ZwoCamera()
        cam.open(camera_id=0, width=1920, height=1080, exposure_us=50000, gain=50)

        # Snap a single raw frame
        frame: np.ndarray = cam.grab_frame()

        # With context manager (auto-close)
        with ZwoCamera.open(camera_id=0, width=640, height=480) as cam:
            frame = cam.grab_frame()
    """

    # ------------------------------------------------------------------
    # ASI_CAMERA_INFO struct
    # ------------------------------------------------------------------
    class _ASI_CAMERA_INFO(ctypes.Structure):
        _fields_ = [
            ("Name", ctypes.c_char * 64),
            ("CameraID", ctypes.c_int),
            ("MaxHeight", ctypes.c_long),
            ("MaxWidth", ctypes.c_long),
            ("IsColorCam", ctypes.c_int),
            ("BayerPattern", ctypes.c_int),
            ("SupportedBins", ctypes.c_int * 16),
            ("SupportedVideoFormat", ctypes.c_int * 8),
            ("PixelSize", ctypes.c_double),
            ("MechanicalShutter", ctypes.c_int),
            ("ST4Port", ctypes.c_int),
            ("IsCoolerCam", ctypes.c_int),
            ("IsUSB3Host", ctypes.c_int),
            ("IsUSB3Camera", ctypes.c_int),
            ("ElecPerADU", ctypes.c_float),
            ("BitDepth", ctypes.c_int),
            ("IsTriggerCam", ctypes.c_int),
            ("Unused", ctypes.c_char * 16),
        ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, sdk_path: str = DEFAULT_SDK_PATH) -> None:
        self._lib = ctypes.CDLL(sdk_path)
        self._camera_id: int = -1
        self._opened = False
        self._width: int = 0
        self._height: int = 0
        self._exposure_us: int = 0
        self._gain: int = 0
        self._setup_signatures()

    def _setup_signatures(self) -> None:
        lib = self._lib

        lib.ASIGetNumOfConnectedCameras.restype = ctypes.c_int

        lib.ASIGetCameraProperty.restype = ctypes.c_int
        lib.ASIGetCameraProperty.argtypes = [ctypes.c_void_p, ctypes.c_int]

        lib.ASIOpenCamera.restype = ctypes.c_int
        lib.ASIOpenCamera.argtypes = [ctypes.c_int]

        lib.ASIInitCamera.restype = ctypes.c_int
        lib.ASIInitCamera.argtypes = [ctypes.c_int]

        lib.ASISetControlValue.restype = ctypes.c_int
        lib.ASISetControlValue.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_int,
        ]

        lib.ASISetROIFormat.restype = ctypes.c_int
        lib.ASISetROIFormat.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        lib.ASIStartVideoCapture.restype = ctypes.c_int
        lib.ASIStartVideoCapture.argtypes = [ctypes.c_int]

        lib.ASIStopVideoCapture.restype = ctypes.c_int
        lib.ASIStopVideoCapture.argtypes = [ctypes.c_int]

        lib.ASIGetVideoData.restype = ctypes.c_int
        lib.ASIGetVideoData.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
        ]

        lib.ASICloseCamera.restype = ctypes.c_int
        lib.ASICloseCamera.argtypes = [ctypes.c_int]

    @staticmethod
    def _check(ret: int, label: str) -> None:
        if ret != ASI_SUCCESS:
            raise RuntimeError(f"[ZWO] {label} failed (code={ret})")

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------
    def open(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        exposure_us: int = 50000,
        gain: int = 50,
    ) -> "ZwoCamera":
        self._camera_id = camera_id
        self._width = width
        self._height = height

        num = self._lib.ASIGetNumOfConnectedCameras()
        if num <= 0:
            raise RuntimeError("No ZWO camera detected")
        print(f"[ZWO] {num} camera(s) connected")

        sys = self._lib
        cid = camera_id

        # Get info
        info = self._ASI_CAMERA_INFO()
        self._check(sys.ASIGetCameraProperty(ctypes.byref(info), cid), "ASIGetCameraProperty")
        self._info = CameraInfo(
            name=info.Name.decode().strip(),
            camera_id=info.CameraID,
            max_width=info.MaxWidth,
            max_height=info.MaxHeight,
            is_color=bool(info.IsColorCam),
            pixel_size_um=info.PixelSize,
            bit_depth=info.BitDepth,
            is_usb3=bool(info.IsUSB3Camera),
            is_cooled=bool(info.IsCoolerCam),
        )
        print(f"[ZWO] Found: {self._info.name} ({info.MaxWidth}x{info.MaxHeight}, "
              f"{info.PixelSize:.2f}um pixel)")

        # Open, init, configure
        self._check(sys.ASIOpenCamera(cid), "ASIOpenCamera")
        self._check(sys.ASIInitCamera(cid), "ASIInitCamera")

        self._check(
            sys.ASISetControlValue(cid, ASI_EXPOSURE, exposure_us, ASI_FALSE),
            "ASISetControlValue(EXPOSURE)",
        )
        self._check(
            sys.ASISetControlValue(cid, ASI_GAIN, gain, ASI_FALSE),
            "ASISetControlValue(GAIN)",
        )
        self._exposure_us = exposure_us
        self._gain = gain

        self._check(
            sys.ASISetROIFormat(cid, width, height, 1, ASI_IMG_RAW8),
            "ASISetROIFormat",
        )

        self._check(sys.ASIStartVideoCapture(cid), "ASIStartVideoCapture")
        self._opened = True
        print(f"[ZWO] Video capture started: {width}x{height}, "
              f"exposure={exposure_us}us, gain={gain}")

        return self

    @classmethod
    def open_new(
        cls,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        exposure_us: int = 50000,
        gain: int = 50,
        sdk_path: str = DEFAULT_SDK_PATH,
    ) -> "ZwoCamera":
        """Factory method that creates and opens in one call."""
        cam = cls(sdk_path)
        cam.open(camera_id, width, height, exposure_us, gain)
        return cam

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self._lib.ASIStopVideoCapture(self._camera_id)
            self._lib.ASICloseCamera(self._camera_id)
        except Exception:
            pass
        self._opened = False
        self._camera_id = -1

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "ZwoCamera":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def camera_info(self) -> CameraInfo:
        return self._info

    @property
    def is_opened(self) -> bool:
        return self._opened

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def set_exposure(self, exposure_us: int) -> None:
        self._check(
            self._lib.ASISetControlValue(
                self._camera_id, ASI_EXPOSURE, exposure_us, ASI_FALSE,
            ),
            "ASISetControlValue(EXPOSURE)",
        )
        self._exposure_us = exposure_us

    def set_gain(self, gain: int) -> None:
        self._check(
            self._lib.ASISetControlValue(
                self._camera_id, ASI_GAIN, gain, ASI_FALSE,
            ),
            "ASISetControlValue(GAIN)",
        )
        self._gain = gain

    @property
    def exposure_us(self) -> int:
        return self._exposure_us

    @property
    def gain(self) -> int:
        return self._gain

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------
    def grab_frame(self) -> np.ndarray:
        """Grab a single RAW8 frame and return as a (H, W) uint8 array."""
        buf_size = self._width * self._height
        buf = ctypes.create_string_buffer(buf_size)
        pbuf = ctypes.cast(buf, ctypes.c_void_p)
        self._check(
            self._lib.ASIGetVideoData(self._camera_id, pbuf, buf_size, 1000),
            "ASIGetVideoData",
        )
        return np.frombuffer(buf.raw, dtype=np.uint8).reshape(self._height, self._width)

    def grab_frame_rgb(self) -> np.ndarray:
        """Grab a frame and convert to (H, W, 3) BGR for OpenCV display."""
        gray = self.grab_frame()
        return np.stack([gray] * 3, axis=-1)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_num_cameras(sdk_path: str = DEFAULT_SDK_PATH) -> int:
        if not os.path.isfile(sdk_path):
            return 0
        lib = ctypes.CDLL(sdk_path)
        lib.ASIGetNumOfConnectedCameras.restype = ctypes.c_int
        return lib.ASIGetNumOfConnectedCameras()
