from __future__ import annotations

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

SCRIPTS_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPTS_DIR / "static"

# ---------------------------------------------------------------------------
# Robot interface
# ---------------------------------------------------------------------------

def _get_robot(port: Optional[str]):
    try:
        from sdk import DummyRobot, VirtualDummyRobot
        if port:
            r = DummyRobot(port)
        else:
            r = VirtualDummyRobot()
        r.connect()
        r.enable()
        print(f"[robot] Connected: {'virtual' if not port else port}")
        return r
    except Exception as e:
        print(f"[robot] {e}, falling back to virtual")
        from sdk import VirtualDummyRobot
        try:
            r = VirtualDummyRobot()
            r.connect()
            r.enable()
            return r
        except Exception as e2:
            print(f"[robot] Virtual also failed: {e2}")
            raise RuntimeError(f"Failed to create robot: {e2}") from e2


# ---------------------------------------------------------------------------
# HTTP server with port reuse
# ---------------------------------------------------------------------------

class _ReusableServer(http.server.HTTPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        super().server_bind()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class VisualAPIHandler(http.server.SimpleHTTPRequestHandler):
    robot = None
    port_arg = None
    _lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass  # suppress log spam

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/joints":
            self._json_response(self._get_joints())
        elif path == "/api/status":
            self._json_response(self._status())
        elif path == "/api/connect":
            try:
                self._connect_robot()
                self._json_response({"ok": True})
            except Exception as e:
                self._json_response({"ok": False, "error": str(e)})
        elif path.startswith("/api/move"):
            self._handle_move(parsed)
        elif path == "/api/home":
            try:
                self._home()
            except Exception as e:
                self._json_response({"ok": False, "error": str(e)})
            else:
                self._json_response({"ok": True})
        elif path == "/api/enable":
            try:
                with self._lock:
                    if self.robot is None:
                        raise RuntimeError("Robot not connected")
                    self.robot.enable()
                self._json_response({"ok": True})
            except Exception as e:
                self._json_response({"ok": False, "error": str(e)})
        elif path == "/api/disable":
            try:
                with self._lock:
                    if self.robot is None:
                        raise RuntimeError("Robot not connected")
                    self.robot.disable()
                self._json_response({"ok": True})
            except Exception as e:
                self._json_response({"ok": False, "error": str(e)})
        elif path == "/api/disconnect":
            with self._lock:
                if self.robot:
                    try:
                        self.robot.disable()
                    except Exception:
                        pass
                    try:
                        self.robot.disconnect()
                    except Exception:
                        pass
                    self.robot = None
            self._json_response({"ok": True})
        else:
            super().do_GET()

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _get_joints(self) -> dict:
        with self._lock:
            if self.robot:
                try:
                    jp = self.robot.get_joint_positions()
                    return {"joints": jp.values, "connected": True}
                except Exception:
                    pass
        return {"joints": [0, 0, 0, 0, 0, 0], "connected": False}

    def _connect_robot(self):
        with self._lock:
            if self.robot:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
            try:
                self.robot = _get_robot(self.port_arg)
            except Exception:
                self.robot = _get_robot(None)

    def _handle_move(self, parsed):
        qs = parse_qs(parsed.query)
        try:
            index = int(qs.get("j", ["0"])[0]) + 1
            value = float(qs.get("v", ["0"])[0])
            speed = float(qs.get("s", ["50"])[0])
            with self._lock:
                if self.robot is None:
                    self._json_response({"ok": False, "error": "Robot not connected"})
                    return
                self.robot.move_single_joint(index, target=value, speed=speed)
            self._json_response({"ok": True})
        except Exception as e:
            self._json_response({"ok": False, "error": str(e)})

    def _home(self):
        with self._lock:
            if self.robot is None:
                raise RuntimeError("Robot not connected")
            self.robot.home()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(port: int, robot_port: Optional[str]):
    VisualAPIHandler.port_arg = robot_port

    try:
        r = _get_robot(robot_port)
        VisualAPIHandler.robot = r
    except Exception:
        pass

    server = _ReusableServer(("127.0.0.1", port), VisualAPIHandler)
    print(f"\n  ========================================")
    print(f"  Visual Control → http://127.0.0.1:{port}")
    print(f"  ========================================\n")

    t = threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}"))
    t.daemon = True
    t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if VisualAPIHandler.robot:
            try:
                VisualAPIHandler.robot.disable()
                VisualAPIHandler.robot.disconnect()
            except Exception:
                pass
        server.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DummyRobot Visual Control Server")
    p.add_argument("--port", type=int, default=8765, help="HTTP server port")
    p.add_argument("--robot-port", default=None, help="Serial port for real robot")
    args = p.parse_args()
    run_server(args.port, args.robot_port)
