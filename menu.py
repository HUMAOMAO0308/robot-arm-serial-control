from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent

JOINT_LIMITS = {
    1: (-170, 170, "J1 底座旋转"),
    2: (-75, 90, "J2 肩部"),
    3: (35, 180, "J3 肘部"),
    4: (-180, 180, "J4 腕部旋转"),
    5: (-120, 120, "J5 腕部俯仰"),
    6: (-360, 360, "J6 末端旋转"),
}

_viz_process: subprocess.Popen | None = None


def _clear() -> None:
    print("\033[2J\033[H", end="")


def _header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def ask(prompt: str, default: Any = None, cast: Callable = str) -> Any:
    label = f"{prompt} [{default}]" if default is not None else prompt
    try:
        val = input(f"  {label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  [取消]")
        return None
    if val == "" and default is not None:
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        print(f"  [无效输入，使用默认值: {default}]")
        return default


def _confirm(msg: str) -> bool:
    try:
        return input(f"  {msg} [y/N]: ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _wait() -> None:
    try:
        input("\n  按 Enter 返回菜单...")
    except (EOFError, KeyboardInterrupt):
        pass


def _run(cmd: list[str]) -> bool:
    """Run a subprocess, return True on success, show friendly error on fail."""
    print(f"\n  → 执行: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n  [失败] 请确保已连接机械臂和 ZWO 相机")
        _wait()
        return False
    print("\n  [完成]")
    _wait()
    return True


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_scan_arc() -> None:
    _header("单轴 arc 扫描")
    joint = ask("关节编号", default=3, cast=int)
    steps = ask("步数", default=200, cast=int)
    start = ask("起始角度（Enter=默认）", default=None, cast=float)
    end = ask("终止角度（Enter=默认）", default=None, cast=float)
    speed = ask("速度", default=50, cast=int)
    pause = ask("每位姿停留秒数", default=1.0, cast=float)
    delay = ask("到位后延迟拍照秒数", default=0.3, cast=float)
    out = ask("输出父目录", default="scans", cast=str)
    name = ask("扫描名称（留空=自动时间戳）", default="", cast=str)
    intrinsics = ask("内参 JSON 路径（留空=跳过）", default=None)
    fk = _confirm("每帧计算 FK 末端位姿？")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "scan_multi_joint.py"),
        "--mode", "arc", "--joint", str(joint),
        "--steps", str(steps), "--speed", str(speed),
        "--pause-time", str(pause), "--capture-delay", str(delay),
        "--output-dir", out,
    ]
    if name:
        cmd += ["--name", name]
    if start is not None:
        cmd += ["--start-angle", str(start)]
    if end is not None:
        cmd += ["--end-angle", str(end)]
    if intrinsics:
        cmd += ["--intrinsics", intrinsics]
    if fk:
        cmd += ["--compute-fk"]
    _run(cmd)


def run_scan_hemisphere() -> None:
    _header("双轴 hemisphere 扫描")
    j1r = ask("J1 范围 (start,end)", default="-60,60")
    j1s = ask("J1 每层步数", default=7, cast=int)
    j2r = ask("J2 范围 (start,end)", default="-65,-50")
    j3r = ask("J3 范围 (start,end)", default="140,160")
    elev = ask("高度层数", default=4, cast=int)
    speed = ask("速度", default=50, cast=int)
    pause = ask("每位姿停留秒数", default=1.0, cast=float)
    delay = ask("到位后延迟拍照秒数", default=0.3, cast=float)
    out = ask("输出父目录", default="scans", cast=str)
    name = ask("扫描名称（留空=自动时间戳）", default="", cast=str)
    intrinsics = ask("内参 JSON 路径（留空=跳过）", default=None)
    fk = _confirm("每帧计算 FK 末端位姿？")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "scan_multi_joint.py"),
        "--mode", "hemisphere",
        "--j1-range", j1r, "--j1-steps", str(j1s),
        "--j2-range", j2r, "--j3-range", j3r,
        "--elevation-steps", str(elev), "--speed", str(speed),
        "--pause-time", str(pause), "--capture-delay", str(delay),
        "--output-dir", out,
    ]
    if name:
        cmd += ["--name", name]
    if intrinsics:
        cmd += ["--intrinsics", intrinsics]
    if fk:
        cmd += ["--compute-fk"]
    _run(cmd)


def run_scan_file() -> None:
    _header("自定义轨迹扫描")
    wp = ask("轨迹文件路径 (JSON)", default="sample_waypoints.json")
    speed = ask("速度", default=50, cast=int)
    pause = ask("每位姿停留秒数", default=1.0, cast=float)
    delay = ask("到位后延迟拍照秒数", default=0.3, cast=float)
    out = ask("输出父目录", default="scans", cast=str)
    name = ask("扫描名称（留空=自动时间戳）", default="", cast=str)
    intrinsics = ask("内参 JSON 路径（留空=跳过）", default=None)
    fk = _confirm("每帧计算 FK 末端位姿？")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "scan_multi_joint.py"),
        "--mode", "file", "--waypoints", wp,
        "--speed", str(speed),
        "--pause-time", str(pause), "--capture-delay", str(delay),
        "--output-dir", out,
    ]
    if name:
        cmd += ["--name", name]
    if intrinsics:
        cmd += ["--intrinsics", intrinsics]
    if fk:
        cmd += ["--compute-fk"]
    _run(cmd)


def run_calibrate_capture() -> None:
    _header("内参标定 — 采集棋盘格")
    width = ask("分辨率宽", default=1920, cast=int)
    height = ask("分辨率高", default=1080, cast=int)
    exp = ask("曝光 (µs)", default=50000, cast=int)
    gain = ask("增益", default=50, cast=int)
    out = ask("图像保存目录", default="calib_images")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "calibrate.py"), "capture",
        "--width", str(width), "--height", str(height),
        "--exposure", str(exp), "--gain", str(gain),
        "--output-dir", out,
    ]
    _run(cmd)


def run_calibrate_compute() -> None:
    _header("内参标定 — 计算内参")
    indir = ask("图像目录", default="calib_images")
    sq = ask("棋盘格方格尺寸 (mm)", default=25.0, cast=float)
    out = ask("输出 JSON", default="camera_intrinsics.json")
    show_corners = _confirm("逐张显示角点检测结果？")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "calibrate.py"), "compute",
        "--input-dir", indir, "--square-size", str(sq),
        "--output", out,
    ]
    if show_corners:
        cmd += ["--show-corners"]
    _run(cmd)


def run_hand_eye() -> None:
    _header("手眼标定")
    port = ask("机械臂串口", default="/dev/ttyACM0")
    intrinsics = ask("内参 JSON 路径", default="camera_intrinsics.json")
    sq = ask("棋盘格方格尺寸 (mm)", default=25.0, cast=float)
    method = ask("标定方法 (tsai/park/horaud)", default="tsai")
    out = ask("输出目录", default="hand_eye_calib")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "hand_eye_calib.py"),
        "--port", port, "--intrinsics", intrinsics,
        "--square-size", str(sq), "--method", method,
        "--output-dir", out,
    ]
    _run(cmd)


def show_limits() -> None:
    _header("固件关节角度限制")
    print()
    print("  Joint   Min       Max       说明")
    print("  ─────   ────────  ────────  ────────")
    for j in range(1, 7):
        lo, hi, label = JOINT_LIMITS[j]
        print(f"  J{j}      {lo:>6}°   {hi:>6}°   {label}")
    _wait()


def run_visual_control() -> None:
    """Start the Three.js visual control web server."""
    global _viz_process
    if _viz_process is not None and _viz_process.poll() is None:
        print("  [INFO] Visual server already running at http://127.0.0.1:8765")
        _wait()
        return

    _header("可视化控制")
    print()
    print("  Launching 3D visual control in browser ...")
    cmd = [sys.executable, str(SCRIPT_DIR / "visual_server.py")]
    _viz_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  Server started → http://127.0.0.1:8765")
    print("  Open this URL in your browser if it doesn't open automatically.")
    _wait()


def show_help() -> None:
    _header("使用帮助")
    print()
    print("  本工具是机械臂控制 + ZWO 相机拍照 + 标定的统一入口。")
    print()
    print("  日常使用:")
    print("    1. 扫描拍照 → arc/hemisphere → 获得图片 + 关节角 CSV")
    print()
    print("  首次安装:")
    print("    2. 标定工具 → 相机内参 + 手眼标定（做一次即可）")
    print()
    print("  三维重建:")
    print("    去畸变图片 + FK 末端位姿 → COLMAP/3DGS → 植物点云")
    _wait()


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------

def scan_menu() -> None:
    while True:
        _clear()
        _header("扫描拍照")
        print()
        print("  1. 单轴 arc 扫描  (只动一个关节)")
        print("  2. 双轴 hemisphere 扫描  (J1+J2+J3 球面覆盖)")
        print("  3. 自定义轨迹扫描  (JSON 文件)")
        print("  0. 返回主菜单")
        print()
        try:
            ch = input("  请选择 [1/2/3/0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if ch == "1":
            run_scan_arc()
        elif ch == "2":
            run_scan_hemisphere()
        elif ch == "3":
            run_scan_file()
        elif ch == "0":
            return


def calib_menu() -> None:
    while True:
        _clear()
        _header("标定工具")
        print()
        print("  ⚠ 标定只需在首次连接时做一次，日常实验不用重复")
        print()
        print("  1. 相机内参 — 采集棋盘格图像")
        print("  2. 相机内参 — 计算内参 (fx/fy/cx/cy)")
        print("  3. 手眼标定 — T_cam_to_ee")
        print("  0. 返回主菜单")
        print()
        try:
            ch = input("  请选择 [1/2/3/0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if ch == "1":
            run_calibrate_capture()
        elif ch == "2":
            run_calibrate_compute()
        elif ch == "3":
            run_hand_eye()
        elif ch == "0":
            return


def main_menu() -> None:
    while True:
        _clear()
        print()
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║         DummyRobot 控制与标定工具                ║")
        print("  ╠══════════════════════════════════════════════════╣")
        print("  ║                                                  ║")
        print("  ║   1. 扫描拍照      (机械臂 + ZWO 逐帧拍照)       ║")
        print("  ║   2. 标定工具      (内参 + 手眼，首次做一次)     ║")
        print("  ║   ────────────────────────────────────           ║")
        print("  ║   3. 查看关节限制                                ║")
        print("  ║   4. 帮助                                        ║")
        print("  ║   5. 可视化控制    (3D Web 视图 + 滑块)          ║")
        print("  ║   0. 退出                                        ║")
        print("  ║                                                  ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print()
        try:
            ch = input("  请选择 [1/2/3/4/5/0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            return
        if ch == "1":
            scan_menu()
        elif ch == "2":
            calib_menu()
        elif ch == "3":
            show_limits()
        elif ch == "4":
            show_help()
        elif ch == "5":
            run_visual_control()
        elif ch == "0":
            print("\n  再见!")
            break
        else:
            print("  [无效选择]")


if __name__ == "__main__":
    main_menu()
