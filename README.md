# Robot Arm Serial Control

Dummy 机械臂串口控制 + ZWO 相机拍照 + 相机标定流水线。

## 项目结构（5 个模块）

| 文件                  | 功能                                               |
| --------------------- | -------------------------------------------------- |
| `scan_multi_joint.py` | **扫描+拍照**：单轴/双轴/自定义轨迹 + ZWO 逐帧拍照 |
| `calibrate.py`        | **相机内参标定**：棋盘格采集 + 内参计算            |
| `hand_eye_calib.py`   | **手眼标定**：解算相机相对末端法兰的安装位姿       |
| `zwo_camera.py`       | ZWO ASI 相机驱动模块（被以上脚本引用）             |
| `robot_kinematics.py` | 正运动学/逆运动学模块（从 URDF 提取的 DH 参数）    |

## 环境准备

```bash
pip install -r requirements.txt
```

依赖：`numpy` `opencv-python` `Pillow` `pyserial` `zwoasi`，ZWO ASI SDK 需要预先安装。

## 运行流程概览

```
┌──────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  calibrate.py    │ ──→ │  camera_     │ ──→ │  hand_eye_      │
│  capture 采集     │     │  intrinsics  │     │  calib.py       │
│  compute 计算     │     │  .json       │     │  手眼标定         │
└──────────────────┘     └──────┬───────┘     └────────┬────────┘
                                │                       │
                                ▼                       ▼
                       ┌──────────────────────────────────────┐
                       │  scan_multi_joint.py 扫描+拍照        │
                       │  --intrinsics       去畸变             │
                       │  --compute-fk       记录末端位姿       │
                       └──────────────────────────────────────┘
```

---

## 一、扫描+拍照 (`scan_multi_joint.py`)

### 单轴扫描 (`--mode arc`)

只动一个关节，其余固定。**等价于原来的 J3 单轴扫描**：

```bash
python3 scan_multi_joint.py --mode arc --joint 3 \
    --j1 0.0 --j2 -64.0 --j3 155.0 --j4 -1.0 --j5 0.0 --j6 0.0 \
    --start-angle 155 --end-angle 135 --steps 200 \
    --pause-time 1.0 --capture-delay 0.3 \
    --output-dir scans/my_plant
```

| 关键参数            | 含义                                   |
| ------------------- | -------------------------------------- |
| `--joint 3`         | 选 J3（肘部），也可以选 1-6 中任意一个 |
| `--start-angle 155` | 起始角度（度）                         |
| `--end-angle 135`   | 终止角度（度）                         |
| `--steps 200`       | 分多少步，每步 = (155-135)/200 = 0.1°  |

### 双轴扫描 (`--mode hemisphere`)

J1 水平扫描 × (J2+J3) 联动分层，zigzag 来回，覆盖植物正面一个大弧形：

```bash
python3 scan_multi_joint.py --mode hemisphere \
    --j1-range "-60,60" --j1-steps 5 \
    --j2-range "-70,-55" --j3-range "140,160" \
    --elevation-steps 3 \
    --output-dir scans/hemisphere_01
```

| 关键参数                    | 含义                        |
| --------------------------- | --------------------------- |
| `--j1-range "-60,60"`       | J1（底座）左右扫 120°       |
| `--j1-steps 5`              | 每层水平拍 5 张             |
| `--j2-range` / `--j3-range` | 高度变化范围，J2 和 J3 配对 |
| `--elevation-steps 3`       | 低/中/高 3 层               |

总拍照数 = `elevation_steps × j1_steps = 3×5 = 15 张`。

### 自定义轨迹 (`--mode file`)

JSON 格式自己定义每个位姿：

```bash
python3 scan_multi_joint.py --mode file --waypoints my_poses.json
```

`my_poses.json`：

```json
{
  "waypoints": [
    [0.0, -64.0, 155.0, -1.0, 0.0, 0.0],
    [20.0, -64.0, 155.0, -1.0, 0.0, 0.0],
    [40.0, -64.0, 155.0, -1.0, 0.0, 0.0]
  ],
  "speeds": [50, 50, 50],
  "labels": ["左侧", "正面", "右侧"]
}
```

### 通用参数

| 参数                    | 默认值         | 含义                                   |
| ----------------------- | -------------- | -------------------------------------- |
| `--port`                | `/dev/ttyACM0` | 机械臂串口                             |
| `--baudrate`            | `115200`       | 波特率                                 |
| `--speed`               | `50`           | 运动速度                               |
| `--pause-time`          | `1.0`          | 每位姿停留时间（秒）                   |
| `--capture-delay`       | `0.3`          | 到位后等多久再拍照（秒）               |
| `--camera-width/height` | `1920/1080`    | ZWO 拍照分辨率                         |
| `--exposure`            | `50000`        | ZWO 曝光（µs）                         |
| `--gain`                | `50`           | ZWO 增益                               |
| `--intrinsics`          | `None`         | 内参 JSON 路径，提供后每帧自动去畸变   |
| `--compute-fk`          | `False`        | 每帧通过正运动学计算末端位姿，写入 CSV |
| `--output-dir`          | `scans/`       | 图片 + CSV + JSON 的输出目录           |

---

## 二、相机内参标定 (`calibrate.py`)

### 前提

打印一张 10×7（9×6 内角点）的棋盘格，贴到刚性平板上。用尺子量好方格的物理尺寸。

生成打印用棋盘格 PNG：

```bash
python3 -c "
import cv2, numpy as np
sz = 120
w, h = 10*sz, 7*sz
board = np.zeros((h,w), np.uint8)
for i in range(7):
    for j in range(10):
        if (i+j)%2==0: board[i*sz:(i+1)*sz, j*sz:(j+1)*sz] = 255
cv2.imwrite('chessboard_9x6.png', board)
" && echo "Saved: chessboard_9x6.png"
```

### 步骤 1：采集图像

连接 ZWO 相机后运行，手持棋盘格变换角度，按空格保存：

```bash
python3 calibrate.py capture \
    --width 1920 --height 1080 \
    --exposure 50000 --gain 50 \
    --output-dir calib_images
```

检测到角点（画面出现彩色线）时按 **空格** 保存，按 **Q** 退出。拍 20-30 张。

### 步骤 2：计算内参

```bash
python3 calibrate.py compute \
    --input-dir calib_images \
    --square-size 25.0 \
    --show-corners
```

输出 `camera_intrinsics.json`，包含 `fx/fy/cx/cy` 和畸变系数 `k1/k2/p1/p2/k3`。

---

## 三、手眼标定 (`hand_eye_calib.py`)

标定相机相对于机械臂末端法兰的安装位姿（T_cam_to_ee）。

```bash
python3 hand_eye_calib.py \
    --port /dev/ttyACM0 \
    --intrinsics camera_intrinsics.json \
    --square-size 25.0 \
    --method tsai
```

程序自动驱动机械臂走到 15 个不同位姿，每个位姿 ZWO 拍照 + solvePnP 解棋盘格位姿，最终用 Tsai 方法解算 T_cam_to_ee，保存到 `hand_eye_calib/hand_eye_result.json`。

---

## 常见使用场景

```bash
# 最简单的单轴扫描（等同原来的脚本）
python3 scan_multi_joint.py --mode arc --joint 3 --start 155 --end 135 --steps 200

# 加入去畸变 + 记录末端位姿（给 3D 重建用）
python3 scan_multi_joint.py --mode arc --joint 3 \
    --start 155 --end 135 --steps 200 \
    --intrinsics camera_intrinsics.json --compute-fk

# 双轴球面扫描植物
python3 scan_multi_joint.py --mode hemisphere \
    --j1-range "-60,60" --j1-steps 7 \
    --elevation-steps 4 \
    --intrinsics camera_intrinsics.json --compute-fk

# 分析 FK 数据（Python 交互）
python3 -c "
from robot_kinematics import forward_kinematics, forward_kinematics_chain
chain = forward_kinematics_chain([0, -64, 155, -1, 0, 0])
for name, p in zip(['base']+['link%d'%i for i in range(1,7)], chain.link_poses):
    print(f'{name:8s}  x={p.x*1000:6.1f} y={p.y*1000:6.1f} z={p.z*1000:6.1f} mm')
"
```
