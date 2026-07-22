# Robot Arm Serial Control

Dummy 机械臂串口控制 + ZWO 相机拍照 + 相机标定流水线。

## 快速开始

```bash
pip install -r requirements.txt
python3 menu.py
```

```
╔══════════════════════════════════════════════════╗
║   1. 扫描拍照      (机械臂 + ZWO 逐帧拍照)        ║
║   2. 标定工具      (内参 + 手眼，首次做一次)      ║
║   ────────────────────────────────────           ║
║   3. 查看关节限制                                  ║
║   4. 帮助                                         ║
║   5. 可视化控制    (3D Web 视图 + 滑块 + 示教，可离线)  ║
║   0. 退出                                         ║
╚══════════════════════════════════════════════════╝
```

所有参数都有默认值，**直接按 Enter 即可**。不接硬件也能打开浏览菜单（选择 3、4、5）。

## 菜单选项说明

| 选项                | 做什么                                      | 频率           |
| ------------------- | ------------------------------------------- | -------------- |
| **1. 扫描拍照**     | 机械臂走轨迹 + ZWO 逐帧拍照，输出图片和 CSV | 每次实验       |
| → 单轴 arc          | 只动一个关节（如 J3 155→135°）              |                |
| → 双轴 hemisphere   | J1 水平扫 × J2/J3 高度分层                  |                |
| → 自定义轨迹        | JSON 文件定义位姿                           |                |
| **2. 标定工具**     | 相机内参 + 手眼标定                         | 首次安装做一次 |
| → 采集棋盘格        | 连接 ZWO，手持棋盘格拍 20-30 张             |                |
| → 计算内参          | 算出 fx/fy/cx/cy + 畸变系数                 |                |
| → 手眼标定          | 算出相机在末端法兰的安装位姿 T_cam_to_ee    |                |
| **3. 查看关节限制** | 打印 6 个关节的固件角度范围                 | 偶尔查看       |
| **4. 帮助**         | 使用提示                                    | 随时           |
| **5. 可视化控制**   | Three.js 3D 模型 + 6 关节滑块 + 示教位姿记录，浏览器运行 | 随时           |

> 选项 3、4、5 不需要连接任何硬件，可以先打开浏览。

---

## 关节角度范围

来自固件源代码 `dummy_robot.cpp`，所有控制指令都受此限制：

| 关节 | 最小值 | 最大值 | 说明                           |
| ---- | ------ | ------ | ------------------------------ |
| J1   | -170°  | 170°   | 底座旋转                       |
| J2   | -75°   | 90°    | 肩部                           |
| J3   | 35°    | 180°   | 肘部（最低 35°，不能打到负角） |
| J4   | -180°  | 180°   | 腕部旋转                       |
| J5   | -120°  | 120°   | 腕部俯仰                       |
| J6   | -360°  | 360°   | 末端旋转                       |

---

## 项目结构

| 文件                  | 功能                                   |
| --------------------- | -------------------------------------- |
| `menu.py`             | **交互式菜单（推荐入口）**             |
| `visual_server.py`    | **3D 可视化控制 + 示教（Three.js Web 服务）** |
| `scan_multi_joint.py` | 扫描+拍照：单轴/双轴/自定义轨迹        |
| `calibrate.py`        | 相机内参标定：棋盘格采集 + 内参计算    |
| `hand_eye_calib.py`   | 手眼标定：T_cam_to_ee                  |
| `zwo_camera.py`       | ZWO ASI 相机驱动                       |
| `robot_kinematics.py` | 正运动学/逆运动学（URDF 参数）         |
| `sdk/`                | DummyRobot SDK（串口控制+虚拟机器人）  |
| `static/`             | 可视化前端（HTML + STL 网格）          |

---

## 工作流程

```
首次安装（只做一次）:
  菜单 2 标定工具 → 采集棋盘格 → 计算内参 → camera_intrinsics.json
  装好相机         → 手眼标定   → hand_eye_calib/hand_eye_result.json

 日常实验（每次）:
   菜单 1 扫描拍照 → arc 或 hemisphere → 图片 + CSV + (可选:去畸变 +FK)
   菜单 5 可视化控制 → 3D 预览机械臂姿态 + 滑块控制

 三维重建（离线）:
  图片 + 每帧位姿 → COLMAP / 3DGS → 植物点云
```

---

## 命令行参考（高级用户）

`menu.py` 内部通过命令行调用以下脚本，你也可以直接使用命令行：

### 扫描拍照

```bash
# 单轴扫描 J3（默认 155→135°） → scans/20260720_1430_arc_J3/
python3 scan_multi_joint.py --mode arc --joint 3

# 指定扫描名称 → scans/20260720_1430_arc_J3_plant1/
python3 scan_multi_joint.py --mode arc --joint 3 --name plant1

# 自定义单轴参数 + 去畸变 + FK
python3 scan_multi_joint.py --mode arc --joint 3 \
    --start-angle 155 --end-angle 135 --steps 200 \
    --pause-time 1.0 --capture-delay 0.3 \
    --intrinsics camera_intrinsics.json --compute-fk \
    --name my_experiment

# 双轴 hemisphere
python3 scan_multi_joint.py --mode hemisphere \
    --j1-range "-60,60" --j1-steps 7 \
    --j2-range "-65,-50" --j3-range "140,160" \
    --elevation-steps 4 --name plant_scan

# 查看关节限制
python3 scan_multi_joint.py --show-limits
```

| 通用参数          | 默认值         | 含义                                 |
| ----------------- | -------------- | ------------------------------------ |
| `--port`          | `/dev/ttyACM0` | 机械臂串口                           |
| `--speed`         | `50`           | 运动速度                             |
| `--pause-time`    | `1.0`          | 每位姿停留（秒）                     |
| `--capture-delay` | `0.3`          | 到位后延迟拍照（秒）                 |
| `--exposure`      | `50000`        | ZWO 曝光（µs）                       |
| `--gain`          | `50`           | ZWO 增益                             |
| `--intrinsics`    | `None`         | 内参 JSON，提供后每帧去畸变          |
| `--compute-fk`    | `False`        | 每帧计算末端位姿写入 CSV             |
| `--output-dir`    | `scans/`       | 输出父目录（自动创建时间戳子文件夹） |
| `--name`          | `""`           | 扫描名称，追加到子文件夹名中         |

> 输出自动组织为 `scans/YYYYMMDD_HHMM_mode_name/`，不填 `--name` 则只有时间戳+模式名。镜像保存在子目录的 `images/` 下。

### 标定

```bash
# 采集棋盘格图像
python3 calibrate.py capture --width 1920 --height 1080 --output-dir calib_images

# 计算内参
python3 calibrate.py compute --input-dir calib_images --square-size 25.0

# 手眼标定
python3 hand_eye_calib.py --port /dev/ttyACM0 \
    --intrinsics camera_intrinsics.json --square-size 25.0 --method tsai
```

### 可视化控制

```bash
# 启动 Three.js 3D 可视化（虚拟模式，不连硬件）
python3 visual_server.py

# 连接真实机械臂
python3 visual_server.py --robot-port /dev/ttyACM0

# 浏览器打开 http://127.0.0.1:8765
```

**页面布局**：
- 左侧：**示教位姿面板** — 记录、编辑、导出轨迹路径点
- 中央：**3D 模型视图** — 鼠标拖拽旋转/缩放，实时跟随关节变化
- 底部：**6 关节滑块** — 拖动滑块即可控制对应关节

**按钮说明**：

| 按钮      | 功能                                                     |
| --------- | -------------------------------------------------------- |
| Connect   | 连接机械臂（虚拟或真机），连接后变 Disconnect            |
| Home      | 机械臂回零                                               |
| Send      | 将当前 6 个滑块值一次性发送给机械臂                      |
| Record    | **示教**：记录当前关节位姿到左侧列表                     |
| Read      | 读取机械臂当前关节角度，更新滑块和 3D 模型               |
| Reset     | 滑块和 3D 模型恢复到默认位姿（不影响实物）               |
| Stop      | 紧急停止，发送 !STOP 指令                                |

**示教流程**：
1. Connect 连接机械臂
2. 拖动滑块 → Send 发送位姿 → 机械臂移动到目标
3. 点 **Record** 记录当前位姿到左侧列表
4. 重复 2-3，构建完整轨迹
5. 点列表中的 **▶** 跳转到任意位姿，或点 **✕** 删除
6. 点 **导出JSON** 下载轨迹文件（可用于 `--mode file` 扫描）
7. 点 **清空** 清除所有记录

### 运动学

```bash
# 测试正运动学
python3 robot_kinematics.py

# Python 交互
python3 -c "
from robot_kinematics import forward_kinematics, forward_kinematics_chain
chain = forward_kinematics_chain([0, -64, 155, -1, 0, 0])
for name, p in zip(['base']+['link%d'%i for i in range(1,7)], chain.link_poses):
    print(f'{name:8s}  x={p.x*1000:6.1f} y={p.y*1000:6.1f} z={p.z*1000:6.1f} mm')
"
```
