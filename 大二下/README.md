# LIMO 智能车多模态竞赛任务系统

![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue.svg) ![Platform](https://img.shields.io/badge/Platform-LIMO_Robot-orange.svg) ![Vision](https://img.shields.io/badge/Vision-YOLO%20%7C%20OCR%20%7C%20VLM-green.svg) ![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-purple.svg)

> **中山大学智能工程学院“人工智能基础与进阶实训”大二下期末项目**
>
> 期末竞赛任务覆盖红绿灯识别、文字识别与决策、图像识别、视觉巡线、结果汇报与 BONUS 图像描述等环节。

## 📖 项目简介 (Introduction)

本项目面向 LIMO 智能车竞赛场景，将视觉感知、语音/文字理解、导航控制与 ROS 任务调度整合到同一套工程中。系统最初尝试使用传统 CV 完成红绿灯识别，并尝试用 YOLO 统一处理红绿灯、文字与人脸/图像识别任务；在实际调试中，根据不同任务的稳定性和泛化效果进行了路线调整。

最终方案中，红绿灯识别转向半自动标注数据后的 YOLO 检测；文字识别与指令决策使用 OCR/文本选择逻辑；图像识别与 BONUS 描述任务结合视觉语言接口实现；视觉巡线由图像处理节点提取车道/线条偏差并发布控制量。`dzy` 负责上层任务逻辑与感知模块，`limo_ros` 保留 LIMO 平台运行环境和底层 ROS 包。

## ✨ 核心功能 (Key Features)

* **🚦 红绿灯识别**：从传统 HSV/轮廓等 CV 方法迭代到 YOLO 检测，配合半自动标注数据增强识别鲁棒性。
* **🔤 文字识别与决策**：由 YOLO 尝试转向 OCR 与文本选择器，识别文字后映射到路线、动作或任务参数。
* **🖼️ 图像识别与描述**：保留水果/人脸等 YOLO 训练与推理代码，同时通过视觉语言 API 支持更灵活的图片描述与汇报。
* **🛣️ 视觉巡线控制**：通过 `vision_line_follow` 模块检测线条位置，计算偏差并输出速度控制命令。
* **🧠 任务状态机**：使用 `mission_state_machine.py` 系列脚本串联导航、识别、决策、倒车、汇报等竞赛流程。
* **🗺️ 路径录制与复现**：支持 waypoint 与 `cmd_vel` 录制/回放，便于在固定赛道上快速复现稳定路线。

## 📂 文件结构 (File Structure)

```text
大二下/
├── README.md                         # 当前说明文件
└── 期末大作业/
    ├── dzy/                          # 竞赛主工程
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── cmd_vel_paths/            # 速度指令录制结果
    │   ├── config/                   # 不同路线与任务配置
    │   ├── launch/                   # 上层任务启动文件
    │   ├── maps/                     # 任务地图
    │   ├── scripts/                  # 状态机、路径点、cmd_vel 录制与回放
    │   │   ├── mission_state_machine.py
    │   │   ├── movebase_waypoint_runner.py
    │   │   ├── record_cmd_vel.py
    │   │   └── play_cmd_vel.py
    │   └── src/
    │       ├── traffic_light/         # YOLO 红绿灯识别节点
    │       ├── text_recognition/      # OCR/文本选择与决策
    │       ├── image_reccognition_code/ # 图像识别训练与推理代码
    │       ├── vision_line_follow/    # 视觉巡线检测与控制
    │       └── voice_result/          # 语音、视觉语言模型与结果汇报
    │
    └── limo_ros/                      # LIMO 平台 ROS 工程
        ├── learning_limo/             # LIMO 学习与示例代码
        └── limo_bringup/              # 启动、地图、导航参数与传感器配置
```

## 🚀 安装与运行 (Installation & Usage)

### 1. 环境依赖

* **系统**：Ubuntu 20.04 或课程实验平台镜像
* **ROS 版本**：ROS Noetic
* **Python**：建议 Python 3.x
* **视觉依赖**：OpenCV、Ultralytics/YOLO、OCR 相关库、视觉语言 API SDK 或本地模型环境
* **硬件依赖**：LIMO 小车、摄像头、雷达、语音输出设备、网络/API 配置

### 2. 编译 ROS 工程

可将 `dzy` 与 `limo_ros` 放入同一 catkin 工作空间的 `src/` 下，或按课程原环境路径组织：

```bash
mkdir -p ~/limo_ws/src
cd ~/limo_ws/src
# 放置 dzy 与 limo_ros
cd ..
catkin_make
source devel/setup.bash
```

### 3. 运行任务模块

**启动 LIMO 底层与导航环境**

```bash
roslaunch limo_bringup limo_start.launch
```

**启动红绿灯识别节点**

```bash
roslaunch traffic_light yolo_traffic_light.launch
```

**启动视觉巡线模块**

```bash
roslaunch vision_line_follow line_follow.launch
```

**运行任务状态机**

```bash
python3 dzy/scripts/mission_state_machine.py
```

**录制/回放速度控制指令**

```bash
python3 dzy/scripts/record_cmd_vel.py
python3 dzy/scripts/play_cmd_vel.py
```

## 🧪 技术路线记录 (Technical Notes)

* 红绿灯模块曾尝试传统 CV，实测对光照、角度与图像翻转敏感；后续改为 YOLO，并通过半自动标注扩充训练数据。
* 文字识别与人脸/图像识别最初都尝试 YOLO 统一方案，但文字任务更适合 OCR，人脸/图像描述则更适合视觉语言接口。
* 视觉语言接口同时服务于图像识别和 BONUS 图片描述，降低了后处理规则复杂度。
* 视觉巡线模块应在固定曝光、固定相机姿态下调参，重点关注 ROI、阈值、线条中心偏差和速度限幅。

## 📌 上传范围说明 (Repository Scope)

当前目录只上传期末大作业中的两个核心代码目录：

- `期末大作业/dzy/`
- `期末大作业/limo_ros/`

项目报告、比赛视频、截图文档、压缩包、提交要求文件等不纳入仓库，避免仓库体积过大并保持代码目录清晰。
