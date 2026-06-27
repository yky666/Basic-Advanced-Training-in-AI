# 面向复杂动态环境的移动机器人多约束自主导航与任务调度系统

![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue.svg) ![Platform](https://img.shields.io/badge/Platform-Scorpio_Robot-orange.svg) ![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green.svg) ![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

> **中山大学智能工程学院“人工智能基础与进阶实训”大二上期末项目**
>
> **小组名称**：智工小趴虎（第二组）

## 📖 项目简介 (Introduction)

本项目针对 Scorpio 移动机器人平台，设计并实现了一套面向复杂物理环境的自主导航与任务调度系统。系统围绕高速遥控竞速、大场景建图、长走廊定位、坡道位姿估计、动态障碍物规避、定点启停与倒车入库等任务展开。

项目采用“**示教录制 + 轨迹优化 + 导航栈调参**”的混合架构：通过路径录制脚本获取稳定轨迹，再结合 MoveBase、TEB 局部规划、Cartographer 建图、EKF 多传感器融合和 Python 状态机调度完成完整任务链。

## ✨ 核心功能 (Key Features)

* **🏎️ 高速遥控竞速**：通过 `record_smart_keypoints.py`、`goal_pub_loop.py` 等脚本记录并复现关键路径点，提升高速行驶稳定性。
* **🗺️ 高精度建图定位**：使用 Cartographer、Gmapping、Hector、Karto 等多种建图方案进行对比，并保留地图文件与导航参数。
* **⛰️ 坡道与复杂路况适配**：结合底盘里程计、雷达与 IMU 信息，配合 EKF/导航参数调优降低坡道漂移影响。
* **🅿️ 倒车入库轨迹优化**：通过 `optimize_parking_path.py` 与 `optimized_path.yaml` 对停车路径进行后处理，提升终点姿态一致性。
* **🤖 状态机任务调度**：基于 Python 脚本组织导航、路径回放、异常恢复和任务阶段切换，减少人工干预。
* **🧪 多算法探索保留**：仓库中保留 `rrt_exploration`、多 SLAM 启动文件和导航参数，便于复盘不同算法效果。

## 📂 文件结构 (File Structure)

```text
大二上/
├── README.md                         # 当前说明文件
├── README.txt                        # 原始项目说明
├── LICENSE                           # 项目许可证
├── 启动脚本.txt                       # 常用启动命令记录
├── map/                              # 课程任务中使用的地图文件
├── 图片/                             # 展示图片与调试资料
└── src/                              # ROS 工作空间源码
    ├── robot_description/            # 机器人 URDF、mesh 与仿真描述
    ├── robot_navigation/             # 通用导航、建图、路径点导航配置
    ├── rrt_exploration/              # RRT 自主探索算法尝试
    ├── scorpio/                      # Scorpio 底盘、描述、遥控与测试包
    │   └── scorpio_teleop/scripts/
    │       ├── goal_pub_loop.py          # 任务调度状态机主程序
    │       ├── record_smart_keypoints.py # 路径点记录工具
    │       ├── optimize_parking_path.py  # 倒车路径优化脚本
    │       └── optimized_path.yaml       # 优化后的路径配置
    ├── scorpio_app/                  # 导航、SLAM、跟随等上层应用包
    └── scorpio_driver/               # 底盘、相机、雷达等驱动包
```

## 🚀 安装与运行 (Installation & Usage)

### 1. 环境依赖

* **系统**：Ubuntu 20.04
* **ROS 版本**：ROS Noetic
* **核心依赖**：

```bash
sudo apt-get update
sudo apt-get install ros-noetic-move-base ros-noetic-teb-local-planner ros-noetic-robot-localization ros-noetic-cartographer-ros
```

### 2. 编译项目

```bash
cd 大二上
catkin_make
source devel/setup.bash
```

若课程环境中已将 `src/` 作为工作空间源码目录，也可以将本目录中的 `src/` 放入现有 catkin workspace 后编译。

### 3. 常用启动流程

**步骤 1：启动底层驱动与导航栈**

```bash
roslaunch scorpio_navigation navigation.launch
```

**步骤 2：启动任务调度脚本**

```bash
python3 src/scorpio/scorpio_teleop/scripts/goal_pub_loop.py nav
```

**可选：重新录制路径点**

```bash
python3 src/scorpio/scorpio_teleop/scripts/record_smart_keypoints.py
```

**可选：优化倒车入库路径**

```bash
python3 src/scorpio/scorpio_teleop/scripts/optimize_parking_path.py
```

## 🔧 调参与复现实用提示 (Tuning Notes)

* 导航效果与 `teb_local_planner_params.yaml`、代价地图膨胀半径、局部规划器速度限制强相关。
* 长走廊与低特征场景建议优先使用 Cartographer 地图，并结合实际雷达安装角度调整 TF。
* 路径回放前应确认地图坐标系、起点姿态、`optimized_path.yaml` 与当前场地一致。
* 硬件串口、雷达型号、摄像头参数可能随实验室设备变化，需要在 launch 与 yaml 文件中同步修改。

## 📄 许可证 (License)

本项目沿用原工程许可证，详见 `LICENSE`。
