# 面向复杂动态环境的移动机器人多约束自主导航与任务调度系统
# Autonomous Mobile Robot Multi-Constraint Navigation and Task Scheduling System

![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue.svg) ![Platform](https://img.shields.io/badge/Platform-Scorpio_Robot-orange.svg) ![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green.svg) ![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> **中山大学智能工程学院“人工智能基础与进阶实训”**
> 
> **小组名称**：智工小趴虎 (第二组)

## 📖 项目简介 (Introduction)

本项目针对天蝎座（Scorpio）移动机器人平台，设计并实现了一套在复杂物理环境下具备高鲁棒性的全自主导航系统。系统旨在解决**高速遥控竞速、大场景长走廊建图、坡道位姿估计、动态障碍物规避、定点精准启停及高精度倒车入库**等核心挑战。

通过提出“**基于示教录制与轨迹优化 (Teach-and-Repeat with Optimization)**”的混合架构，结合 EKF 多传感器融合、Cartographer 图优化建图、TEB 局部规划器深度调优以及 Python 状态机调度，本系统在最终考核中实现了 **零失误、满分 (72/72)** 的优异成绩。

## ✨ 核心功能 (Key Features)

* **🏎️ 高速遥控竞速**：基于类车模型（Car-like）约束，通过 `record_path.py` 实现稀疏化变频录制，配合 TEB 前瞻预测，实现 1.3m/s 高速稳定切弯。
* **🗺️ 高精度建图**：采用 **Google Cartographer** 替代 Gmapping，有效解决了长走廊环境下的特征退化与回环重影问题。
* **⛰️ 坡道稳健通行**：引入 **EKF (扩展卡尔曼滤波)**，融合轮式里程计与 IMU 数据，彻底消除坡道打滑导致的定位漂移。
* **🅿️ 厘米级倒车入库**：首创 **四元数对齐 (Quaternion Alignment)** 轨迹优化算法，构建“虚拟导轨”，实现倒车终点误差 < 2.3cm。
* **🤖 智能任务调度**：自主研发 Python 状态机 (`goal_pub_loop.py`)，集成毫秒级死锁检测 (Guardian System) 与抗漂移重置 (Anti-Drift Reset) 机制。

## 📂 文件结构 (File Structure)

```text
src/
├── scorpio/                      # 机器人底层控制与遥控功能包
│   ├── scorpio_teleop/
│   │   └── scripts/              
│   │       ├── goal_pub_loop.py          # [核心] 任务调度状态机主程序
│   │       ├── record_path.py            # [工具] 稀疏化变频录制脚本
│   │       └── optimized_parking_path.py # [工具] 倒车路径四元数优化脚本
│   └── ... 
├── scorpio_app/                  # 机器人上层应用功能包
│   ├── scorpio_navigation/       # 导航核心功能包
│   │   ├── launch/
│   │   │   ├── param/            # [配置] 代价地图与规划器参数
│   │   │   │   ├── teb_local_planner_params.yaml
│   │   │   │   └── costmap_common_params.yaml
│   │   │   └── navigation.launch # 导航启动文件
│   │   └── maps/                 # Cartographer 构建的高精度地图
│   └── ... 
└── ...



🚀 安装与运行 (Installation & Usage)
1. 环境依赖
Ubuntu 20.04

ROS Noetic

依赖包：move_base, teb_local_planner, robot_localization, cartographer_ros

2. 编译
Bash
cd ~/scorpio_ws
catkin_make
source devel/setup.bash
3. 运行步骤
步骤 1：启动底层驱动与导航栈

Bash
roslaunch scorpio_navigation navigation.launch
步骤 2：启动任务调度器 (开始自动导航)

Bash
# 确保已加载优化后的路径文件 (optimized_path.yaml)
python3 src/scorpio/scorpio_teleop/scripts/goal_pub_loop.py nav
（可选）步骤 0：录制新路径

Bash
python3 src/scorpio/scorpio_teleop/scripts/goal_pub_loop.py record
# 遥控机器人行驶，脚本将自动根据速度切换采样频率
