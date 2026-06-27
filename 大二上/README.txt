========================================================================
            人工智能基础与进阶实训 - 期末项目源代码说明
========================================================================

项目名称：面向复杂动态环境的移动机器人多约束自主导航与任务调度系统
小组名称：智工小趴虎 (第二组)
提交日期：2026年2月6日

【目录结构概览】

Project_Source_Code/
└── src/                              # ROS 工作空间源码目录
    ├── scorpio/                      # 机器人底层控制与遥控功能包
    │   ├── scorpio_teleop/
    │   │   └── scripts/              # [核心修改] 自研业务逻辑与算法脚本
    │   └── ... (其他底层驱动)
    │
    ├── scorpio_app/                  # 机器人上层应用功能包
    │   ├── scorpio_navigation/       # 导航核心功能包
    │   │   ├── launch/
    │   │   │   └── param/            # [核心修改] 路径规划与代价地图参数
    │   │   └── maps/                 # Cartographer 构建的高精度地图
    │   └── ... (其他应用)
    │
    ├── rrt_exploration/              # [历史尝试] RRT自主探索算法 (非最终方案)
    │
    └── ... (其他通用依赖包)


【核心模块详细说明】

1. 决策与控制脚本 (位于 src/scorpio/scorpio_teleop/scripts/)
   这是本项目的“大脑”部分，包含所有自研的Python控制逻辑。
   - goal_pub_loop.py: [主程序] 基于状态机(State Machine)的任务调度系统。实现了任务队列管理、死锁检测(Guardian System)以及AMCL抗漂移重置逻辑。
   - record_path.py: [工具] 稀疏化变频录制脚本。用于在示教阶段根据速度自动切换采样频率（竞速段稀疏，倒车段密集）。
   - optimized_parking_path.py: [工具] 倒车路径数学优化脚本。用于对录制的倒车轨迹进行四元数对齐优化，生成“虚拟导轨”。
   - recorded_path.yaml / optimized_path.yaml: 存储示教录制和优化后的路径点数据。

2. 导航参数配置 (位于 src/scorpio_app/scorpio_navigation/launch/param/)
   这是本项目的“小脑”部分，针对赛道特性进行了深度调优。
   - teb_local_planner_params.yaml: 
     * 启用了倒车支持 (allow_init_with_backwards_motion)。
     * 设置了类车模型约束 (min_turning_radius: 0.25)。
     * 调整了动力学参数 (max_vel_x: 1.3, acc_lim_x: 1.8) 以适应竞速。
   - costmap_common_params.yaml: 
     * 配置了分层代价地图 (Static, Obstacle, Inflation)。
     * 调整了膨胀半径与清除范围，防止动态障碍物留影。

3. 自主探索尝试 (位于 src/rrt_exploration/)
   - 说明：该模块包含我们在期末初期针对“逸仙勇士杯”尝试使用的RRT自主探索算法。
   - 状态：虽然在初期进行了部署和测试，但考虑到比赛场地固定且对速度/精度要求极高，RRT方案在稳定性和效率上不如“示教-优化-复现”框架。最终方案未启用此模块，保留此代码旨在展示我们的探索过程与技术选型思考。

【环境与编译】

1. 运行环境：Ubuntu 20.04 + ROS Noetic
2. 硬件平台：天蝎座 (Scorpio) 移动机器人 (Jetson Nano/Orin)
3. 编译方式：
   cd [workspace_root]
   catkin_make
   source devel/setup.bash

【运行简述】

1. 启动底层驱动与定位：
   roslaunch scorpio_navigation navigation.launch
2. 启动任务调度主程序（根据模式选择）：
   python3 src/scorpio/scorpio_teleop/scripts/goal_pub_loop.py [nav/record]

========================================================================