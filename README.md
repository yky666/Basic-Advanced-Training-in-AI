# 人工智能基础与进阶实训课程项目合集

![Course](https://img.shields.io/badge/Course-Basic%20%26%20Advanced%20Training%20in%20AI-blue.svg) ![ROS](https://img.shields.io/badge/ROS-Noetic-orange.svg) ![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green.svg) ![Platform](https://img.shields.io/badge/Platform-Scorpio%20%7C%20LIMO-purple.svg)

> **中山大学智能工程学院“人工智能基础与进阶实训”课程项目整理**
>
> 本仓库按学期归档大二上与大二下课程项目代码，保留核心工程、地图、配置和运行脚本，便于复现实验环境与后续迭代。

## 📖 仓库简介 (Overview)

本仓库目前包含两个阶段的课程实践项目：

- **大二上期末大作业**：面向 Scorpio 移动机器人平台，围绕自主导航、建图定位、路径规划、遥控竞速与任务调度进行系统设计。
- **大二下期末大作业**：面向 LIMO 小车竞赛任务，围绕红绿灯识别、文字识别与决策、图像识别、视觉巡线、语音/视觉汇报等模块构建 ROS 集成方案。

仓库整理时将两个学期内容拆分到独立目录，并分别提供 README，避免原始代码、文档和实验资料混在根目录中。

## ✨ 项目亮点 (Highlights)

* **🧭 跨学期归档**：统一整理大二上、大二下课程项目，根目录只承担导航入口与总体说明。
* **🤖 ROS 工程保留**：保留 Scorpio 与 LIMO 平台相关的 ROS 包、launch 文件、地图与参数配置。
* **👁️ 多模态感知任务**：大二下项目包含 YOLO 红绿灯识别、OCR 文字识别、图像识别、视觉语言 API 汇报等模块。
* **🛣️ 导航与巡线结合**：既包含传统导航栈、Cartographer/MoveBase/TEB 等模块，也包含视觉巡线和路径录制回放逻辑。
* **📚 README 分层说明**：根目录负责总览，`大二上/` 与 `大二下/` 负责各自项目背景、结构和运行提示。

## 📂 文件结构 (File Structure)

```text
.
├── README.md                         # 仓库总览说明
├── .gitignore                        # 忽略报告、视频、压缩包、构建产物等
├── 大二上/                            # 大二上期末大作业
│   ├── README.md                     # 大二上项目说明
│   ├── README.txt                    # 原项目说明文件
│   ├── LICENSE
│   ├── map/                          # 建图结果与地图配置
│   ├── src/                          # Scorpio/导航/SLAM/调度等 ROS 源码
│   └── 图片/                         # 展示与调试图片资料
│
└── 大二下/                            # 大二下课程项目
    ├── README.md                     # 大二下项目说明
    └── 期末大作业/
        ├── dzy/                      # 竞赛任务主工程与感知/决策模块
        └── limo_ros/                 # LIMO 平台 ROS 包与底层运行环境
```

## 🚀 使用方式 (Usage)

### 1. 克隆仓库

```bash
git clone git@github.com:yky666/Basic-Advanced-Training-in-AI.git
cd Basic-Advanced-Training-in-AI
```

### 2. 查看对应学期说明

```bash
# 大二上项目
cd 大二上
cat README.md

# 大二下项目
cd ../大二下
cat README.md
```

### 3. ROS 环境建议

* **系统**：Ubuntu 20.04 或课程实验平台对应系统
* **ROS**：ROS Noetic
* **构建方式**：根据对应目录内 `src/`、`dzy/`、`limo_ros/` 的 ROS 包结构使用 `catkin_make` 或课程环境脚本构建
* **硬件依赖**：Scorpio/LIMO 小车、摄像头、雷达、串口设备名和模型路径需按实际实验平台调整

## 📌 上传范围说明 (Repository Scope)

本仓库主要保存可复用代码与必要配置。为控制仓库体积，以下内容不作为主要上传对象：

- 项目报告、答辩 PPT、截图文档等提交材料
- 比赛视频、录屏、大体积压缩包
- LaTeX 编译中间文件、ROS 构建产物、IDE 缓存

如需查看某一学期的具体实现，请进入对应目录阅读子 README。
