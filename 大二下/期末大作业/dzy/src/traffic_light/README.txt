==================================================
  松灵小车 — 红绿灯识别（YOLO）移植包
  来源：ai-race-vision-deploy（识别准确版参数）
==================================================

一、需要给别的车的文件（本文件夹全部）

  traffic_light_deploy/
  ├── models/
  │   └── traffic_light_best.pt          ← YOLO 模型（约 6MB，必带）
  ├── scripts/
  │   ├── traffic_light_yolo.py          ← 检测算法（YOLO + HSV）
  │   └── traffic_light_ros_node.py      ← ROS 节点（发布 /competition/traffic_light/*）
  ├── launch/
  │   └── traffic_light_only.launch      ← 可选：只起红绿灯节点
  └── README.txt

  另：目标车 scorpio_yolo 包内也须有同名节点（或 catkin_make 后安装）：
  /home/agilex/scorpio/src/scorpio_yolo/scripts/traffic_light_ros_node.py

二、目标车目录（推荐）

  /home/agilex/scorpio/ai-race-vision-deploy/
    models/traffic_light_best.pt
    scripts/traffic_light_yolo.py
    launch/traffic_light_only.launch

  /home/agilex/scorpio/src/scorpio_yolo/scripts/
    traffic_light_ros_node.py

三、拷贝到新车（PC 上执行，PowerShell 示例）

  scp -r E:\ros\期末\traffic_light_deploy\models agilex@192.168.43.21:/home/agilex/scorpio/ai-race-vision-deploy/
  scp -r E:\ros\期末\traffic_light_deploy\scripts agilex@192.168.43.21:/home/agilex/scorpio/ai-race-vision-deploy/
  scp E:\ros\期末\traffic_light_deploy\launch\traffic_light_only.launch agilex@192.168.43.21:/home/agilex/scorpio/ai-race-vision-deploy/launch/
  scp E:\ros\期末\traffic_light_deploy\scripts\traffic_light_ros_node.py agilex@192.168.43.21:/home/agilex/scorpio/src/scorpio_yolo/scripts/

  密码一般：agx

  然后在车上：
  cd /home/agilex/scorpio && catkin_make
  source devel/setup.bash

四、新车测试步骤（SSH agilex@192.168.43.21）

  终端1 — 相机：
  source /opt/ros/noetic/setup.bash
  roslaunch astra_camera dabai_u3.launch \
    product_id:=0x0657 uvc_product_id:=0x0557 \
    serial_number:=CC15C5201HC uvc_retry_count:=200 connection_delay:=500 \
    enable_depth:=false enable_ir:=false enable_point_cloud:=false

  终端2 — 红绿灯（二选一）：

  A) rosrun：
  source /opt/ros/noetic/setup.bash
  source /home/agilex/scorpio/devel/setup.bash
  rosrun scorpio_yolo traffic_light_ros_node.py \
    _model_path:=/home/agilex/scorpio/ai-race-vision-deploy/models/traffic_light_best.pt \
    _script_path:=/home/agilex/scorpio/ai-race-vision-deploy/scripts/traffic_light_yolo.py \
    _show_debug:=true _nogui:=true \
    _min_side_green_score:=1500 _brightness_threshold:=120 \
    _right_roi_x:=0.52 _right_roi_w:=0.38

  B) launch（需先拷贝 launch 目录）：
  roslaunch scorpio_yolo traffic_light_only.launch

  终端3 — 看结果：
  rostopic echo /competition/traffic_light/route
  rostopic echo /competition/traffic_light/go

五、发布话题

  /competition/traffic_light/go      Bool   可发车
  /competition/traffic_light/route   String left | right
  /competition/traffic_light/state   String GREEN / RED / UNKNOWN
  /competition/traffic_light/debug/image  调试图像

六、依赖

  ROS Noetic, ultralytics (YOLO), cv_bridge, OpenCV
  相机话题：/camera/color/image_raw

—— 2026-06-18
