# YOLO Traffic Light Recognition

ROS1 node for red/green traffic light recognition with YOLOv8.

## Files

- `scripts/yolo_traffic_light_node.py`: ROS image subscriber and state publisher.
- `yolo_detector.py`: YOLOv8 inference and left/right assignment.
- `weights/best.pt`: trained YOLOv8 weight.
- `launch/yolo_traffic_light.launch`: launch when this folder is a catkin package.
- `launch/yolo_standalone.launch`: launch when this folder is kept under `dzy/src/traffic_light`.

## Dependencies

```bash
pip3 install ultralytics torch torchvision
```

## Run As Independent Package

If this package is copied to `~/agilex_ws/src/traffic_light`:

```bash
cd ~/agilex_ws
catkin_make
source devel/setup.bash
roslaunch traffic_light yolo_traffic_light.launch
```

## Run Under dzy/src/traffic_light

If this folder is kept at:

```text
/home/agilex/agilex_ws/src/dzy/src/traffic_light
```

run:

```bash
cd /home/agilex/agilex_ws
catkin_make
source devel/setup.bash
chmod +x /home/agilex/agilex_ws/src/dzy/src/traffic_light/scripts/yolo_traffic_light_node.py
roslaunch /home/agilex/agilex_ws/src/dzy/src/traffic_light/launch/yolo_standalone.launch
```

Defaults:

```text
image_topic: /camera/color/image_raw
depth_topic: /camera/depth/image_raw
weights: /home/agilex/agilex_ws/src/dzy/src/traffic_light/weights/best.pt
```

## Topics

```text
/traffic_light_node/state          std_msgs/String, JSON result
/traffic_light_node/can_start      std_msgs/Bool
/traffic_light_node/selected_lane  std_msgs/String
/traffic_light_node/debug_image    sensor_msgs/Image
```

View debug image:

```bash
rqt_image_view /traffic_light_node/debug_image
```
