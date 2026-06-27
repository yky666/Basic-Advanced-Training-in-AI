#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_ros_node.py

YOLOv + 左右亮度阈值识别红绿灯，并发布：
  /competition/traffic_light/go      可发车
  /competition/traffic_light/route   left | right
  /competition/traffic_light/state   GREEN/RED/UNKNOWN
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

DEFAULT_YOLO_SCRIPT = "/home/agilex/agilex_ws/src/dzy/src/traffic_light/scripts/traffic_light_yolo.py"
DEFAULT_MODEL = "/home/agilex/agilex_ws/src/dzy/src/traffic_light/weights/traffic_light_best.pt"


def load_module(script_path: str):
    path = Path(script_path)
    if not path.is_file():
        raise FileNotFoundError(f"traffic light script not found: {path}")
    spec = importlib.util.spec_from_file_location("traffic_light_yolo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DualTrafficLightController:
    """连续稳定检测到可通行方向后，锁存 GO + route。"""

    def __init__(self, detector, stable_frames=8, min_side_green_score=800.0):
        self.detector = detector
        self.stable_frames = stable_frames
        self.min_side_green_score = min_side_green_score
        self.history = deque(maxlen=stable_frames)
        self.latched_go = False
        self.latched_route = ""
        self.display_state = "UNKNOWN"

    def update(self, frame, debug_frame=None):
        result = self.detector.detect(frame, debug_frame=debug_frame)
        route = result.route

        if route and not self.latched_go:
            left_ok = result.left.is_green and (
                result.left.yolo_green
                or result.left.green_score >= self.min_side_green_score
            )
            right_ok = result.right.is_green and (
                result.right.yolo_green
                or result.right.green_score >= self.min_side_green_score
            )
            both_ok = left_ok and right_ok
            if route == "left" and left_ok and not right_ok:
                self.history.append("left")
            elif route == "right" and both_ok:
                self.history.append("right")
            elif route == "right" and right_ok and not left_ok:
                if result.right.green_score >= result.left.green_score * 2.5:
                    self.history.append("right")
                else:
                    self.history.append("")
            else:
                self.history.append("")
        elif not self.latched_go:
            self.history.append("")

        if not self.latched_go and len(self.history) == self.stable_frames:
            if all(item == self.history[0] and item for item in self.history):
                self.latched_go = True
                self.latched_route = self.history[0]
                self.display_state = "GREEN"
                rospy.loginfo(
                    "Traffic GO latched route=%s after %d frames (Lg=%.0f Rg=%.0f)",
                    self.latched_route,
                    self.stable_frames,
                    result.left.green_score,
                    result.right.green_score,
                )
            else:
                if result.left.is_red or result.right.is_red:
                    self.display_state = "RED"
                else:
                    self.display_state = "UNKNOWN"
        elif self.latched_go:
            self.display_state = "GREEN"
        else:
            if result.left.is_red or result.right.is_red:
                self.display_state = "RED"
            else:
                self.display_state = "UNKNOWN"

        return self.display_state, result, self.latched_go, self.latched_route


class TrafficLightRosNode:
    def __init__(self):
        deploy_dir = rospy.get_param("~deploy_dir", "/home/agilex/agilex_ws/src/dzy/src/traffic_light")
        script_path = rospy.get_param("~script_path", DEFAULT_YOLO_SCRIPT)
        model_path = rospy.get_param("~model_path", f"{deploy_dir}/weights/traffic_light_best.pt")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.stable_frames = int(rospy.get_param("~stable_frames", 8))
        self.min_side_green_score = float(rospy.get_param("~min_side_green_score", 800.0))
        self.brightness_threshold = float(rospy.get_param("~brightness_threshold", 140.0))
        self.show_debug = bool(rospy.get_param("~show_debug", True))
        self.nogui = bool(rospy.get_param("~nogui", False))
        self.use_yolo_roi = bool(rospy.get_param("~use_yolo_roi", True))
        self.yolo_conf = float(rospy.get_param("~yolo_conf", 0.5))

        left_roi = (
            float(rospy.get_param("~left_roi_x", 0.0)),
            float(rospy.get_param("~left_roi_y", 0.0)),
            float(rospy.get_param("~left_roi_w", 0.45)),
            float(rospy.get_param("~left_roi_h", 0.70)),
        )
        right_roi = (
            float(rospy.get_param("~right_roi_x", 0.45)),
            float(rospy.get_param("~right_roi_y", 0.0)),
            float(rospy.get_param("~right_roi_w", 0.45)),
            float(rospy.get_param("~right_roi_h", 0.70)),
        )

        tl = load_module(script_path)
        detector = tl.DualTrafficLightDetector(
            model_path=model_path if Path(model_path).is_file() else None,
            left_roi=left_roi,
            right_roi=right_roi,
            brightness_threshold=self.brightness_threshold,
            min_green_score=self.min_side_green_score,
            use_yolo_roi=self.use_yolo_roi and Path(model_path).is_file(),
            yolo_conf=self.yolo_conf,
            use_yolo_class_lights=bool(rospy.get_param("~use_yolo_class_lights", True)),
        )
        self.controller = DualTrafficLightController(
            detector,
            stable_frames=self.stable_frames,
            min_side_green_score=self.min_side_green_score,
        )

        self.bridge = CvBridge()
        self.last_state = "UNKNOWN"

        self.state_pub = rospy.Publisher(
            "/competition/traffic_light/state", String, queue_size=1, latch=True
        )
        self.route_pub = rospy.Publisher(
            "/competition/traffic_light/route", String, queue_size=1, latch=True
        )
        self.raw_pub = rospy.Publisher(
            "/competition/traffic_light/raw_color", String, queue_size=1, latch=True
        )
        self.go_pub = rospy.Publisher(
            "/competition/traffic_light/go", Bool, queue_size=1, latch=True
        )
        self.debug_pub = rospy.Publisher(
            "/competition/traffic_light/debug/image", Image, queue_size=1
        )

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        model_text = model_path if Path(model_path).is_file() else "fixed ROI only"
        rospy.loginfo(
            "Traffic light YOLO node ready. model=%s bright>=%.0f stable=%d",
            model_text,
            self.brightness_threshold,
            self.stable_frames,
        )

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logerr_throttle(5.0, "Image conversion failed: %s", exc)
            return

        debug_frame = frame.copy() if self.show_debug else None
        stable_state, result, latched_go, latched_route = self.controller.update(
            frame, debug_frame
        )
        self.publish_state(stable_state, result, latched_go, latched_route)

        if not self.show_debug:
            return

        display = debug_frame if debug_frame is not None else frame.copy()
        if latched_go:
            cv2.putText(
                display,
                f"GO -> {latched_route}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        if not self.nogui:
            cv2.imshow("traffic_light_yolo", display)
            cv2.waitKey(1)

    def publish_state(self, stable_state, result, latched_go, latched_route):
        if stable_state != self.last_state:
            rospy.loginfo(
                "Traffic light state: %s route=%s (Lg=%.0f Rg=%.0f Lb=%.0f Rb=%.0f latched=%s)",
                stable_state,
                latched_route or result.route or "none",
                result.left.green_score,
                result.right.green_score,
                result.left.brightness,
                result.right.brightness,
                latched_go,
            )
            self.last_state = stable_state
        elif stable_state == "UNKNOWN" and not latched_go:
            rospy.logwarn_throttle(
                5.0,
                "Waiting green route (Lg=%.0f Rg=%.0f bright L=%.0f R=%.0f, need>=%.0f)",
                result.left.green_score,
                result.right.green_score,
                result.left.brightness,
                result.right.brightness,
                self.brightness_threshold,
            )

        state_msg = String(data=stable_state)
        self.state_pub.publish(state_msg)

        raw_msg = String(data=result.route or "none")
        self.raw_pub.publish(raw_msg)

        # 仅锁存 GO 后发布 route，避免 mission 提前读到瞬时 left/right
        route_msg = String(data=latched_route if latched_go else "")
        self.route_pub.publish(route_msg)

        go_msg = Bool(data=latched_go)
        self.go_pub.publish(go_msg)


def main():
    rospy.init_node("traffic_light_ros_node", anonymous=False)
    TrafficLightRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
