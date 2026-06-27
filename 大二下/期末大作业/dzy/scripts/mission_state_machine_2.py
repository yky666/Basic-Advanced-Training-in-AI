#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
import time
import cv2
import numpy as np
from pathlib import Path

import actionlib
import rospy
import tf
import yaml
import roslaunch
import roslaunch.rlutil
import roslaunch.parent
from actionlib_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class MissionStateMachine:
    def __init__(self):
        rospy.init_node("mission_state_machine", anonymous=True)

        self.bridge = CvBridge()
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.tf_listener = tf.TransformListener()

        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        self.camera_topic = rospy.get_param("~camera_topic", "/camera/color/image_raw")
        self.traffic_can_start_topic = rospy.get_param("~traffic_can_start_topic", "/traffic_light/can_start")
        self.traffic_lane_topic = rospy.get_param("~traffic_lane_topic", "/traffic_light/selected_lane")
        self.traffic_state_topic = rospy.get_param("~traffic_state_topic", "/traffic_light/state")
        self.line_done_topic = rospy.get_param("~line_done_topic", "/line_follow/done")
        self.vision_result_topic = rospy.get_param("~vision_result_topic", "/vision_result")

        self.route_a1_b1_file = rospy.get_param(
            "~route_a1_b1_file",
            str(Path(__file__).resolve().parent.parent / "config" / "route_a1_b1.yaml"),
        )
        self.route_a1_b2_file = rospy.get_param("~route_a1_b2_file", str(Path(__file__).resolve().parent.parent / "config" / "route_a1_b2.yaml"))
        self.route_a2_b1_file = rospy.get_param("~route_a2_b1_file", str(Path(__file__).resolve().parent.parent / "config" / "route_a2_b1.yaml"))
        self.route_a2_b2_file = rospy.get_param("~route_a2_b2_file", str(Path(__file__).resolve().parent.parent / "config" / "route_a2_b2.yaml"))
        self.line_follow_launch = rospy.get_param(
            "~line_follow_launch",
            str(Path(__file__).resolve().parent.parent / "src" / "vision_line_follow" / "launch" / "line_follow.launch"),
        )

        # 路由与分支：traffic_light 决定 A1/A2，文字识别决定 B1/B2
        self.route_param = str(rospy.get_param("~route", "auto")).strip().lower()
        self.fallback_route = str(rospy.get_param("~fallback_route", "a1")).strip().lower()
        self.text_branch_face = str(rospy.get_param("~text_branch_face", "b2")).strip().lower()
        self.text_branch_fruit = str(rospy.get_param("~text_branch_fruit", "b1")).strip().lower()

        # waypoint ??????????????? a1/b1?a1/b2?a2/b1?a2/b2
        self.traffic_view_waypoint = rospy.get_param("~traffic_view_waypoint", "p001")
        self.a_text_view_waypoint = rospy.get_param("~a_text_view_waypoint", "p004")
        self.a_to_b2_mid_waypoint = rospy.get_param("~a_to_b2_mid_waypoint", "p005")
        self.b2_face_waypoint = rospy.get_param("~b2_face_waypoint", "p006")
        self.b2_to_b1_mid_waypoint = rospy.get_param("~b2_to_b1_mid_waypoint", "p007")
        self.b1_fruit_waypoint = rospy.get_param("~b1_fruit_waypoint", "p008")
        self.b1_to_c_mid_waypoint = rospy.get_param("~b1_to_c_mid_waypoint", "p009")
        self.c_pre_align_waypoint = rospy.get_param("~c_pre_align_waypoint", "p010")
        self.c_line_start_waypoint = rospy.get_param("~c_line_start_waypoint", "p011")
        self.ramp_entry_waypoint = rospy.get_param("~ramp_entry_waypoint", "p012")
        self.ramp_top_waypoint = rospy.get_param("~ramp_top_waypoint", "p013")
        self.ramp_exit_waypoint = rospy.get_param("~ramp_exit_waypoint", "p014")
        self.d_report_waypoint = rospy.get_param("~d_report_waypoint", "p015")

        self.text_model_path = rospy.get_param("~text_model_path", "")
        # B???????????????? B1 ?? B2???????????
        self.semantic_script = rospy.get_param(
            "~semantic_script",
            str(Path(__file__).resolve().parent.parent / "src" / "image_reccognition_code" / "scripts" / "image_recognition_node.py"),
        )
        self.semantic_model_path = rospy.get_param("~semantic_model_path", "")
        self.semantic_capture_max_tries = int(rospy.get_param("~semantic_capture_max_tries", 6))
        self.semantic_capture_timeout = float(rospy.get_param("~semantic_capture_timeout", 18.0))
        self.text_confidence_threshold = float(rospy.get_param("~text_confidence_threshold", 0.55))
        self.semantic_invalid_terms = self._parse_list(rospy.get_param("~semantic_invalid_terms", "未识别,看不清,无法判断,不确定,没有,不是,unknown,none"))

        self.traffic_timeout = float(rospy.get_param("~traffic_timeout", 10.0))
        self.camera_timeout = float(rospy.get_param("~camera_timeout", 5.0))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 90.0))
        self.position_tolerance = float(rospy.get_param("~position_tolerance", 0.22))
        self.yaw_tolerance = float(rospy.get_param("~yaw_tolerance", 0.35))
        self.wait_after_goal = float(rospy.get_param("~wait_after_goal", 0.0))
        self.publish_report_topic = rospy.get_param("~publish_report_topic", "/mission/report")
        self.speak_report = self.parse_bool(rospy.get_param("~speak_report", True))
        self.use_vision_tts = self.parse_bool(rospy.get_param("~use_vision_tts", True))
        self.show_debug_window = self.parse_bool(rospy.get_param("~show_debug_window", True))
        self.semantic_debug_window = self.parse_bool(rospy.get_param("~semantic_debug_window", True))
        self.debug_wait_ms = int(rospy.get_param("~debug_wait_ms", 1))

        self.latest_image = None
        self.latest_traffic_can_start = None
        self.latest_traffic_lane = None
        self.latest_traffic_state = None
        self.traffic_gate_wait_timeout = float(rospy.get_param("~traffic_gate_wait_timeout", 3.2))
        self.latest_vision_result = None
        self.frame_counter = 0

        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self._image_cb, queue_size=1)
        self.traffic_can_start_subs = []
        self.traffic_lane_subs = []
        self.traffic_state_subs = []
        for topic in self._topic_candidates(self.traffic_can_start_topic, [
            "/traffic_light/can_start",
            "/yolo_traffic_light_node/can_start",
            "/traffic_light_node/can_start",
            "traffic_light/can_start",
        ]):
            self.traffic_can_start_subs.append(rospy.Subscriber(topic, Bool, self._traffic_can_start_cb, queue_size=1))
        for topic in self._topic_candidates(self.traffic_lane_topic, [
            "/traffic_light/selected_lane",
            "/yolo_traffic_light_node/selected_lane",
            "/traffic_light_node/selected_lane",
            "traffic_light/selected_lane",
        ]):
            self.traffic_lane_subs.append(rospy.Subscriber(topic, String, self._traffic_lane_cb, queue_size=1))
        for topic in self._topic_candidates(self.traffic_state_topic, [
            "/traffic_light/state",
            "/yolo_traffic_light_node/state",
            "/traffic_light_node/state",
            "traffic_light/state",
        ]):
            self.traffic_state_subs.append(rospy.Subscriber(topic, String, self._traffic_state_cb, queue_size=1))
        self.vision_result_sub = rospy.Subscriber(self.vision_result_topic, String, self._vision_result_cb, queue_size=1)

        self.report_pub = rospy.Publisher(self.publish_report_topic, String, queue_size=1)

        self.external_modules = self._load_external_modules()
        self.predict_text_mode = self.external_modules["predict_text_mode"]
        # ?????? `hdmi_tts.py`?B ????? `vision_ark_tts_node.py`
        self.vision_module = self.external_modules["vision_ark_tts"]
        self.tts_module = self.external_modules["hdmi_tts"]

        rospy.loginfo("Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(20.0)):
            raise RuntimeError("move_base action server not available")
        rospy.loginfo("Connected to move_base.")

        # 四套路由：A1/A2 × B1/B2
        self.route_files = {
            "route_a1_b1": self.route_a1_b1_file,
            "route_a1_b2": self.route_a1_b2_file,
            "route_a2_b1": self.route_a2_b1_file,
            "route_a2_b2": self.route_a2_b2_file,
        }
        self.route_waypoints = {}
        for route_name, route_file in self.route_files.items():
            loaded_name, frame_id, waypoints = self.load_waypoints(route_file)
            self.route_waypoints[route_name] = {
                "file": route_file,
                "loaded_name": loaded_name,
                "frame_id": frame_id,
                "waypoints": waypoints,
                "sequence": self._ordered_waypoint_names(waypoints),
            }
        first_route = self.route_waypoints["route_a1_b1"]
        self.frame_id = first_route["frame_id"]
        self.waypoints = first_route["waypoints"]
        self.current_route_key = None
        self.current_branch = None
        self.buffer = {
            "traffic": None,
            "route": None,
            "text_card": None,
            "semantic": [],
            "report": [],
        }
        self.semantic_route_map = {
            "route_a1_b1": {"p006": "fruit", "p008": "face"},
            "route_a1_b2": {"p006": "face", "p008": "fruit"},
            "route_a2_b1": {"p006": "fruit", "p008": "face"},
            "route_a2_b2": {"p006": "face", "p008": "fruit"},
        }

        rospy.loginfo(
            "Route files: a1_b1=%s, a1_b2=%s, a2_b1=%s, a2_b2=%s",
            self.route_a1_b1_file,
            self.route_a1_b2_file,
            self.route_a2_b1_file,
            self.route_a2_b2_file,
        )
        rospy.loginfo("Loaded route templates: %d", len(self.route_waypoints))
        rospy.loginfo("Camera topic: %s", self.camera_topic)
        rospy.loginfo("Traffic topics: %s / %s", self.traffic_can_start_topic, self.traffic_lane_topic)
        rospy.loginfo("Waypoint mapping: traffic=%s, A=%s, B1=%s, B2=%s, C=%s, D=%s",
                      self.traffic_view_waypoint, self.a_text_view_waypoint, self.b1_fruit_waypoint,
                      self.b2_face_waypoint, self.c_line_start_waypoint, self.d_report_waypoint)

    def parse_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "y"]
        return bool(value)

    def _parse_list(self, value):
        if isinstance(value, (list, tuple)):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        if not value:
            return []
        return [item.strip().lower() for item in str(value).split(",") if item.strip()]

    def _topic_candidates(self, preferred, defaults):
        topics = []
        for topic in [preferred] + list(defaults):
            topic = str(topic).strip()
            if topic and topic not in topics:
                topics.append(topic)
        return topics

    def _ordered_waypoint_names(self, waypoints):
        names = list(waypoints.keys())
        def sort_key(name):
            if name.startswith("p") and name[1:].isdigit():
                return (0, int(name[1:]))
            return (1, name)
        return sorted(names, key=sort_key)

    def _load_module(self, file_path: Path, module_name: str):
        if not file_path.exists():
            raise FileNotFoundError(f"Module file not found: {file_path}")

        module_dir = str(file_path.parent)
        if module_dir not in os.sys.path:
            os.sys.path.insert(0, module_dir)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_external_modules(self):
        root = Path(__file__).resolve().parent.parent
        text_selector_path = root / "src" / "text_recognition" / "scripts" / "text_selector.py"
        vision_ark_path = root / "src" / "voice_result" / "scripts" / "vision_ark_tts_node.py"
        hdmi_tts_path = root / "src" / "voice_result" / "scripts" / "hdmi_tts.py"
        text_selector_module = self._load_module(text_selector_path, "text_selector_runtime")
        if not hasattr(text_selector_module, "predict_text_mode"):
            raise AttributeError(f"text_selector.py does not expose predict_text_mode: {text_selector_path}")
        return {
            "predict_text_mode": text_selector_module.predict_text_mode,
            "vision_ark_tts": self._load_module(vision_ark_path, "vision_ark_tts_runtime"),
            "hdmi_tts": self._load_module(hdmi_tts_path, "hdmi_tts_runtime"),
        }

    def _image_cb(self, msg):
        self.latest_image = msg

    def _traffic_can_start_cb(self, msg):
        self.latest_traffic_can_start = bool(msg.data)

    def _traffic_lane_cb(self, msg):
        self.latest_traffic_lane = msg.data.strip().lower()

    def _traffic_state_cb(self, msg):
        self.latest_traffic_state = msg.data.strip()
        try:
            parsed = json.loads(self.latest_traffic_state)
            if "can_start" in parsed:
                self.latest_traffic_can_start = bool(parsed.get("can_start"))
            lane = str(parsed.get("selected_lane", "")).strip().lower()
            if lane:
                self.latest_traffic_lane = lane
            if "reason" in parsed:
                self.buffer["traffic"] = parsed
        except Exception:
            pass

    def _vision_result_cb(self, msg):
        self.latest_vision_result = msg.data.strip()

    def publish_zero_cmd(self, repeat=5, interval=0.05):
        cmd = Twist()
        for _ in range(repeat):
            self.cmd_pub.publish(cmd)
            rospy.sleep(interval)

    def show_debug_image(self, window_name, image, wait_ms=1):
        # ??????????????????????????????
        if not self.show_debug_window or image is None:
            return
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image)
            cv2.waitKey(wait_ms)
        except Exception as exc:
            rospy.logwarn("Debug window failed: %s", str(exc))

    def load_waypoints(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Waypoints file not found: %s" % path)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise RuntimeError("Waypoints file is empty: %s" % path)

        frame_id = data.get("frame_id", "map")
        route_name = data.get("route_name", Path(path).stem)
        waypoint_list = data.get("waypoints", [])
        if not waypoint_list:
            raise RuntimeError("No waypoints found in file: %s" % path)

        waypoints = {}
        for waypoint in waypoint_list:
            name = waypoint.get("name")
            if not name:
                continue
            waypoints[name] = waypoint
        if not waypoints:
            raise RuntimeError("No valid named waypoints found.")
        return route_name, frame_id, waypoints

    def build_goal(self, point):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(point["x"])
        goal.target_pose.pose.position.y = float(point["y"])
        goal.target_pose.pose.position.z = float(point.get("z", 0.0))

        if "orientation" in point:
            quat = point["orientation"]
            goal.target_pose.pose.orientation.x = float(quat.get("x", 0.0))
            goal.target_pose.pose.orientation.y = float(quat.get("y", 0.0))
            goal.target_pose.pose.orientation.z = float(quat.get("z", 0.0))
            goal.target_pose.pose.orientation.w = float(quat.get("w", 1.0))
        else:
            yaw = float(point.get("yaw", 0.0))
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
            goal.target_pose.pose.orientation.x = qx
            goal.target_pose.pose.orientation.y = qy
            goal.target_pose.pose.orientation.z = qz
            goal.target_pose.pose.orientation.w = qw
        return goal

    def angle_diff(self, a, b):
        diff = a - b
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return diff

    def get_robot_pose(self):
        try:
            trans, rot = self.tf_listener.lookupTransform(self.frame_id, "base_link", rospy.Time(0))
            _, _, yaw = euler_from_quaternion(rot)
            return trans[0], trans[1], yaw
        except Exception:
            return None

    def is_close_enough(self, point):
        pose = self.get_robot_pose()
        if pose is None:
            return False
        rx, ry, ryaw = pose
        tx, ty = float(point["x"]), float(point["y"])
        tyaw = self._target_yaw(point)
        dist = math.hypot(rx - tx, ry - ty)
        yaw_err = abs(self.angle_diff(ryaw, tyaw))
        if self.parse_bool(point.get("position_only", False)):
            return dist <= self.position_tolerance
        return dist <= self.position_tolerance and yaw_err <= self.yaw_tolerance

    def _target_yaw(self, point):
        if "yaw" in point:
            return float(point["yaw"])
        if "orientation" in point:
            q = point["orientation"]
            quat = [float(q.get("x", 0.0)), float(q.get("y", 0.0)), float(q.get("z", 0.0)), float(q.get("w", 1.0))]
            _, _, yaw = euler_from_quaternion(quat)
            return yaw
        return 0.0

    def wait_after_reach(self, name):
        if self.wait_after_goal <= 0:
            return
        rospy.loginfo("Reached waypoint: %s. Waiting %.1f seconds...", name, self.wait_after_goal)
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < self.wait_after_goal:
            rate.sleep()

    def send_goal_and_wait(self, waypoint_name):
        if waypoint_name not in self.waypoints:
            rospy.logerr("Waypoint not found: %s", waypoint_name)
            return False

        point = self.waypoints[waypoint_name]
        goal = self.build_goal(point)
        rospy.loginfo("Navigating to waypoint: %s", waypoint_name)
        self.client.send_goal(goal)

        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.is_close_enough(point):
                self.client.cancel_goal()
                rospy.sleep(0.5)
                self.wait_after_reach(waypoint_name)
                return True

            state = self.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                self.wait_after_reach(waypoint_name)
                return True
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST, GoalStatus.PREEMPTED]:
                rospy.logwarn("Goal failed at %s, state=%d", waypoint_name, state)
                return False

            if (rospy.Time.now() - start).to_sec() > self.goal_timeout:
                rospy.logwarn("Goal timeout at %s", waypoint_name)
                self.client.cancel_goal()
                rospy.sleep(1.0)
                return False
            rate.sleep()
        return False

    def wait_for_traffic(self):
        # ?? traffic_light ???? can_start??????????????????????
        if self.route_param in ["a1", "a2"]:
            self.current_route_key = self.route_param
            return self.current_route_key

        if self.route_param not in ["auto", "traffic", "branch"]:
            rospy.logwarn("Unknown route parameter: %s, fallback to auto.", self.route_param)

        rospy.loginfo("Waiting for traffic light can_start signal...")
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.latest_traffic_can_start is True:
                lane = (self.latest_traffic_lane or "").strip().lower()
                self.buffer["traffic"] = {
                    "can_start": True,
                    "selected_lane": lane,
                    "raw": self.latest_traffic_state,
                }
                if lane in ["a1", "left", "left_green", "route_a1", "route_a1_b1", "route_a1_b2"]:
                    self.current_route_key = "a1"
                elif lane in ["a2", "right", "right_green", "route_a2", "route_a2_b1", "route_a2_b2"]:
                    self.current_route_key = "a2"
                else:
                    self.current_route_key = self.fallback_route if self.fallback_route in ["a1", "a2"] else "a1"
                rospy.loginfo("Traffic gate passed, route=%s", self.current_route_key)
                return self.current_route_key

            if (rospy.Time.now() - start).to_sec() > self.traffic_gate_wait_timeout:
                rospy.logwarn("Traffic gate wait timeout after %.1f s, fallback route=%s", self.traffic_gate_wait_timeout, self.fallback_route)
                self.current_route_key = self.fallback_route if self.fallback_route in ["a1", "a2"] else "a1"
                return self.current_route_key
            rate.sleep()

        return self.fallback_route

    def _snapshot_path(self, prefix):
        output_dir = Path(tempfile.gettempdir()) / "mission_state_machine"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{prefix}_{int(time.time() * 1000)}.jpg"

    def capture_snapshot(self, prefix="capture"):
        start = rospy.Time.now()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                try:
                    frame = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    path = self._snapshot_path(prefix)
                    cv2.imwrite(str(path), frame)
                    return path, frame
                except Exception as exc:
                    rospy.logwarn("Snapshot conversion failed: %s", str(exc))
                    return None, None
            if (rospy.Time.now() - start).to_sec() > self.camera_timeout:
                break
            rate.sleep()
        return None, None

    def recognize_text_branch(self):
        # A ??????????? B1 ?? B2????????????
        snapshot_path, frame = self.capture_snapshot("text_card")
        if snapshot_path is None or frame is None:
            raise RuntimeError("Cannot capture text card image.")

        result = self.predict_text_mode(snapshot_path, self.text_model_path or None)
        self.buffer["text_card"] = result

        try:
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (10, 10), (debug_frame.shape[1] - 10, 90), (0, 0, 0), -1)
            cv2.putText(debug_frame, f"A result: {result.get('mode', 'unknown')} ({float(result.get('confidence', 0.0)):.2f})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"image: {Path(snapshot_path).name}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.show_debug_image("A_Text_Recognition", debug_frame, 1)
        except Exception as exc:
            rospy.logwarn("A debug window failed: %s", str(exc))

        mode = str(result.get("mode", "")).strip().lower()
        confidence = float(result.get("confidence", 0.0))
        if mode not in ["face", "fruit"] or confidence < self.text_confidence_threshold:
            raise RuntimeError(f"Invalid text selector result: {result}")

        if mode == "face":
            self.current_branch = self.text_branch_face
        else:
            self.current_branch = self.text_branch_fruit

        self.buffer["route"] = {
            "mode": mode,
            "confidence": confidence,
            "branch": self.current_branch,
        }
        rospy.loginfo("Text card selected branch=%s mode=%s confidence=%.3f", self.current_branch, mode, confidence)
        return self.current_branch

    def _clean_semantic_text(self, text):
        cleaned = str(text).strip().replace("\n", " ")
        for token in ["。", ",", ";", "；", "：", "!", "！", "?", "？", "\"", "'"]:
            cleaned = cleaned.replace(token, " ")
        cleaned = " ".join(cleaned.split())
        return cleaned.lower()

    def _is_semantic_valid(self, text, expected_mode):
        cleaned = self._clean_semantic_text(text)
        if not cleaned:
            return False
        if any(term in cleaned for term in self.semantic_invalid_terms):
            return False
        if expected_mode == "face":
            return any(name in cleaned for name in ["dengziqi", "邓紫棋", "liuyifei", "刘亦菲", "renxianqi", "任贤齐", "sabeining", "撒贝宁"])
        if expected_mode == "fruit":
            return any(name in cleaned for name in ["apple", "苹果", "banana", "香蕉", "grape", "葡萄", "orange", "橙"])
        return True

    def recognize_semantic_target(self, expected_mode, label=None):
        # B???????????????? vision_ark_tts_node.py ?????
        # ?????????/?????????????????????
        label = label or expected_mode
        start = rospy.Time.now()
        best_text = None
        rospy.loginfo("Capturing semantic result for %s...", label)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > self.semantic_capture_timeout:
                break

            snapshot_path, frame = self.capture_snapshot(prefix=f"semantic_{label}")
            if snapshot_path is None or frame is None:
                rospy.sleep(0.2)
                continue

            try:
                text = self.run_vision_ark_once(frame)
            except Exception as exc:
                rospy.logwarn("Semantic inference failed: %s", str(exc))
                rospy.sleep(0.3)
                continue

            cleaned = str(text).strip()
            best_text = cleaned
            rospy.loginfo("Semantic raw result: %s", cleaned)
            if self._is_semantic_valid(cleaned, expected_mode):
                self.buffer["semantic"].append({
                    "label": label,
                    "expected_mode": expected_mode,
                    "result": cleaned,
                    "image": str(snapshot_path),
                })
                try:
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (10, 10), (debug_frame.shape[1] - 10, 110), (0, 0, 0), -1)
                    cv2.putText(debug_frame, f"B result: {cleaned}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(debug_frame, f"image: {Path(snapshot_path).name}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    self.show_debug_image(f"B_{label}_Recognition", debug_frame, self.debug_wait_ms)
                except Exception as exc:
                    rospy.logwarn("B debug window failed: %s", str(exc))
                return cleaned

            rospy.logwarn("Ignored semantic result: %s", cleaned)
            rospy.sleep(0.3)

        raise RuntimeError(f"Semantic recognition failed for {label}, last={best_text}")

    def run_vision_ark_once(self, frame):
        # B???????????? vision_ark_tts_node.py ?? Ark ??????
        # ??????????????? ark_vision_base64 ?????
        if not hasattr(self.vision_module, "ark_vision_base64"):
            raise RuntimeError("vision_ark_tts_node.py does not expose ark_vision_base64")
        return self._run_async(self.vision_module.ark_vision_base64(frame))

    def resolve_route_key(self, traffic_route, text_branch):
        traffic_route = (traffic_route or "").strip().lower()
        text_branch = (text_branch or "").strip().lower()
        if traffic_route not in ["a1", "a2"]:
            traffic_route = self.fallback_route if self.fallback_route in ["a1", "a2"] else "a1"
        if text_branch not in ["b1", "b2"]:
            text_branch = "b1"
        return f"route_{traffic_route}_{text_branch}"

    def get_route_sequence(self, route_key):
        route_info = self.route_waypoints.get(route_key)
        if not route_info:
            raise KeyError(f"Unknown route key: {route_key}")
        return route_info["sequence"]

    def select_route(self, route_key):
        route_info = self.route_waypoints.get(route_key)
        if not route_info:
            raise KeyError(f"Route file not found for {route_key}")
        self.current_route_key = route_key
        self.frame_id = route_info["frame_id"]
        self.waypoints = route_info["waypoints"]
        self.buffer["route"] = {
            "route_key": route_key,
            "file": route_info["file"],
            "loaded_name": route_info["loaded_name"],
            "sequence": route_info["sequence"],
        }
        rospy.loginfo("Selected route file: %s", route_key)

    def run_route_sequence(self, waypoints, text_after=None, semantic_waypoints=None, line_follow_after=None):
        semantic_waypoints = set(semantic_waypoints or [])
        for waypoint_name in waypoints:
            if waypoint_name not in self.waypoints:
                rospy.logwarn("Waypoint not configured in current route: %s", waypoint_name)
                continue
            self.send_goal_and_wait(waypoint_name)
            if text_after and waypoint_name == text_after:
                self.recognize_text_branch()
            if waypoint_name in semantic_waypoints:
                self.maybe_run_semantic_at_waypoint(waypoint_name)
            if line_follow_after and waypoint_name == line_follow_after:
                if not self.run_line_follow():
                    raise RuntimeError("Line follow failed")

    def maybe_run_semantic_at_waypoint(self, waypoint_name):
        route_key = self.current_route_key or self.buffer.get("route", {}).get("route_key")
        if waypoint_name not in ["p006", "p008"]:
            return None
        for item in reversed(self.buffer["semantic"]):
            if item.get("waypoint") == waypoint_name and item.get("route_key") == route_key:
                return item.get("result")
        expected_map = {
            "route_a1_b1": {"p006": "face", "p008": "fruit"},
            "route_a1_b2": {"p006": "face", "p008": "fruit"},
            "route_a2_b1": {"p006": "fruit", "p008": "face"},
            "route_a2_b2": {"p006": "face", "p008": "fruit"},
        }
        expected_mode = expected_map.get(route_key, {}).get(waypoint_name)
        if expected_mode not in ["face", "fruit"]:
            return None
        rospy.loginfo("Capturing semantic result at %s for %s...", waypoint_name, expected_mode)
        result = self.recognize_semantic_target(expected_mode, label=waypoint_name)
        self.buffer["semantic"].append({
            "waypoint": waypoint_name,
            "route_key": route_key,
            "expected_mode": expected_mode,
            "result": result,
        })
        return result

    def maybe_run_text_at_waypoint(self, waypoint_name):
        if waypoint_name != self.a_text_view_waypoint:
            return None
        return self.recognize_text_branch()

    def run_line_follow(self):
        launch_path = self.resolve_launch_file(self.line_follow_launch)
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        parent = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        rospy.loginfo("Starting line follow launch: %s", launch_path)
        self.client.cancel_goal()
        self.publish_zero_cmd()
        try:
            parent.start()
            rospy.loginfo("Line follow launch started.")
            start = rospy.Time.now()
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                try:
                    msg = rospy.wait_for_message(self.line_done_topic, Bool, timeout=0.3)
                    if msg.data:
                        rospy.loginfo("Line follow done received.")
                        return True
                except rospy.ROSException:
                    pass
                if (rospy.Time.now() - start).to_sec() > 120.0:
                    rospy.logwarn("Line follow timeout.")
                    return False
                rate.sleep()
        finally:
            try:
                parent.shutdown()
            except Exception:
                pass
            self.publish_zero_cmd(repeat=10, interval=0.05)
            rospy.loginfo("Line follow launch stopped.")
        return False

    def resolve_launch_file(self, launch_file):
        resolved = os.path.expanduser(launch_file)
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(resolved)
        if not os.path.exists(resolved):
            raise FileNotFoundError("Launch file not found: %s" % resolved)
        return resolved

    def _run_async(self, coro):
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        return asyncio.run(coro)

    def report_results(self):
        traffic = self.buffer.get("traffic") or {}
        text_card = self.buffer.get("text_card") or {}
        semantic = self.buffer.get("semantic")
        route = self.current_route_key or self.buffer.get("route", {}).get("route_key", "unknown")
        branch = self.current_branch or self.buffer.get("route", {}).get("branch", "unknown")

        report_lines = [
            f"交通灯路线：{route}",
            f"文字识别：{text_card.get('mode', 'unknown')}，置信度 {float(text_card.get('confidence', 0.0)):.2f}",
            f"分支选择：{branch}",
        ]
        for item in semantic:
            report_lines.append(f"{item.get('waypoint', item.get('label', 'target'))}：{item.get('result', '')}")

        report_text = "；".join(report_lines)
        self.buffer["report"] = report_lines
        self.report_pub.publish(String(data=report_text))
        rospy.loginfo("Mission report: %s", report_text)
        try:
            report_img = 255 * np.ones((420, 960, 3), dtype=np.uint8)
            y = 40
            for line in report_lines:
                cv2.putText(report_img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                y += 50
            self.show_debug_image("D_Report", report_img, 1)
        except Exception:
            pass
        if self.speak_report:
            try:
                self.tts_module.tts_speak(report_text)
            except Exception:
                try:
                    self.tts_module.tts_speak(report_text.replace("；", ","))
                except Exception as exc:
                    rospy.logwarn("TTS failed: %s", str(exc))
        return report_text

    def run(self):
        rospy.loginfo("========== Mission Start ==========")
        traffic_route = self.wait_for_traffic()
        if not self.current_branch:
            self.current_branch = self.text_branch_fruit

        route_key = self.resolve_route_key(traffic_route, self.current_branch)
        self.select_route(route_key)
        route_sequence = self.get_route_sequence(route_key)
        self.run_route_sequence(
            route_sequence,
            text_after=self.a_text_view_waypoint,
            semantic_waypoints=[self.b2_face_waypoint, self.b1_fruit_waypoint],
            line_follow_after=self.c_line_start_waypoint,
        )

        self.report_results()
        self.publish_zero_cmd()
        rospy.loginfo("========== Mission Finished ==========")
        if self.show_debug_window:
            cv2.destroyAllWindows()

    def cancel_current_goal(self):
        self.client.cancel_goal()
        self.publish_zero_cmd()


if __name__ == "__main__":
    sm = None
    try:
        sm = MissionStateMachine()
        sm.run()
    except KeyboardInterrupt:
        if sm is not None:
            sm.cancel_current_goal()
        rospy.logwarn("Interrupted by user.")
    except Exception as exc:
        if sm is not None:
            sm.cancel_current_goal()
        rospy.logerr("Error: %s", str(exc))
        raise
