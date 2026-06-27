#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


PACKAGE_NAME = "traffic_light"
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_PATH = SCRIPT_PATH.parents[1]
try:
    import rospkg

    PACKAGE_PATH = Path(rospkg.RosPack().get_path(PACKAGE_NAME))
except Exception:
    PACKAGE_PATH = SOURCE_PATH

if str(PACKAGE_PATH) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PATH))
if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))

from yolo_detector import YoloTrafficLightDetection, YoloTrafficLightDetector  # noqa: E402


class TrafficLightStartGate:
    def __init__(self, hold_seconds: float = 3.0, prefer_lane: str = "left") -> None:
        self.hold_seconds = hold_seconds
        self.prefer_lane = prefer_lane
        self._green_since: Optional[float] = None
        self._candidate_lane: Optional[str] = None

    def update(self, state: Dict[str, str], now: Optional[float] = None):
        now = rospy.Time.now().to_sec() if now is None else now
        green_lanes = [lane for lane, color in state.items() if color == "green"]
        if not green_lanes:
            self._green_since = None
            self._candidate_lane = None
            return False, None, "no green light"

        lane = self._choose_lane(green_lanes)
        if lane != self._candidate_lane:
            self._candidate_lane = lane
            self._green_since = now

        elapsed = now - (self._green_since or now)
        if elapsed >= self.hold_seconds:
            return True, lane, f"green stable for {elapsed:.1f}s"
        return False, lane, f"waiting green stable: {elapsed:.1f}s"

    def _choose_lane(self, green_lanes: List[str]) -> str:
        if self.prefer_lane in green_lanes:
            return self.prefer_lane
        return sorted(green_lanes)[0]


class YoloTrafficLightNode:
    def __init__(self) -> None:
        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        weights_path = rospy.get_param(
            "~weights_path",
            str(PACKAGE_PATH / "weights" / "best.pt"),
        )
        publish_debug_image = bool(rospy.get_param("~publish_debug_image", True))
        show_window = bool(rospy.get_param("~show_window", False))
        hold_seconds = float(rospy.get_param("~hold_seconds", 3.0))
        prefer_lane = str(rospy.get_param("~prefer_lane", "left"))
        conf_threshold = float(rospy.get_param("~conf_threshold", 0.35))
        iou_threshold = float(rospy.get_param("~iou_threshold", 0.45))
        imgsz = int(rospy.get_param("~imgsz", 640))
        device_param = str(rospy.get_param("~device", "")).strip()
        device = device_param if device_param else None
        left_right_split = float(rospy.get_param("~left_right_split", 0.5))

        self.bridge = CvBridge()
        self.detector = YoloTrafficLightDetector(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            device=device,
            left_right_split=left_right_split,
        )
        self.start_gate = TrafficLightStartGate(
            hold_seconds=hold_seconds,
            prefer_lane=prefer_lane,
        )
        self.publish_debug_image = publish_debug_image
        self.show_window = show_window

        self.state_pub = rospy.Publisher("~state", String, queue_size=1)
        self.start_pub = rospy.Publisher("~can_start", Bool, queue_size=1)
        self.lane_pub = rospy.Publisher("~selected_lane", String, queue_size=1)
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
        self.latest_depth = None

        rospy.loginfo("yolo_traffic_light_node started, image_topic=%s", image_topic)
        rospy.loginfo("traffic_light depth_topic=%s", depth_topic)
        rospy.loginfo("traffic_light weights=%s", weights_path)
        rospy.loginfo(
            "traffic_light yolo params: conf=%.2f iou=%.2f imgsz=%d device=%s split=%.2f",
            conf_threshold,
            iou_threshold,
            imgsz,
            device or "auto",
            left_right_split,
        )

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth = msg

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge conversion failed: %s", exc)
            return

        detections = self.detector.detect(frame)
        state = self.detector.state_from_detections(detections)
        stamp = msg.header.stamp.to_sec()
        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()
        can_start, lane, reason = self.start_gate.update(state, now=stamp)

        payload = {
            "stamp": msg.header.stamp.to_sec(),
            "state": state,
            "can_start": can_start,
            "selected_lane": lane or "",
            "reason": reason,
            "detections": [self._detection_to_dict(item) for item in detections],
        }

        payload_json = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(String(data=payload_json))
        self.start_pub.publish(Bool(data=can_start))
        self.lane_pub.publish(String(data=lane or ""))

        if self.publish_debug_image and self.debug_pub.get_num_connections() > 0:
            debug = self.detector.draw(frame, detections)
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
            except CvBridgeError as exc:
                rospy.logwarn("debug image publish failed: %s", exc)

        if self.show_window:
            cv2.imshow("yolo_traffic_light", self.detector.draw(frame, detections))
            cv2.waitKey(1)

    @staticmethod
    def _detection_to_dict(detection: YoloTrafficLightDetection) -> Dict[str, object]:
        x, y, w, h = detection.bbox
        cx, cy = detection.center
        return {
            "side": detection.side,
            "color": detection.color,
            "center": {"x": cx, "y": cy},
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "score": round(detection.score, 4),
            "class_id": detection.class_id,
        }


def main() -> None:
    rospy.init_node("yolo_traffic_light_node")
    YoloTrafficLightNode()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
