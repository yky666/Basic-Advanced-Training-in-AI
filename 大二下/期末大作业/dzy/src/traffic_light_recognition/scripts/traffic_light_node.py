#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_node.py

融合 HSV 颜色检测、轮廓筛选以及状态机框架的交通灯识别脚本（红/绿两色）。
支持 ROS 摄像头实时检测与静态图片测试，针对 LED 高亮/过曝中心做了适配。

用法:
  python scripts/traffic_light_node.py                    # 默认使用 ROS 摄像头
  python scripts/traffic_light_node.py --image test.png   # 测试单张图片
  python scripts/traffic_light_node.py --image test.png --debug --output out.png
按 'q' 键退出（图片模式按任意键）。
"""

import argparse
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# 导入 ROS 库
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge


def imread_unicode(path: str):
    """兼容 Windows 中文路径的图像读取。"""
    data = Path(path).read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """兼容 Windows 中文路径的图像保存。"""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    Path(path).write_bytes(buf.tobytes())
    return True


class TrafficLightDetector:
    """交通灯检测器：HSV 双阈值（饱和色 + LED 过曝）+ 轮廓评分"""

    # 饱和色区间（常规亮灯）
    SAT_RANGES = {
        "red_low": (np.array([0, 70, 80]), np.array([12, 255, 255])),
        "red_high": (np.array([168, 70, 80]), np.array([180, 255, 255])),
        "green": (np.array([35, 70, 80]), np.array([90, 255, 255])),
    }

    # LED 中心过曝：色相保留、饱和度低、亮度极高
    BLOOM_RANGES = {
        "red_low": (np.array([0, 15, 180]), np.array([12, 255, 255])),
        "red_high": (np.array([168, 15, 180]), np.array([180, 255, 255])),
        "green": (np.array([35, 15, 180]), np.array([90, 255, 255])),
    }

    MIN_CIRCULARITY = 0.35
    MAX_ASPECT_RATIO = 3.0
    MIN_MEAN_VALUE = 110          # 轮廓区域平均亮度，过滤未点亮的暗色灯罩
    MIN_SCORE_RATIO = 1.8         # 胜出颜色得分需达到次优色的倍数
    MIN_WIN_SCORE = 200.0         # 最低有效得分，抑制噪声误判

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.last_debug = {}

    def _area_bounds(self, frame_shape):
        h, w = frame_shape[:2]
        img_area = w * h
        min_area = max(80, int(0.0001 * img_area))
        max_area = max(3000, int(0.01 * img_area))
        return min_area, max_area

    def _preprocess(self, frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def _build_mask(self, hsv, color):
        if color == "red":
            sat = cv2.bitwise_or(
                cv2.inRange(hsv, *self.SAT_RANGES["red_low"]),
                cv2.inRange(hsv, *self.SAT_RANGES["red_high"]),
            )
            bloom = cv2.bitwise_or(
                cv2.inRange(hsv, *self.BLOOM_RANGES["red_low"]),
                cv2.inRange(hsv, *self.BLOOM_RANGES["red_high"]),
            )
        else:
            sat = cv2.inRange(hsv, *self.SAT_RANGES["green"])
            bloom = cv2.inRange(hsv, *self.BLOOM_RANGES["green"])

        mask = cv2.bitwise_or(sat, bloom)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def _contour_metrics(self, contour):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        return area, (x, y, w, h), aspect_ratio, circularity

    def _score_color(self, hsv, color, frame_shape, draw_frame=None):
        mask = self._build_mask(hsv, color)
        min_area, max_area = self._area_bounds(frame_shape)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_score = 0.0
        valid_contours = []

        for contour in contours:
            area, (x, y, w, h), aspect_ratio, circularity = self._contour_metrics(contour)
            if area < min_area or area > max_area:
                continue
            if aspect_ratio > self.MAX_ASPECT_RATIO:
                continue
            if circularity < self.MIN_CIRCULARITY:
                continue

            roi_value = hsv[y : y + h, x : x + w, 2]
            if roi_value.size == 0 or float(roi_value.mean()) < self.MIN_MEAN_VALUE:
                continue

            score = area * circularity
            total_score += score
            valid_contours.append((contour, score))

            if draw_frame is not None:
                color_bgr = (0, 0, 255) if color == "red" else (0, 255, 0)
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), color_bgr, 2)
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(draw_frame, (cx, cy), 3, (255, 255, 255), -1)

        return total_score, valid_contours, mask

    def detect_color(self, frame, debug_frame=None):
        """
        返回 ('red'|'green'|None, scores_dict)。
        使用加权得分而非简单优先级，减少暗色灯罩与侧向反光干扰。
        """
        processed = self._preprocess(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        red_score, red_contours, red_mask = self._score_color(
            hsv, "red", frame.shape, debug_frame
        )
        green_score, green_contours, green_mask = self._score_color(
            hsv, "green", frame.shape, debug_frame
        )

        scores = {"red": red_score, "green": green_score}
        self.last_debug = {
            "red_mask": red_mask,
            "green_mask": green_mask,
            "red_contours": red_contours,
            "green_contours": green_contours,
            "scores": scores,
        }

        best_color = None
        best_score = 0.0
        second_score = 0.0
        for color, score in scores.items():
            if score > best_score:
                second_score = best_score
                best_score = score
                best_color = color
            elif score > second_score:
                second_score = score

        if best_score < self.MIN_WIN_SCORE:
            return None, scores

        if second_score > 0 and best_score < second_score * self.MIN_SCORE_RATIO:
            return None, scores

        return best_color, scores


class TrafficLightStateMachine:
    """状态机：连续多帧一致才切换，抑制单帧抖动。"""

    def __init__(self, stable_frames=5):
        self.stable_frames = stable_frames
        self.current_state = "UNKNOWN"
        self.history = deque(maxlen=stable_frames)

    def update(self, detected_color):
        if detected_color == "red":
            raw_state = "RED"
        elif detected_color == "green":
            raw_state = "GREEN"
        else:
            raw_state = "UNKNOWN"

        self.history.append(raw_state)

        if len(self.history) == self.stable_frames:
            first = self.history[0]
            if all(state == first for state in self.history):
                self.current_state = first
                self.history.clear()
                self.history.append(first)

        return self.current_state

    def get_state(self):
        return self.current_state


def draw_overlay(frame, stable_state, detected_color, scores):
    # 👇 核心修改：把中文替换成了纯英文
    color_display = {
        "RED": ("RED", (0, 0, 255)),
        "GREEN": ("GREEN", (0, 255, 0)),
        "UNKNOWN": ("UNKNOWN", (128, 128, 128)),
    }

    text, color_bgr = color_display[stable_state]
    cv2.putText(
        frame,
        f"Traffic Light: {text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_bgr,
        2,
    )

    if detected_color:
        raw_text = detected_color.upper()
        cv2.putText(
            frame,
            f"Raw: {raw_text}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        frame,
        f"Score R:{scores['red']:.0f} G:{scores['green']:.0f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )


def make_debug_panel(frame, detector):
    h, w = frame.shape[:2]
    red_mask = detector.last_debug.get("red_mask")
    green_mask = detector.last_debug.get("green_mask")
    if red_mask is None or green_mask is None:
        return frame

    red_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    green_bgr = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    panel = np.hstack([red_bgr, green_bgr])
    panel = cv2.resize(panel, (w, h // 4))
    return np.vstack([frame, panel])


def parse_args():
    parser = argparse.ArgumentParser(description="交通灯红/绿识别")
    parser.add_argument("--image", type=str, help="测试图片路径（不指定则使用摄像头）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号（ROS环境下已失效，改用话题）")
    parser.add_argument("--stable-frames", type=int, default=5, help="状态机稳定帧数")
    parser.add_argument("--debug", action="store_true", help="显示掩码调试面板")
    parser.add_argument("--output", type=str, help="保存标注结果到文件")
    parser.add_argument("--headless", action="store_true", help="不弹出窗口，仅打印/保存结果")
    return parser.parse_args()


def process_frame(frame, detector, state_machine, show_debug=False):
    debug_frame = frame.copy() if show_debug else None
    detected, scores = detector.detect_color(frame, debug_frame=debug_frame)
    stable_state = state_machine.update(detected)

    display = debug_frame if show_debug else frame.copy()
    draw_overlay(display, stable_state, detected, scores)
    if show_debug:
        display = make_debug_panel(display, detector)
    return display, detected, stable_state, scores


def run_image(path, detector, state_machine, show_debug=False, output_path=None, headless=False):
    frame = imread_unicode(path)
    if frame is None:
        print(f"错误：无法读取图片 {path}")
        return 1

    debug_frame = frame.copy() if show_debug else None
    detected, scores = detector.detect_color(frame, debug_frame=debug_frame)
    for _ in range(state_machine.stable_frames):
        stable_state = state_machine.update(detected)

    display = debug_frame if show_debug else frame.copy()
    draw_overlay(display, stable_state, detected, scores)
    if show_debug:
        display = make_debug_panel(display, detector)

    print(f"图片: {path}")
    print(f"原始检测: {detected or 'none'}")
    print(f"稳定状态: {stable_state}")
    print(f"得分: red={scores['red']:.1f}, green={scores['green']:.1f}")

    if output_path:
        if imwrite_unicode(output_path, display):
            print(f"结果已保存: {output_path}")
        else:
            print(f"保存失败: {output_path}")

    if not headless:
        cv2.imshow("Traffic Light Detection", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# =========================================================================
# 核心修改：将 OpenCV 摄像头读取替换为 ROS 话题订阅
# =========================================================================
def run_camera(camera_id, detector, state_machine, show_debug=False):
    # 确保 ROS 节点已初始化
    try:
        rospy.init_node("traffic_light_recognition_node", anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    bridge = CvBridge()
    shared_data = {"frame": None, "new": False}
    frame_lock = threading.Lock()

    # ROS 图像回调函数
    def image_callback(msg):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            with frame_lock:
                shared_data["frame"] = cv_img
                shared_data["new"] = True
        except Exception as e:
            rospy.logwarn(f"Image transfer error: {e}")

    # 订阅相机 RGB 话题
    topic_name = "/camera/color/image_raw"
    rospy.Subscriber(topic_name, RosImage, image_callback, queue_size=1)
    print(f"已订阅 ROS 相机话题: {topic_name}. 正在等待图像...")
    print("按 'q' 退出程序")

    while not rospy.is_shutdown():
        frame = None
        # 安全地从子线程获取最新一帧图像
        with frame_lock:
            if shared_data["new"] and shared_data["frame"] is not None:
                frame = shared_data["frame"].copy()
                shared_data["new"] = False

        # 如果没有获取到新画面，暂时休眠并处理 OpenCV 键盘事件，防止界面卡死
        if frame is None:
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        display, _, _, _ = process_frame(frame, detector, state_machine, show_debug=show_debug)
        cv2.imshow("Traffic Light Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return 0


def main():
    args = parse_args()
    detector = TrafficLightDetector()
    state_machine = TrafficLightStateMachine(stable_frames=args.stable_frames)

    if args.image:
        return run_image(
            args.image,
            detector,
            state_machine,
            show_debug=args.debug,
            output_path=args.output,
            headless=args.headless,
        )
    return run_camera(args.camera, detector, state_machine, show_debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())