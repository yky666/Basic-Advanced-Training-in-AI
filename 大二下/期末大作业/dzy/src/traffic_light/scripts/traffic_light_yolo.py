#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_yolo.py

YOLOv 定位左右红绿灯 + 各侧 ROI 内绿色亮度阈值判定。
有 YOLO 模型时优先用 green_light / red_light 检测框判定（不依赖高亮度 HSV）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class SideResult:
    __slots__ = ("green_score", "red_score", "brightness", "is_green", "is_red", "yolo_green")

    def __init__(self, green_score, red_score, brightness, is_green, is_red, yolo_green=False):
        self.green_score = green_score
        self.red_score = red_score
        self.brightness = brightness
        self.is_green = is_green
        self.is_red = is_red
        self.yolo_green = yolo_green


class FrameResult:
    __slots__ = ("left", "right", "route", "raw_route", "left_roi", "right_roi")

    def __init__(self, left, right, route, raw_route, left_roi, right_roi):
        self.left = left
        self.right = right
        self.route = route
        self.raw_route = raw_route
        self.left_roi = left_roi
        self.right_roi = right_roi


class DualTrafficLightDetector:
    """左右分侧：YOLO 类别 + ROI HSV 亮度（双通道 OR）。"""

    GREEN_SAT = (np.array([35, 60, 60]), np.array([90, 255, 255]))
    GREEN_BLOOM = (np.array([35, 10, 160]), np.array([90, 255, 255]))
    RED_SAT_LOW = (np.array([0, 60, 60]), np.array([12, 255, 255]))
    RED_SAT_HIGH = (np.array([168, 60, 60]), np.array([180, 255, 255]))
    YOLO_GREEN_SCORE = 5000.0

    def __init__(
        self,
        model_path: Optional[str] = None,
        left_roi=(0.0, 0.0, 0.45, 0.70),
        right_roi=(0.45, 0.0, 0.45, 0.70),
        brightness_threshold: float = 140.0,
        min_green_score: float = 800.0,
        min_red_score: float = 400.0,
        yolo_conf: float = 0.5,
        use_yolo_roi: bool = True,
        use_yolo_class_lights: bool = True,
    ):
        self.left_roi_norm = left_roi
        self.right_roi_norm = right_roi
        self.brightness_threshold = float(brightness_threshold)
        self.min_green_score = float(min_green_score)
        self.min_red_score = float(min_red_score)
        self.yolo_conf = float(yolo_conf)
        self.use_yolo_roi = bool(use_yolo_roi)
        self.use_yolo_class_lights = bool(use_yolo_class_lights)
        self.model = None
        self.last_debug: Dict = {}

        if model_path and Path(model_path).is_file() and YOLO is not None:
            self.model = YOLO(model_path)
            self.class_names = {v.lower(): k for k, v in self.model.names.items()}
        elif model_path and not Path(model_path).is_file():
            print(f"[traffic_light_yolo] model not found: {model_path}, use fixed ROI")

    def _norm_roi_to_px(self, frame, roi_norm):
        h, w = frame.shape[:2]
        rx, ry, rw, rh = roi_norm
        x0 = max(0, min(w - 1, int(rx * w)))
        y0 = max(0, min(h - 1, int(ry * h)))
        x1 = max(x0 + 1, min(w, int((rx + rw) * w)))
        y1 = max(y0 + 1, min(h, int((ry + rh) * h)))
        return x0, y0, x1, y1

    def _yolo_predict(self, frame):
        if self.model is None:
            return None
        results = self.model.predict(source=frame, verbose=False, conf=self.yolo_conf)
        if not results:
            return None
        return results[0]

    def _parse_yolo(self, yolo_result, frame_shape):
        left_box = right_box = None
        left_green_conf = right_green_conf = 0.0
        left_red_conf = right_red_conf = 0.0

        if yolo_result is None or yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return left_box, right_box, left_green_conf, right_green_conf, left_red_conf, right_red_conf

        h, w = frame_shape[:2]
        mid_x = w * 0.5

        for box in yolo_result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_result.names[cls_id].lower()
            if "sign_" in name:
                continue

            cx = (x1 + x2) * 0.5
            is_left = cx < mid_x
            pad = 4
            bx = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))

            if name == "green_light" and conf >= self.yolo_conf:
                if is_left:
                    left_green_conf = max(left_green_conf, conf)
                else:
                    right_green_conf = max(right_green_conf, conf)
            elif name == "red_light" and conf >= self.yolo_conf:
                if is_left:
                    left_red_conf = max(left_red_conf, conf)
                else:
                    right_red_conf = max(right_red_conf, conf)

            if "light" not in name and "red" not in name and "green" not in name:
                continue

            if "left" in name or ("green" in name and is_left) or ("red" in name and is_left):
                if left_box is None or (x2 - x1) * (y2 - y1) > (left_box[2] - left_box[0]) * (left_box[3] - left_box[1]):
                    left_box = bx
            if "right" in name or ("green" in name and not is_left) or ("red" in name and not is_left):
                area = (x2 - x1) * (y2 - y1)
                if right_box is None or area > (right_box[2] - right_box[0]) * (right_box[3] - right_box[1]):
                    right_box = bx

        return left_box, right_box, left_green_conf, right_green_conf, left_red_conf, right_red_conf

    def _merge_yolo_green(self, side: SideResult, green_conf: float, red_conf: float) -> SideResult:
        strong_green_min = max(self.yolo_conf + 0.10, 0.65)
        green_min = strong_green_min if red_conf < self.yolo_conf else self.yolo_conf
        if (
            red_conf >= self.yolo_conf
            and green_conf >= self.yolo_conf
            and green_conf < strong_green_min
        ):
            return SideResult(
                side.green_score,
                max(side.red_score, red_conf * 2000.0),
                side.brightness,
                False,
                True,
                False,
            )
        if not self.use_yolo_class_lights or green_conf < green_min:
            if red_conf >= self.yolo_conf:
                return SideResult(
                    side.green_score,
                    max(side.red_score, red_conf * 2000.0),
                    side.brightness,
                    False,
                    True,
                    side.yolo_green,
                )
            if green_conf >= self.yolo_conf and green_conf < strong_green_min:
                return SideResult(
                    side.green_score,
                    side.red_score,
                    side.brightness,
                    False,
                    side.is_red,
                    False,
                )
            return side

        score = max(side.green_score, green_conf * self.YOLO_GREEN_SCORE)
        return SideResult(
            score,
            side.red_score,
            max(side.brightness, 200.0),
            True,
            False,
            True,
        )

    def _color_scores(self, roi_bgr) -> SideResult:
        if roi_bgr.size == 0:
            return SideResult(0, 0, 0, False, False)

        blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        green_mask = cv2.bitwise_or(
            cv2.inRange(hsv, *self.GREEN_SAT),
            cv2.inRange(hsv, *self.GREEN_BLOOM),
        )
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, *self.RED_SAT_LOW),
            cv2.inRange(hsv, *self.RED_SAT_HIGH),
        )

        green_pixels = int(cv2.countNonZero(green_mask))
        red_pixels = int(cv2.countNonZero(red_mask))
        green_brightness = float(v[green_mask > 0].mean()) if green_pixels > 0 else 0.0
        red_brightness = float(v[red_mask > 0].mean()) if red_pixels > 0 else 0.0
        mean_v = float(v.mean())

        green_score = green_pixels * (green_brightness / 255.0)
        red_score = red_pixels * (red_brightness / 255.0)

        is_green = (
            green_score >= self.min_green_score
            and green_brightness >= self.brightness_threshold
            and green_score > red_score * 1.2
        )
        is_red = red_score >= self.min_red_score and red_score > green_score * 1.2

        return SideResult(green_score, red_score, mean_v, is_green, is_red)

    def _pick_route(self, left: SideResult, right: SideResult) -> Optional[str]:
        if left.is_green and right.is_green:
            return "right"
        if left.is_green:
            return "left"
        if right.is_green:
            return "right"
        return None

    def detect(self, frame, debug_frame=None) -> FrameResult:
        yolo_result = self._yolo_predict(frame)
        left_box, right_box, lg_conf, rg_conf, lr_conf, rr_conf = self._parse_yolo(
            yolo_result, frame.shape
        )

        left_px = self._norm_roi_to_px(frame, self.left_roi_norm)
        right_px = self._norm_roi_to_px(frame, self.right_roi_norm)

        if self.use_yolo_roi and left_box is not None:
            left_px = left_box
        if self.use_yolo_roi and right_box is not None:
            right_px = right_box

        lx0, ly0, lx1, ly1 = left_px
        rx0, ry0, rx1, ry1 = right_px
        left_roi = frame[ly0:ly1, lx0:lx1]
        right_roi = frame[ry0:ry1, rx0:rx1]

        left = self._merge_yolo_green(self._color_scores(left_roi), lg_conf, lr_conf)
        right = self._merge_yolo_green(self._color_scores(right_roi), rg_conf, rr_conf)
        route = self._pick_route(left, right)

        self.last_debug = {"left": left, "right": right, "lg_conf": lg_conf, "rg_conf": rg_conf}

        if debug_frame is not None:
            cv2.rectangle(debug_frame, (lx0, ly0), (lx1, ly1), (255, 200, 0), 2)
            cv2.rectangle(debug_frame, (rx0, ry0), (rx1, ry1), (0, 200, 255), 2)
            self._draw_side(debug_frame, lx0, ly0, left, "L")
            self._draw_side(debug_frame, rx0, ry0, right, "R")
            if lg_conf > 0 or rg_conf > 0:
                cv2.putText(
                    debug_frame,
                    f"YOLO g L={lg_conf:.2f} R={rg_conf:.2f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            route_text = route or "wait"
            cv2.putText(
                debug_frame,
                f"route={route_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if route else (128, 128, 128),
                2,
            )

        return FrameResult(
            left=left,
            right=right,
            route=route,
            raw_route=route,
            left_roi=left_px,
            right_roi=right_px,
        )

    @staticmethod
    def _draw_side(frame, x0, y0, side: SideResult, tag: str):
        color = (0, 255, 0) if side.is_green else (0, 0, 255) if side.is_red else (180, 180, 180)
        yolo_tag = " Y" if side.yolo_green else ""
        cv2.putText(
            frame,
            f"{tag} G:{side.green_score:.0f} B:{side.brightness:.0f}{yolo_tag}",
            (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def decide_route(left: SideResult, right: SideResult) -> Optional[str]:
    det = DualTrafficLightDetector()
    return det._pick_route(left, right)
