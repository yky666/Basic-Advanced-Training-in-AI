#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict whether a photographed text card means face or fruit.
Now supports real-time ROS camera testing for debugging accuracy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

# 确保能找到同级目录下的模块
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_text_selector import MODEL_PATH, load_image_features


# =========================================================================
# 核心推理模块 (重构为类，避免摄像头模式下每帧都重新加载模型)
# =========================================================================
class TextPredictor:
    def __init__(self, model_path: str | Path | None = None):
        self.selected_model_path = Path(model_path) if model_path else MODEL_PATH
        if not self.selected_model_path.exists():
            raise FileNotFoundError(
                f"Text selector model not found: {self.selected_model_path}. "
                "Run `python scripts/train_text_selector.py` first."
            )

        print(f"Loading text selector model from: {self.selected_model_path}")
        payload = joblib.load(self.selected_model_path)
        self.model = payload["model"]
        self.feature_size = int(payload.get("feature_size", 96))
        self.class_names = list(self.model.classes_)
        
        # 使用内存盘提升实时暂存速度，防止损伤硬盘
        self.tmp_path = "/dev/shm/tmp_text_card.jpg" if os.path.exists("/dev/shm") else "/tmp/tmp_text_card.jpg"

    def predict_image_file(self, image_path: str | Path) -> dict:
        """用于单张静态图片的预测（兼容旧版逻辑）"""
        image_path = Path(image_path)
        features = load_image_features(image_path, size=self.feature_size).reshape(1, -1)
        return self._do_predict(features, image_path)

    def predict_frame(self, frame: np.ndarray) -> dict:
        """用于摄像头的实时预测"""
        # 将当前帧高速保存到内存盘，借用原有的特征提取函数
        cv2.imwrite(self.tmp_path, frame)
        features = load_image_features(Path(self.tmp_path), size=self.feature_size).reshape(1, -1)
        return self._do_predict(features, self.tmp_path)

    def _do_predict(self, features: np.ndarray, source_path) -> dict:
        probabilities = self.model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        return {
            "mode": self.class_names[best_index],
            "confidence": float(probabilities[best_index]),
            "probabilities": {name: float(prob) for name, prob in zip(self.class_names, probabilities)},
            "image": str(source_path),
            "model_path": str(self.selected_model_path),
        }


# =========================================================================
# UI 与状态机模块 (用于实时调试)
# =========================================================================
class ModeStateMachine:
    """平滑预测结果，避免因画面抖动导致的模式乱跳"""
    def __init__(self, stable_frames=5):
        self.stable_frames = stable_frames
        self.history = deque(maxlen=stable_frames)
        self.current_state = "UNKNOWN"

    def update(self, detected_mode):
        self.history.append(detected_mode)
        if len(self.history) == self.stable_frames:
            first = self.history[0]
            if all(state == first for state in self.history):
                self.current_state = first
        return self.current_state


def draw_debug_ui(frame, result, stable_mode):
    """绘制全英文界面，展示置信度等信息以辅助排查问题"""
    h, w = frame.shape[:2]
    
    # 状态栏颜色分配：face(蓝色), fruit(橙色), unknown(灰色)
    stable_mode_lower = stable_mode.lower()
    if stable_mode_lower == "face":
        color = (255, 144, 30)   
    elif stable_mode_lower == "fruit":
        color = (0, 165, 255)    
    else:
        color = (128, 128, 128)  

    # 1. 绘制防抖后的最终结果
    cv2.putText(frame, f"Stable Mode: {stable_mode.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 2. 绘制当前帧的原始识别结果和置信度
    raw_mode = result["mode"].upper()
    conf = result["confidence"] * 100
    cv2.putText(frame, f"Raw: {raw_mode} ({conf:.1f}%)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # 3. 打印模型给出的各项概率分布 (对于排查准确率低非常有用)
    y_offset = 120
    cv2.putText(frame, "Probabilities:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_offset += 25
    for name, prob in result["probabilities"].items():
        text = f"  - {name}: {prob*100:.1f}%"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
    # 4. 绘制中心对准引导框 (假设模型训练时卡片在中间)
    cx, cy = w // 2, h // 2
    box_w, box_h = int(w * 0.45), int(h * 0.45)
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "Align Card Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# =========================================================================
# 运行模式控制
# =========================================================================
def run_camera(predictor):
    """ROS 相机实时调试模式"""
    try:
        rospy.init_node("text_selector_debug_node", anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    bridge = CvBridge()
    shared_data = {"frame": None, "new": False}
    frame_lock = threading.Lock()

    def image_callback(msg):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            with frame_lock:
                shared_data["frame"] = cv_img
                shared_data["new"] = True
        except Exception as e:
            rospy.logwarn(f"Image transfer error: {e}")

    topic_name = "/camera/color/image_raw"
    rospy.Subscriber(topic_name, RosImage, image_callback, queue_size=1)
    print(f"\n[DEBUG MODE] 已订阅 ROS 相机话题: {topic_name}")
    print("请在弹出的图像窗口中按 'q' 键退出测试。\n")

    state_machine = ModeStateMachine(stable_frames=3)

    while not rospy.is_shutdown():
        frame = None
        with frame_lock:
            if shared_data["new"] and shared_data["frame"] is not None:
                frame = shared_data["frame"].copy()
                shared_data["new"] = False

        if frame is None:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        # 预测当前帧
        result = predictor.predict_frame(frame)
        
        # 设置置信度阈值，低于该阈值认为没有有效识别
        if result["confidence"] > 0.60:
            stable_mode = state_machine.update(result["mode"])
        else:
            stable_mode = state_machine.update("UNKNOWN")

        # 绘制 UI 并显示
        display_frame = frame.copy()
        draw_debug_ui(display_frame, result, stable_mode)
        cv2.imshow("Text Selector - Camera Debug", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# 保留原有的快捷调用接口
def predict_text_mode(image_path: str | Path, model_path: str | Path | None = None) -> dict:
    predictor = TextPredictor(model_path)
    return predictor.predict_image_file(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict text-card mode: face or fruit.")
    # 将 required=True 去掉，使其变为可选参数
    parser.add_argument("--source", help="Text image path. 如果不填，默认开启 ROS 摄像头进行实时测试。")
    parser.add_argument("--model", help="Custom text selector model path.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    predictor = TextPredictor(args.model)

    if args.source:
        # 【兼容原有流程】执行静态图片预测并打印
        result = predictor.predict_image_file(args.source)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"mode={result['mode']} confidence={result['confidence']:.4f}")
    else:
        # 【新增排查流程】打开摄像头实时可视化
        run_camera(predictor)


if __name__ == "__main__":
    main()