#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified image recognition node.

Default behavior:
- Subscribe to ROS camera topic and run real-time recognition.
- Use --mode to switch between face and fruit classifiers.
- Use --source to process an image / folder / video path instead of camera.
- Optional DeepSeek-VL description can be triggered for stable detections.
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image

# 导入 ROS 相关库
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = {
    "face": PROJECT_ROOT / "models" / "face_yolo.pt",
    "fruit": PROJECT_ROOT / "models" / "fruit_yolo.pt",
}
DEFAULT_CLASS_NAMES = {
    "face": ["dengziqi", "liuyifei", "renxianqi", "sabeining"],
    "fruit": ["apple", "banana", "grape", "orange"],
}

MODE_BY_INPUT = {"0": "face", "1": "fruit"}
MODE_LABELS = {"face": "face classifier", "fruit": "fruit classifier"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def choose_mode(default_mode: str = "face") -> str:
    user_input = input("Select mode: 0(face, default) / 1(fruit): ").strip()
    return MODE_BY_INPUT.get(user_input, default_mode)


def load_model(mode: str, model_path: str | None):
    selected_model = Path(model_path).resolve() if model_path else DEFAULT_MODELS[mode]
    if not selected_model.exists():
        raise FileNotFoundError(f"Model not found: {selected_model}")
    from ultralytics import YOLO
    print(f"Loading {MODE_LABELS[mode]}: {selected_model}")
    return YOLO(str(selected_model))


def build_roi(frame_width: int, frame_height: int, roi_width_ratio: float, roi_height_ratio: float) -> tuple[int, int, int, int]:
    roi_width = int(frame_width * roi_width_ratio)
    roi_height = int(frame_height * roi_height_ratio)
    x1 = max((frame_width - roi_width) // 2, 0)
    y1 = max((frame_height - roi_height) // 2, 0)
    x2 = min(x1 + roi_width, frame_width)
    y2 = min(y1 + roi_height, frame_height)
    return x1, y1, x2, y2


def is_in_roi(info: dict, roi: tuple[int, int, int, int]) -> bool:
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    center = info["center"]
    return roi_x1 <= center["x"] <= roi_x2 and roi_y1 <= center["y"] <= roi_y2


def build_detection_info(box, names: dict[int, str]) -> dict:
    class_id = int(box.cls.item())
    class_name = names[class_id]
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
    width = x2 - x1
    height = y2 - y1
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2],
        "bbox_size": {"width": width, "height": height},
        "center": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
    }


def clean_deepseek_text(text: str) -> str:
    replacements = {
        chr(0x0120): " ",
        chr(0x010A): "\n",
        chr(0x0109): "\t",
        chr(0x00C2): "",
    }
    for old, new_value in replacements.items():
        text = text.replace(old, new_value)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def crop_detection(frame, info: dict, padding: int = 12) -> Image.Image:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in info["bbox"]]
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, frame_width)
    y2 = min(y2 + padding, frame_height)
    crop_bgr = frame[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def draw_roi(frame, roi: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
    cv2.putText(frame, "valid ROI", (x1 + 8, max(y1 + 24, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2)


def draw_detection(frame, info: dict, valid: bool) -> None:
    x1, y1, x2, y2 = [int(value) for value in info["bbox"]]
    color = (0, 220, 0) if valid else (80, 80, 80)
    label = f"{info['class_name']} {info['confidence']:.2f}"
    if not valid:
        label = f"ignored {label}"
    size_text = f"box=({info['bbox_size']['width']:.0f}x{info['bbox_size']['height']:.0f}) pos=({x1},{y1})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 28, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, size_text, (x1, max(y1 - 6, 42)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def print_detections(frame_index: int, detections: list[dict], ignored_count: int) -> None:
    if not detections:
        print(f"[frame {frame_index}] no valid ROI detections, ignored_outside_roi={ignored_count}")
        return
    for index, info in enumerate(detections, start=1):
        x1, y1, x2, y2 = info["bbox"]
        width = info["bbox_size"]["width"]
        height = info["bbox_size"]["height"]
        center = info["center"]
        print(
            f"[frame {frame_index}] #{index} class={info['class_name']}({info['class_id']}) "
            f"conf={info['confidence']:.4f} bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
            f"size=({width:.1f}x{height:.1f}) center=({center['x']:.1f}, {center['y']:.1f}) "
            f"ignored_outside_roi={ignored_count}"
        )


def save_prediction_json(output_path: Path, mode: str, source: str, detections: list[dict], frame_size: tuple[int, int], extra: dict | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "source": source,
        "frame_size": {"width": frame_size[0], "height": frame_size[1]},
        "objects": detections,
    }
    if extra:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prediction JSON saved: {output_path}")


def build_default_output_path(mode: str, source: str) -> Path:
    output_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if source == "camera":
        return output_dir / f"predictions_{mode}_camera.json"
    stem = Path(source).stem or mode
    return output_dir / f"predictions_{mode}_{stem}.json"


def maybe_describe_with_vl(args, frame, detections: list[dict], stable_state: dict, vl_state: dict) -> str | None:
    if not args.enable_vl or not detections:
        return None

    best = max(detections, key=lambda item: item["confidence"])
    if best["confidence"] < args.vl_conf:
        stable_state["class_name"] = None
        stable_state["count"] = 0
        return None

    if stable_state.get("class_name") == best["class_name"]:
        stable_state["count"] += 1
    else:
        stable_state["class_name"] = best["class_name"]
        stable_state["count"] = 1

    now = time.time()
    if stable_state["count"] < args.vl_stable_frames:
        return None
    if now - vl_state.get("last_call_time", 0.0) < args.vl_cooldown:
        return None

    crop = crop_detection(frame, best, padding=args.vl_crop_padding)
    crop_dir = PROJECT_ROOT / "outputs" / "vl_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{int(now)}_{best['class_name']}_{best['confidence']:.2f}.jpg"
    crop.save(crop_path)

    if vl_state.get("describer") is None:
        print("Loading DeepSeek-VL describer. This may take a while...")
        from vl_describe import DeepSeekVLDescriber
        vl_state["describer"] = DeepSeekVLDescriber(
            model_path=args.vl_model_path,
            repo_path=args.vl_repo_path,
            max_new_tokens=args.vl_max_tokens,
        )

    prompt = (
        f"YOLO detected class '{best['class_name']}' with confidence {best['confidence']:.2f}. "
        "Describe the cropped image briefly and verify whether the object matches the detected class."
    )
    try:
        description = vl_state["describer"].describe_image(crop, prompt=prompt)
    except Exception as exc:
        description = f"DeepSeek-VL description failed: {exc}"

    description = clean_deepseek_text(description)
    vl_state["last_call_time"] = now
    vl_state["last_description"] = description
    vl_state["last_crop_path"] = str(crop_path)
    print(f"[DeepSeek-VL] crop={crop_path}")
    print(f"[DeepSeek-VL] {description}")
    return description


# =========================================================================
# 核心修改：将 OpenCV 摄像头读取替换为 ROS 话题订阅
# =========================================================================
def run_camera(args, mode: str, model) -> None:
    # 确保 ROS 节点已初始化
    try:
        rospy.init_node("image_recognition_node", anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    bridge = CvBridge()
    shared_data = {"frame": None, "new": False}
    frame_lock = threading.Lock()

    # ROS 图像回调函数
    def image_callback(msg):
        try:
            # 奥比中光相机发布的话题格式一般为 bgr8 或 rgb8，转为 OpenCV 使用的 bgr8
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            with frame_lock:
                shared_data["frame"] = cv_img
                shared_data["new"] = True
        except Exception as e:
            rospy.logwarn(f"Image transfer error: {e}")

    # 订阅相机 RGB 话题
    topic_name = "/camera/color/image_raw"
    rospy.Subscriber(topic_name, RosImage, image_callback, queue_size=1)
    print(f"Subscribed to ROS topic: {topic_name}. Waiting for images...")

    window_name = f"Image Recognition Node - {mode}"
    frame_index = 0
    fps_frame_count = 0
    last_fps_time = time.time()
    fps = 0.0
    stable_state = {"class_name": None, "count": 0}
    vl_state = {"describer": None, "last_call_time": 0.0, "last_description": ""}

    print("Started. Press q/ESC to quit; press 0 for face, 1 for fruit.")
    print("Only detections whose center is inside the ROI are treated as valid.")
    if args.enable_vl:
        print("DeepSeek-VL is enabled. Descriptions are triggered only for stable high-confidence detections.")

    # 替换原本的 while True，使用 ROS 的关闭信号控制
    while not rospy.is_shutdown():
        frame = None
        # 安全地从子线程获取最新一帧图像
        with frame_lock:
            if shared_data["new"] and shared_data["frame"] is not None:
                frame = shared_data["frame"].copy()
                shared_data["new"] = False

        # 如果没有获取到新画面，暂时休眠并处理 OpenCV 键盘事件，防止界面卡死
        if frame is None:
            time.sleep(0.01)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        frame_index += 1
        fps_frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        roi = build_roi(frame_width, frame_height, args.roi_width, args.roi_height)
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        all_detections = [build_detection_info(box, results[0].names) for box in results[0].boxes]
        valid_detections = [info for info in all_detections if is_in_roi(info, roi)]
        ignored_detections = [info for info in all_detections if not is_in_roi(info, roi)]

        if frame_index % max(args.print_every, 1) == 0:
            print_detections(frame_index, valid_detections, len(ignored_detections))

        maybe_describe_with_vl(args, frame, valid_detections, stable_state, vl_state)

        draw_roi(frame, roi)
        if args.show_ignored:
            for info in ignored_detections:
                draw_detection(frame, info, valid=False)
        for info in valid_detections:
            draw_detection(frame, info, valid=True)

        if vl_state.get("last_description"):
            text = vl_state["last_description"][:90]
            cv2.putText(frame, f"VL: {text}", (10, frame_height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)

        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            last_fps_time = now

        cv2.putText(frame, f"mode={mode} fps={fps:.1f} conf={args.conf:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 120, 0), 2)
        cv2.putText(frame, "q/ESC: quit | 0: face | 1: fruit | ROI=center band", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("0"), ord("1")):
            new_mode = MODE_BY_INPUT[chr(key)]
            if new_mode != mode:
                mode = new_mode
                model = load_model(mode, None)
                cv2.setWindowTitle(window_name, f"Image Recognition Node - {mode}")
                stable_state = {"class_name": None, "count": 0}
                print(f"Switched to {MODE_LABELS[mode]}")

    cv2.destroyAllWindows()


def run_path(args, mode: str, model, source: Path) -> None:
    if source.is_dir():
        files = sorted([p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if not files:
            raise FileNotFoundError(f"No supported images in folder: {source}")
        output_dir = PROJECT_ROOT / "outputs" / "visualized"
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        for image_path in files:
            results = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, save=args.save, project=str(output_dir), name=mode, verbose=False)
            for result in results:
                detections = [build_detection_info(box, result.names) for box in result.boxes]
                image_height, image_width = result.orig_shape
                all_results.append({
                    "mode": mode,
                    "image": str(image_path),
                    "image_size": {"width": image_width, "height": image_height},
                    "objects": detections,
                })
        output_path = Path(args.output) if args.output else build_default_output_path(mode, source.name)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prediction JSON saved: {output_path}")
        return

    if source.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        output_dir = PROJECT_ROOT / "outputs" / "visualized"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        all_results = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            for result in results:
                detections = [build_detection_info(box, result.names) for box in result.boxes]
                image_height, image_width = result.orig_shape
                all_results.append({
                    "frame_index": frame_index,
                    "mode": mode,
                    "video": str(source),
                    "image_size": {"width": image_width, "height": image_height},
                    "objects": detections,
                })
        cap.release()
        output_path = Path(args.output) if args.output else build_default_output_path(mode, source.stem)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prediction JSON saved: {output_path}")
        return

    results = model.predict(str(source), conf=args.conf, imgsz=args.imgsz, save=args.save, project=str(PROJECT_ROOT / "outputs" / "visualized"), name=mode, verbose=False)
    all_results = []
    for result in results:
        detections = [build_detection_info(box, result.names) for box in result.boxes]
        image_height, image_width = result.orig_shape
        all_results.append({
            "mode": mode,
            "image": str(source),
            "image_size": {"width": image_width, "height": image_height},
            "objects": detections,
        })
    output_path = Path(args.output) if args.output else build_default_output_path(mode, source.stem)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prediction JSON saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified image recognition node for camera, image, folder, and video inference.")
    parser.add_argument("--mode", choices=["face", "fruit"], help="Set model directly; if omitted, keyboard input is used")
    parser.add_argument("--source", help="Image / folder / video path. If omitted, camera mode is used.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, default 0 (Ignored when using ROS topics)")
    parser.add_argument("--model", help="Custom model path")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", help="Save visualized results")
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument("--print-every", type=int, default=15, help="Print details every N frames")
    parser.add_argument("--roi-width", type=float, default=0.45, help="Valid ROI width ratio, centered")
    parser.add_argument("--roi-height", type=float, default=0.80, help="Valid ROI height ratio, centered")
    parser.add_argument("--show-ignored", action="store_true", help="Draw detections outside ROI in gray")
    parser.add_argument("--enable-vl", action="store_true", help="Enable DeepSeek-VL description for stable high-confidence detections")
    parser.add_argument("--vl-model-path", default=r"D:\ai_models\deepseek-vl-1.3b-chat", help="DeepSeek-VL local model path")
    parser.add_argument("--vl-repo-path", default=str(PROJECT_ROOT.parent / "voice_result" / "DeepSeek-VL"), help="DeepSeek-VL repo path")
    parser.add_argument("--vl-conf", type=float, default=0.75, help="Minimum confidence required before calling DeepSeek-VL")
    parser.add_argument("--vl-stable-frames", type=int, default=20, help="Consecutive stable frames required before calling DeepSeek-VL")
    parser.add_argument("--vl-cooldown", type=float, default=8.0, help="Seconds between DeepSeek-VL calls")
    parser.add_argument("--vl-crop-padding", type=int, default=16, help="Pixels padded around YOLO box before VL crop")
    parser.add_argument("--vl-max-tokens", type=int, default=96, help="Maximum new tokens for DeepSeek-VL description")
    args = parser.parse_args()

    mode = args.mode or choose_mode(default_mode="face")
    model = load_model(mode, args.model)

    if args.source:
        run_path(args, mode, model, Path(args.source).expanduser().resolve())
    else:
        run_camera(args, mode, model)


if __name__ == "__main__":
    main()