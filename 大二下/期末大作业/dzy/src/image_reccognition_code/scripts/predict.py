import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODELS = {
    "face": PROJECT_ROOT / "models" / "face_yolo.pt",
    "fruit": PROJECT_ROOT / "models" / "fruit_yolo.pt",
}

DESCRIPTIONS = {
    "dengziqi": "Detected dengziqi face.",
    "liuyifei": "Detected liuyifei face.",
    "renxianqi": "Detected renxianqi face.",
    "sabeining": "Detected sabeining face.",
    "apple": "Detected apple.",
    "banana": "Detected banana.",
    "grape": "Detected grape.",
    "orange": "Detected orange.",
}


def detect_position(box: list[float], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    horizontal = "left" if center_x < image_width / 3 else "right" if center_x > image_width * 2 / 3 else "center"
    vertical = "top" if center_y < image_height / 3 else "bottom" if center_y > image_height * 2 / 3 else "middle"
    return f"{vertical}-{horizontal}"


def result_to_dict(result, mode: str) -> dict:
    image_height, image_width = result.orig_shape
    objects = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())
        bbox = [round(value, 2) for value in box.xyxy[0].tolist()]
        width = round(bbox[2] - bbox[0], 2)
        height = round(bbox[3] - bbox[1], 2)
        position = detect_position(bbox, image_width, image_height)
        base_description = DESCRIPTIONS.get(class_name, f"Detected {class_name}.")
        objects.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": bbox,
            "bbox_size": {"width": width, "height": height},
            "position": position,
            "description": f"{base_description} Position: {position}.",
        })
    return {
        "mode": mode,
        "image": str(result.path),
        "image_size": {"width": image_width, "height": image_height},
        "objects": objects,
    }


def build_default_output_path(results, mode: str) -> Path:
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if results:
        save_dir = getattr(results[0], "save_dir", None)
        run_name = Path(save_dir).name if save_dir else mode
    else:
        run_name = mode
    return predictions_dir / f"predictions_{run_name}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run face or fruit YOLO inference.")
    parser.add_argument("--mode", choices=["face", "fruit"], required=True)
    parser.add_argument("--source", required=True, help="Image, folder, or video path")
    parser.add_argument("--model", help="Model path; default uses models/face_yolo.pt or models/fruit_yolo.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", help="Save visualized results")
    parser.add_argument("--output", help="JSON output path; default uses outputs/predictions/predictions_<run_name>.json")
    args = parser.parse_args()

    model_path = Path(args.model).resolve() if args.model else DEFAULT_MODELS[args.mode]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        project=str(PROJECT_ROOT / "outputs" / "visualized"),
        name=args.mode,
    )

    output = [result_to_dict(result, args.mode) for result in results]
    output_path = Path(args.output) if args.output else build_default_output_path(results, args.mode)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prediction JSON saved: {output_path}")


if __name__ == "__main__":
    main()
