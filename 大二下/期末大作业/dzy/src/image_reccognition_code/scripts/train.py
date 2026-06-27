import argparse
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "face": PROJECT_ROOT / "dataset_configs" / "face.yaml",
    "fruit": PROJECT_ROOT / "dataset_configs" / "fruit.yaml",
}


def build_absolute_data_config(mode: str) -> Path:
    data_root = PROJECT_ROOT / "prepared_datasets" / mode
    if not data_root.exists():
        raise FileNotFoundError(f"训练数据不存在，请先运行 prepare_dataset.py: {data_root}")

    names = {
        "face": ["dengziqi", "liuyifei", "renxianqi", "sabeining"],
        "fruit": ["apple", "banana", "grape", "orange"],
    }[mode]

    yaml_lines = [
        f"path: {data_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))

    config_path = Path(tempfile.gettempdir()) / f"{mode}_yolo_data_absolute.yaml"
    config_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练人脸或水果专用 YOLO 模型。")
    parser.add_argument("--mode", choices=["face", "fruit"], required=True)
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型，例如 yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0, help="Windows 建议使用 0，避免多进程重复加载 PyTorch DLL")
    parser.add_argument("--device", default=None, help="例如 cpu、0；页面文件或显存不足时可设为 cpu")
    args = parser.parse_args()

    data_config = build_absolute_data_config(args.mode)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=args.mode,
    )


if __name__ == "__main__":
    main()
