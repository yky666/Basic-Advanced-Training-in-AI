import argparse
import hashlib
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_dataset_root() -> Path:
    """Find the dataset folder without hardcoding its Chinese name."""
    for candidate in PROJECT_ROOT.iterdir():
        if not candidate.is_dir():
            continue
        has_original = (candidate / "face").exists() and (candidate / "fruits").exists()
        has_real_world = (candidate / "real_world_image").exists()
        if has_original or has_real_world:
            return candidate
    raise FileNotFoundError("Dataset root not found under project root.")


DATASET_ROOT = find_dataset_root()
REAL_WORLD_ROOT = DATASET_ROOT / "real_world_image"

MODES = {
    "face": {
        "classes": ["dengziqi", "liuyifei", "renxianqi", "sabeining"],
        "sources": [
            DATASET_ROOT / "face",
            REAL_WORLD_ROOT,
        ],
    },
    "fruit": {
        "classes": ["apple", "banana", "grape", "orange"],
        "sources": [
            DATASET_ROOT / "fruits",
            REAL_WORLD_ROOT,
        ],
    },
}


def iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_mode_images(mode: str) -> list[tuple[str, Path]]:
    config = MODES[mode]
    collected: list[tuple[str, Path]] = []
    for source_root in config["sources"]:
        if not source_root.exists():
            continue
        for class_name in config["classes"]:
            class_dir = source_root / class_name
            if class_dir.exists():
                for image_path in iter_images(class_dir):
                    collected.append((class_name, image_path))
    return collected


def prepare_mode(mode: str, val_ratio: float, seed: int) -> None:
    config = MODES[mode]
    output_root = PROJECT_ROOT / "prepared_datasets" / mode

    if output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    samples = collect_mode_images(mode)
    if not samples:
        raise FileNotFoundError(f"?????????: {config['sources']}")

    class_to_id = {class_name: class_id for class_id, class_name in enumerate(config["classes"])}
    per_class_images: dict[str, list[Path]] = {class_name: [] for class_name in config["classes"]}
    for class_name, image_path in samples:
        per_class_images[class_name].append(image_path)

    total_copied = 0
    total_train = 0
    total_val = 0

    for class_name, images in per_class_images.items():
        if not images:
            continue
        random.shuffle(images)
        val_count = max(1, int(len(images) * val_ratio)) if len(images) > 1 else len(images)
        class_id = class_to_id[class_name]

        for index, image_path in enumerate(images):
            split = "val" if index < val_count else "train"
            unique_id = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:10]
            target_stem = f"{class_name}_{unique_id}_{image_path.stem}"
            target_image = output_root / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = output_root / "labels" / split / f"{target_stem}.txt"

            shutil.copy2(image_path, target_image)
            target_label.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
            total_copied += 1
            if split == "train":
                total_train += 1
            else:
                total_val += 1

    print(f"??? {mode} ???: {output_root}")
    print(f"  train={total_train}, val={total_val}, total={total_copied}")


def main() -> None:
    parser = argparse.ArgumentParser(description="??????????????? YOLO ???????????????")
    parser.add_argument("--mode", choices=["face", "fruit", "all"], default="all")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="???????? 0.2")
    parser.add_argument("--seed", type=int, default=42, help="????????????")
    args = parser.parse_args()

    modes = ["face", "fruit"] if args.mode == "all" else [args.mode]
    for mode in modes:
        prepare_mode(mode, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
