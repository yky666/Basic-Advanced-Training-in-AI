#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a lightweight text-image selector.

The current competition requirement only needs the photographed text card to
select which downstream recognizer should run:

- face  -> call the face classifier
- fruit -> call the fruit classifier

This is intentionally a small image classifier instead of a full OCR pipeline.
It can be replaced later by PaddleOCR / EasyOCR without changing the interface:
`predict_text_mode(image_path) -> "face" | "fruit"`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "text_selector.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "text_selector_report.json"
CLASSES = ["face", "fruit"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_features(image_path: Path, size: int = 96) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    gray = gray.astype(np.float32) / 255.0

    # Add coarse projection features. These are useful for text-card layout.
    row_projection = gray.mean(axis=1)
    col_projection = gray.mean(axis=0)
    return np.concatenate([gray.reshape(-1), row_projection, col_projection])


def collect_dataset() -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []
    for class_name in CLASSES:
        class_dir = DATA_ROOT / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)
                labels.append(class_name)
    if len(set(labels)) < 2:
        raise RuntimeError("Need both face and fruit samples to train the selector.")
    return image_paths, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-card selector: face vs fruit.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=2000, help="LogisticRegression max_iter.")
    args = parser.parse_args()

    random.seed(args.seed)
    image_paths, labels = collect_dataset()
    features = np.stack([load_image_features(path) for path in image_paths])

    train_x, val_x, train_y, val_y, train_paths, val_paths = train_test_split(
        features,
        labels,
        image_paths,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=args.max_iter, class_weight="balanced")),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES, "feature_size": 96}, MODEL_PATH)

    report = {
        "model_path": str(MODEL_PATH),
        "data_root": str(DATA_ROOT),
        "total_count": len(labels),
        "train_count": len(train_y),
        "val_count": len(val_y),
        "classes": CLASSES,
        "classification_report": classification_report(val_y, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(val_y, predictions, labels=CLASSES).tolist(),
        "val_samples": [
            {"image": str(path), "label": label, "prediction": pred}
            for path, label, pred in zip(val_paths, val_y, predictions)
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(classification_report(val_y, predictions, zero_division=0))


if __name__ == "__main__":
    main()
