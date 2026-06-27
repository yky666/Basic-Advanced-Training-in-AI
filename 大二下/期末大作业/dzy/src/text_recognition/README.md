# 文字指令识别模块

本模块用于识别拍摄到的文字指令图片，并输出后续图像识别节点应该调用的模型：

- `face`：调用人脸识别模型。
- `fruit`：调用水果识别模型。

当前数据目录：

```text
text_recognition/
  data/
    face/
    fruit/
  models/
    text_selector.joblib
  outputs/
    text_selector_report.json
  scripts/
    train_text_selector.py
    text_selector.py
```

## 1. 训练

```bash
cd text_recognition
python scripts/train_text_selector.py
```

训练完成后会生成：

```text
models/text_selector.joblib
outputs/text_selector_report.json
```

## 2. 单张图片测试

```bash
python scripts/text_selector.py --source "data/face/10.jpg"
python scripts/text_selector.py --source "data/fruit/15.jpg" --json
```

输出示例：

```text
mode=face confidence=0.9821
```

## 3. 接入图像识别节点

在 `image_reccognition_code/scripts/image_recognition_node.py` 中可以通过 `--text-source` 传入文字图片。

示例：

```bash
python ../image_reccognition_code/scripts/image_recognition_node.py --text-source "data/face/10.jpg"
```

脚本会先识别文字图片属于 `face` 还是 `fruit`，再自动选择对应的 YOLO 模型。

## 4. 后续替换 OCR 的接口

当前实现是轻量图片二分类器，不是完整文字 OCR。原因是当前任务只需要判断“接下来识别人脸还是水果”。

如果后续需要识别具体文字内容，可以保留同一个接口：

```python
predict_text_mode(image_path) -> {"mode": "face" | "fruit", "confidence": 0.0}
```

内部可替换为 PaddleOCR、EasyOCR 或其他 OCR/VLM 模型。
