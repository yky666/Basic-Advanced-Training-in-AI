# 人脸与水果图像识别项目

本项目使用两个 YOLO 分类/检测模型完成图像识别：

- 人脸模型：识别 `dengziqi`、`liuyifei`、`renxianqi`、`sabeining`。
- 水果模型：识别 `apple`、`banana`、`grape`、`orange`。

当前推荐统一入口为 `scripts/image_recognition_node.py`，它已经整合原来的离线推理 `scripts/predict.py` 和实时推理 `scripts/predict_real_time.py`。

## 1. 目录结构

```text
image_reccognition_code/
  dataset_configs/              # YOLO 数据集配置
  models/                       # 部署用权重
    face_yolo.pt
    fruit_yolo.pt
  outputs/
    predictions/                # JSON 推理结果
    visualized/                 # YOLO 可视化结果
    vl_crops/                   # 触发 DeepSeek-VL 时保存的 ROI 裁剪图
  prepared_datasets/            # 训练/验证划分后的数据集
  runs/                         # 训练历史输出
  scripts/
    prepare_dataset.py          # 数据集整理脚本
    train.py                    # 训练脚本
    image_recognition_node.py   # 统一识别节点，推荐使用
    predict.py                  # 旧离线推理脚本，保留兼容
    predict_real_time.py        # 旧实时推理脚本，保留兼容
    vl_describe.py              # DeepSeek-VL 图像描述封装
  水果_人脸数据集/               # 原始数据与真实场景图像
```

## 2. 权重管理

部署推理默认读取：

```text
models/face_yolo.pt
models/fruit_yolo.pt
```

建议：

- `runs/train/` 保留训练历史、指标和中间权重。
- `models/` 只放最终部署使用的权重。
- 重新训练后，把效果最好的权重复制到 `models/face_yolo.pt` 或 `models/fruit_yolo.pt`。

## 3. 数据集准备

数据来源会自动合并：

- `水果_人脸数据集/face`
- `水果_人脸数据集/fruits`
- `水果_人脸数据集/real_world_image`

真实场景图像已经作为增强数据加入训练/验证划分，适合提升摄像头实际使用时的泛化能力。

重新生成全部数据集：

```bash
python scripts/prepare_dataset.py --mode all
```

只更新单个任务：

```bash
python scripts/prepare_dataset.py --mode face
python scripts/prepare_dataset.py --mode fruit
```

注意：重新生成会覆盖 `prepared_datasets/face` 和 `prepared_datasets/fruit`。

## 4. 训练

常规训练：

```bash
python scripts/train.py --mode face --epochs 50
python scripts/train.py --mode fruit --epochs 50
```

Windows 显存/内存不足时推荐：

```bash
python scripts/train.py --mode face --epochs 50 --batch 2 --workers 0 --imgsz 416
python scripts/train.py --mode fruit --epochs 50 --batch 2 --workers 0 --imgsz 416
```

如果出现页面文件太小、CUDA DLL 加载失败等问题，优先降低 `batch`、设置 `workers=0`，并关闭其他占用内存的软件。

## 5. 统一识别节点

### 5.1 默认摄像头实时识别

不传 `--source` 时默认打开摄像头：

```bash
python scripts/image_recognition_node.py --mode face
python scripts/image_recognition_node.py --mode fruit
```

也可以不传 `--mode`，启动后键盘输入：

- `0`：人脸模型，默认值。
- `1`：水果模型。

实时窗口快捷键：

- `q` 或 `ESC`：退出。
- `0`：切换到人脸模型。
- `1`：切换到水果模型。

### 5.2 图片/文件夹/视频离线识别

传入 `--source` 后进入离线模式：

```bash
python scripts/image_recognition_node.py --mode face --source "path/to/image.jpg" --save
python scripts/image_recognition_node.py --mode fruit --source "path/to/folder" --save
python scripts/image_recognition_node.py --mode face --source "path/to/video.mp4"
```

输出 JSON 默认保存到：

```text
outputs/predictions/
```

可视化结果默认保存到：

```text
outputs/visualized/
```

也可以手动指定 JSON 输出位置：

```bash
python scripts/image_recognition_node.py --mode face --source "path/to/image.jpg" --output outputs/predictions/test_face.json
```

## 6. ROI 有效区域过滤

摄像头模式默认只认可画面中心区域内的检测结果，减少周围环境被误识别的情况。

默认参数：

```bash
--roi-width 0.45 --roi-height 0.80
```

示例：

```bash
python scripts/image_recognition_node.py --mode face --conf 0.60 --roi-width 0.45 --roi-height 0.80
```

如果需要显示 ROI 外被忽略的框：

```bash
python scripts/image_recognition_node.py --mode face --show-ignored
```

ROI 只能减少背景误检，不能根治人脸身份混淆。如果多人都被识别成 `renxianqi`，仍然需要继续清洗数据或重新训练人脸模型。

## 7. DeepSeek-VL 图像描述接口

统一节点预留了 DeepSeek-VL 调用接口。开启后，脚本会在检测稳定且置信度较高时裁剪目标 ROI，并调用 `scripts/vl_describe.py` 生成语义描述。

示例：

```bash
python scripts/image_recognition_node.py --mode fruit --enable-vl
```

常用参数：

```bash
--vl-model-path "D:\ai_models\deepseek-vl-1.3b-chat"
--vl-repo-path "..\voice_result\DeepSeek-VL"
--vl-conf 0.75
--vl-stable-frames 20
--vl-cooldown 8
--vl-crop-padding 16
```

当前策略：

- 只对 ROI 内的有效检测触发 VL。
- 只选择当前帧置信度最高的目标。
- 目标类别连续稳定若干帧后才调用，避免频繁请求。
- 裁剪图保存到 `outputs/vl_crops/`，方便回溯。

## 8. 人脸模型效果问题

如果人脸模型经常把不同人识别成同一类，常见原因包括：

- 训练图像背景差异过大，模型学到了背景而不是身份。
- 四类人脸样本数量、清晰度、角度、光照不均衡。
- 摄像头真实画面与原始训练图像分布差异较大。
- YOLO 直接做细粒度人脸身份分类并不是最稳方案。

建议优先级：

1. 增加真实摄像头环境采集的人脸样本。
2. 删除模糊、遮挡、多人同框、目标过小的图像。
3. 尽量让目标脸部或上半身位于画面中心。
4. 保持四个类别样本数量接近。
5. 后续可升级为“两阶段方案”：先做人脸检测，再做人脸身份分类或向量匹配。

## 9. 维护原则

- 不直接覆盖原始数据，所有训练数据从脚本生成。
- 训练历史放在 `runs/`，部署权重放在 `models/`。
- 推理结果 JSON 放在 `outputs/predictions/`。
- 可视化结果放在 `outputs/visualized/`。
- DeepSeek-VL 裁剪图放在 `outputs/vl_crops/`。
- 后续主入口建议统一使用 `scripts/image_recognition_node.py`。
