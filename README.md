# VL Project 2 - Image Classification & Object Detection

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 数据集准备

将数据集放置于 `data/` 目录下：

```
data/
├── flowers102/      # 102 Category Flower Dataset (PyTorch自带)
└── road_vehicle/    # Road Vehicle Images Dataset
```

## 任务1: 卷积神经网络图像分类

### 训练

```bash
cd task1_classification
python train.py --model resnet50 --epochs 50 --batch_size 32 --lr 0.001
```

### 可用参数

- `--model`: 模型选择 (resnet18, resnet50, efficientnet_b0, vit_b_16)
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--data_dir`: 数据目录 (默认: ../data/flowers102)
- `--attention`: 启用注意力模块
- `--wandb`: 启用 wandb 日志

### 运行超参数搜索

```bash
bash run_experiments.sh
```

## 任务2: 目标检测与多目标跟踪

### 训练 YOLOv8

```bash
cd task2_detection
python train_yolo.py --data road_vehicle --epochs 50
```

### 视频推理与跟踪

```bash
python track_and_count.py --source video.mp4 --model yolov8n.pt --line_y 0.5
```

### 参数说明

- `--source`: 视频文件路径或摄像头ID
- `--model`: YOLOv8 模型 (yolov8n, yolov8s, yolov8m)
- `--line_y`: 越线检测 Y 轴位置 (0.0-1.0)
- `--save_occlusion`: 保存遮挡事件帧
- `--output_dir`: 输出目录

## 项目结构

```
PJ2/
├── README.md
├── requirements.txt
├── data/
│   ├── flowers102/
│   └── road_vehicle/
├── task1_classification/
│   ├── train.py
│   └── run_experiments.sh
└── task2_detection/
    ├── train_yolo.py
    ├── track_and_count.py
    └── run_detection.sh
```
