#!/usr/bin/env python3
"""
任务2: YOLOv8 目标检测模型微调
"""

import argparse
import os
import torch
from ultralytics import YOLO

def train_yolo(data_yaml, model='yolov8n.pt', epochs=50, batch=16, imgsz=640, device=None):
    """训练 YOLOv8 模型"""

    # 自动检测设备
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'

    print(f'Loading model: {model}')
    yolo_model = YOLO(model)

    print(f'Starting training on {data_yaml}')
    print(f'Device: {device}')

    # YOLOv8 会自动将日志、权重和可视化图表保存在 project/name 目录下
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project='./runs/detect',
        name='yolov8_finetune',
        exist_ok=True,
        optimizer='auto',  # 让 YOLOv8 自动选择优化器和学习率
        patience=20,  # 早停：验证集 20 轮无提升则停止
        cos_lr=True,  # 余弦退火学习率
        weight_decay=0.0005,
        cache='ram',  # 缓存数据到内存加速训练（小数据集推荐）
        workers=8,  # 数据加载线程数
        verbose=True,
        device=device,
    )

    print('Training completed!')
    print(f'Best model saved to: ./runs/detect/yolov8_finetune/weights/best.pt')

    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 目标检测训练')
    parser.add_argument('--data', type=str, default='road_vehicle', help='数据集名称')
    parser.add_argument('--data_yaml', type=str, default=None, help='数据集yaml文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default=None,
                        help='训练设备 (如 0, 0,1, cpu)，默认自动检测')

    args = parser.parse_args()

    if args.data_yaml is None:
        data_yaml = f'../data/{args.data}/dataset.yaml'
    else:
        data_yaml = args.data_yaml

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f'数据集配置文件未找到: {data_yaml}\n'
            f'请通过 --data_yaml 指定正确路径，或确认 --data 参数对应的目录存在。'
        )

    train_yolo(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

if __name__ == '__main__':
    main()