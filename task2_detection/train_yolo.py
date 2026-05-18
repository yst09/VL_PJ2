#!/usr/bin/env python3
"""
任务2: YOLOv8 目标检测模型微调
"""

import argparse
import os
from ultralytics import YOLO

def train_yolo(data_yaml, model='yolov8n.pt', epochs=50, batch=16, imgsz=640):
    """训练 YOLOv8 模型"""

    print(f'Loading model: {model}')
    yolo_model = YOLO(model)

    print(f'Starting training on {data_yaml}')
    # YOLOv8 会自动将日志、权重和可视化图表保存在 project/name 目录下
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project='./runs/detect',
        name='yolov8_finetune',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        verbose=True,
        device=0, # 指定使用 GPU 0
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

    args = parser.parse_args()

    if args.data_yaml is None:
        data_yaml = f'../data/{args.data}/dataset.yaml'
    else:
        data_yaml = args.data_yaml

    if not os.path.exists(data_yaml):
        print(f'Warning: {data_yaml} not found. Using COCO128 for demo.')
        data_yaml = 'coco128.yaml'

    train_yolo(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )

if __name__ == '__main__':
    main()