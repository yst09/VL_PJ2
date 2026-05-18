#!/bin/bash
# 任务2: 目标检测与跟踪执行脚本

cd "$(dirname "$0")"

# 如果你之前没有准备好 dataset.yaml，YOLO 会默认使用 COCO128
echo "=== Step 1: Training YOLOv8 ==="
python train_yolo.py \
    --data road_vehicle \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640

echo ""
echo "=== Step 2: Running inference with tracking ==="
# 指向服务器上的目标视频路径
VIDEO_SOURCE="/home/sitaoyang/VL/PJ2/data/car.mp4"

if [ -f "$VIDEO_SOURCE" ]; then
    echo "Found video at $VIDEO_SOURCE, starting processing..."
    python track_and_count.py \
        --source "$VIDEO_SOURCE" \
        --model runs/detect/yolov8_finetune/weights/best.pt \
        --line_y 0.5 \
        --save_occlusion \
        --output_dir ./output
else
    echo "Error: Video not found at $VIDEO_SOURCE. Please check the path."
    exit 1
fi

echo ""
echo "Task 2 completed! Check the ./output directory for results."