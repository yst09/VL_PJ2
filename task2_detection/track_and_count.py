#!/usr/bin/env python3
"""
任务2: 视频多目标跟踪、左右越线计数与遮挡帧保存
使用 YOLOv8 内置 BoT-SORT 算法以提高追踪稳定性
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

class LineCounter:
    """左右越线计数器"""
    def __init__(self, line_x_ratio=0.5):
        self.line_x_ratio = line_x_ratio
        self.count_left = 0
        self.count_right = 0
        self.tracked_objects = {}

    def update(self, xyxy, ids, frame_width):
        """更新计数逻辑 (基于 X 轴坐标)"""
        line_x = int(frame_width * self.line_x_ratio)

        for box, track_id in zip(xyxy, ids):
            # 计算检测框中心点的 X 坐标
            current_x = (box[0] + box[2]) / 2

            if track_id in self.tracked_objects:
                prev_x = self.tracked_objects[track_id]['last_x']

                # 从左往右跨越 (上一帧在左，这一帧在右)
                if prev_x < line_x and current_x >= line_x:
                    self.count_right += 1
                    print(f'>> Object {track_id} crossed line to RIGHT. Total Right: {self.count_right}')

                # 从右往左跨越 (上一帧在右，这一帧在左)
                elif prev_x > line_x and current_x <= line_x:
                    self.count_left += 1
                    print(f'<< Object {track_id} crossed line to LEFT. Total Left: {self.count_left}')

            # 更新物体的历史位置
            self.tracked_objects[track_id] = {'last_x': current_x}

    def get_counts(self):
        return {'left': self.count_left, 'right': self.count_right}

def detect_occlusion(xyxy, ids, occlusion_threshold=0.6):
    """检测当前帧内的物体遮挡事件"""
    occlusion_events = []
    num_boxes = len(xyxy)
    
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            box1, box2 = xyxy[i], xyxy[j]
            id1, id2 = ids[i], ids[j]

            # 计算交集区域
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 > x1 and y2 > y1:
                inter_area = (x2 - x1) * (y2 - y1)
                # 计算两个框中较小的那个框的面积
                min_area = min(
                    (box1[2] - box1[0]) * (box1[3] - box1[1]),
                    (box2[2] - box2[0]) * (box2[3] - box2[1])
                )
                
                # 如果重叠面积占比超过阈值，视为发生遮挡
                if min_area > 0 and (inter_area / min_area) > occlusion_threshold:
                    occlusion_events.append((id1, id2))
                    
    return occlusion_events

def process_video(source, model_path, line_x=0.5, save_occlusion=False, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    print(f'Loading model from {model_path}')
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Error: Cannot open video {source}')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, 'output_tracked.mp4'), fourcc, fps, (width, height))

    counter = LineCounter(line_x_ratio=line_x)
    occlusion_dir = os.path.join(output_dir, 'occlusion_frames')
    if save_occlusion:
        os.makedirs(occlusion_dir, exist_ok=True)

    print(f'Processing video: {width}x{height} @ {fps}fps, {total_frames} frames')
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'Processing frame {frame_idx}/{total_frames}')

        # 核心改动：直接调用 YOLO 的 .track() 并使用 BoT-SORT
        # persist=True 告诉模型保留上一帧的轨迹历史
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        boxes = results.boxes

        # 确保当前帧检测到了物体，并且成功分配了 tracking ID
        if boxes is not None and boxes.id is not None:
            # 强制转换为 int 类型，彻底消灭小数 ID
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy()

            # 绘制检测框与整数 ID
            for box, track_id, cls, conf in zip(xyxy, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                label = f'ID:{track_id} {results.names[cls]} {conf:.2f}'
                
                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 写标签
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 更新左右越线计数器
            counter.update(xyxy, ids, width)

            # 检测遮挡并保存
            if save_occlusion:
                occlusions = detect_occlusion(xyxy, ids)
                if occlusions:
                    occlusion_frame_path = os.path.join(occlusion_dir, f'frame_{frame_idx:06d}.jpg')
                    cv2.imwrite(occlusion_frame_path, frame)
                    print(f'[*] Occlusion detected at frame {frame_idx} between IDs: {occlusions}')

        # 绘制垂直参考线
        line_x_px = int(width * line_x)
        cv2.line(frame, (line_x_px, 0), (line_x_px, height), (0, 0, 255), 3)

        # 绘制统计面板
        counts = counter.get_counts()
        info_text = f'Moving Left: {counts["left"]} | Moving Right: {counts["right"]}'
        
        # 增加一个黑底背景让文字更清晰
        cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f'\nFinal counts: Left={counts["left"]}, Right={counts["right"]}')
    print(f'Output video saved to {output_dir}/output_tracked.mp4')

def main():
    parser = argparse.ArgumentParser(description='多目标跟踪与左右越线计数')
    parser.add_argument('--source', type=str, required=True, help='视频文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 模型权重')
    parser.add_argument('--line_x', type=float, default=0.5, help='越线X轴位置比例 (0.0-1.0)')
    parser.add_argument('--save_occlusion', action='store_true', help='保存发生遮挡的帧')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    args = parser.parse_args()

    process_video(
        source=args.source,
        model_path=args.model,
        line_x=args.line_x,
        save_occlusion=args.save_occlusion,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()