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
        self.frame_counter = 0
        self.cleanup_interval = 300  # 每300帧清理一次未活跃的ID

    def update(self, xyxy, ids, frame_width):
        """更新计数逻辑 (基于 X 轴坐标)"""
        line_x = int(frame_width * self.line_x_ratio)
        self.frame_counter += 1
        current_ids = set()

        for box, track_id in zip(xyxy, ids):
            current_ids.add(track_id)
            current_x = (box[0] + box[2]) / 2

            if track_id in self.tracked_objects:
                obj = self.tracked_objects[track_id]
                prev_x = obj['last_x']

                # 已计数标志：避免同一 ID 在线附近抖动反复计数
                if not obj.get('counted_right', False) and prev_x < line_x and current_x >= line_x:
                    self.count_right += 1
                    obj['counted_right'] = True
                    obj['counted_left'] = False
                    print(f'>> Object {track_id} crossed line to RIGHT. Total Right: {self.count_right}')

                elif not obj.get('counted_left', False) and prev_x > line_x and current_x <= line_x:
                    self.count_left += 1
                    obj['counted_left'] = True
                    obj['counted_right'] = False
                    print(f'<< Object {track_id} crossed line to LEFT. Total Left: {self.count_left}')

                obj['last_x'] = current_x
                obj['last_seen'] = self.frame_counter
            else:
                self.tracked_objects[track_id] = {
                    'last_x': current_x,
                    'last_seen': self.frame_counter,
                    'counted_left': False,
                    'counted_right': False,
                }

        # 定期清理长时间未出现的 ID，防止字典无限增长
        if self.frame_counter % self.cleanup_interval == 0:
            stale = [tid for tid, obj in self.tracked_objects.items()
                     if self.frame_counter - obj['last_seen'] > self.cleanup_interval]
            for tid in stale:
                del self.tracked_objects[tid]

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

def process_video(source, model_path, line_x=0.5, save_occlusion=False, output_dir='./output',
                  classes=None, conf=0.4, iou=0.5, imgsz=640, tracker='botsort.yaml'):
    os.makedirs(output_dir, exist_ok=True)
    print(f'Loading model from {model_path}')
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Error: Cannot open video {source}')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
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

        # 保存原始帧用于遮挡帧保存（避免保存到带标注的图）
        raw_frame = frame.copy() if save_occlusion else None

        # 显式传入 imgsz / conf / iou / classes，避免推理与训练设置不一致或检出无关类别
        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=classes,
            verbose=False,
        )[0]
        boxes = results.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy()

            for box, track_id, cls, cf in zip(xyxy, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                label = f'ID:{track_id} {results.names[cls]} {cf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            counter.update(xyxy, ids, width)

            if save_occlusion:
                occlusions = detect_occlusion(xyxy, ids)
                if occlusions:
                    occlusion_frame_path = os.path.join(occlusion_dir, f'frame_{frame_idx:06d}.jpg')
                    cv2.imwrite(occlusion_frame_path, raw_frame)
                    print(f'[*] Occlusion detected at frame {frame_idx} between IDs: {occlusions}')

        line_x_px = int(width * line_x)
        cv2.line(frame, (line_x_px, 0), (line_x_px, height), (0, 0, 255), 3)

        counts = counter.get_counts()
        info_text = f'Moving Left: {counts["left"]} | Moving Right: {counts["right"]}'

        cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    final_counts = counter.get_counts()
    print(f'\nFinal counts: Left={final_counts["left"]}, Right={final_counts["right"]}')
    print(f'Output video saved to {output_dir}/output_tracked.mp4')

def main():
    parser = argparse.ArgumentParser(description='多目标跟踪与左右越线计数')
    parser.add_argument('--source', type=str, required=True, help='视频文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 模型权重')
    parser.add_argument('--line_x', type=float, default=0.5, help='越线X轴位置比例 (0.0-1.0)')
    parser.add_argument('--save_occlusion', action='store_true', help='保存发生遮挡的帧')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='只跟踪指定类别 ID (COCO 车辆: 2 3 5 7 = car motorcycle bus truck)')
    parser.add_argument('--conf', type=float, default=0.4, help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU 阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='推理图像尺寸 (应与训练一致)')
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                        help='跟踪器配置 (botsort.yaml / bytetrack.yaml)')
    args = parser.parse_args()

    process_video(
        source=args.source,
        model_path=args.model,
        line_x=args.line_x,
        save_occlusion=args.save_occlusion,
        output_dir=args.output_dir,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
    )

if __name__ == '__main__':
    main()