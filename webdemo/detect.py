from ultralytics import YOLO
import numpy as np

# 加载YOLOv8模型（可改为你下载的权重）
yolo_model = YOLO('yolov8n.pt')  # yolov8n.pt较小更适合demo用

def detect_chairs(image):
    """
    Detect chairs in the input image using YOLOv8.

    Returns:
        list of bounding boxes (x, y, w, h)
        list of center points [(x, y), ...]
    """
    results = yolo_model(image, verbose=False)[0]
    boxes = []
    centers = []

    for r in results.boxes:
        cls = int(r.cls.item())
        name = yolo_model.names[cls]
        if name == 'chair':
            x1, y1, x2, y2 = r.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            boxes.append((int(x1), int(y1), int(w), int(h)))
            centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

    return boxes, centers
