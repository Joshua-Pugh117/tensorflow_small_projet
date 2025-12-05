# yolo_func.py
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os

# Load model once when the module is imported
print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")  # auto-downloads on first run (~6 MB)
print("YOLOv8n loaded successfully!")

# Load COCO class names
with open("config.json", "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)["class_names"]

def detect_yolo(image_pil: Image.Image, confidence_threshold: float = 0.5):
    """
    Runs YOLOv8n on a PIL image.
    Returns:
        boxes (np.ndarray)   – shape (N,4) normalized [ymin, xmin, ymax, xmax]
        scores (np.ndarray)  – shape (N,)
        classes (np.ndarray) – shape (N,) integer class IDs (0-based for YOLO)
        annotated_image (PIL.Image)
    """
    # Run inference
    results = model(image_pil, conf=confidence_threshold, verbose=False)[0]

    # Handle empty detections
    if results.boxes is None or len(results.boxes) == 0:
        empty_boxes = np.empty((0, 4))
        empty_scores = np.empty((0,))
        empty_classes = np.empty((0,), dtype=int)
        return empty_boxes, empty_scores, empty_classes, image_pil.copy()

    # Extract raw results
    boxes_px = results.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2] in pixels
    scores   = results.boxes.conf.cpu().numpy()
    classes  = results.boxes.cls.cpu().numpy().astype(int)

    h, w = image_pil.size[1], image_pil.size[0]  # height, width

    # Convert to normalized [ymin, xmin, ymax, xmax] (same format as your Faster R-CNN)
    boxes_norm = np.zeros_like(boxes_px)
    boxes_norm[:, 0] = boxes_px[:, 1] / h   # ymin
    boxes_norm[:, 1] = boxes_px[:, 0] / w   # xmin
    boxes_norm[:, 2] = boxes_px[:, 3] / h   # ymax
    boxes_norm[:, 3] = boxes_px[:, 2] / w   # xmax

    # Draw on image (lime green boxes)
    draw = ImageDraw.Draw(image_pil)

    for (ymin, xmin, ymax, xmax), score, cls_id in zip(boxes_norm, scores, classes):
        # Convert to pixel coordinates
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
        label = f"{class_name} {score:.2%}"

        # Box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)

        # Label background + text
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = ImageFont.load_default()

        # Fallback for older Pillow versions
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = font.getsize(label)

        draw.rectangle([x1, y1 - text_h - 10, x1 + text_w + 12, y1], fill="lime")
        draw.text((x1 + 6, y1 - text_h - 8), label, fill="black", font=font)

    return boxes_norm, scores, classes, image_pil