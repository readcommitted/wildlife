"""
yolo_detector.py — Wildlife Object Detection with YOLOv8
--------------------------------------------------------

Provides animal detection and smart cropping using a YOLOv8 model.

Features:
✅ Loads YOLOv8 model (default: `yolov8n.pt`)
✅ Runs detection on PIL images
✅ Filters results by confidence, area, and aspect ratio
✅ Returns high-quality crops or falls back to full image

Used during RAW image ingestion to improve species identification
by focusing on detected animals before CLIP embedding.

Dependencies:
- ultralytics YOLO (v8)
- PIL (Pillow)
- torch

Author: Matt Scardino
Project: Wildlife Vision System
"""

from ultralytics import YOLO
from PIL import Image
import torch


class YOLODetector:
    """
    Wrapper for YOLOv8 animal detection and image cropping.

    Args:
        model_path (str): Path to YOLOv8 model weights (.pt file)
        device (str): 'cuda', 'cpu', or None (auto-detect)
    """
    def __init__(self, model_path="yolov8n.pt", device=None):
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect_and_crop(self, pil_image: Image.Image, conf_threshold=0.3):
        """
        Detects animals in an image and returns valid crops.

        Args:
            pil_image (PIL.Image): Input image for detection
            conf_threshold (float): Minimum confidence threshold (default 0.3)

        Returns:
            list[PIL.Image]: List containing cropped image(s); falls back to full image if no valid detections
        """
        results = self.model(pil_image)
        detections = results[0]

        # Filtering parameters
        min_crop_area = 5000         # Discard small crops
        max_aspect_ratio = 4.0       # Discard extremely wide/tall crops

        for box in detections.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            aspect_ratio = max(w / h, h / w)

            if area < min_crop_area or aspect_ratio > max_aspect_ratio:
                continue

            # Return the first valid crop
            crop = pil_image.crop((x1, y1, x2, y2))
            return [crop]

        # Fallback: return full image if no valid crops
        return [pil_image]
