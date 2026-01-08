"""
Player and Ball Detection Module

Uses YOLOv8 for object detection with optional Roboflow cloud inference.
Detects players, ball, and referees in soccer footage.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import cv2
from pathlib import Path

# PyTorch and Ultralytics
import torch
from ultralytics import YOLO

from loguru import logger


@dataclass
class Detection:
    """A single detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Get bottom center point (foot position)."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3]
        )
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height


class PlayerBallDetector:
    """
    Detects players and ball using YOLOv8.
    
    Supports multiple model sizes and devices (CUDA, MPS, CPU).
    """
    
    # COCO class IDs
    PERSON_CLASS_ID = 0
    SPORTS_BALL_CLASS_ID = 32
    
    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        device: str = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device for inference (cuda/mps/cpu/None for auto)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Initializing detector on device: {device}")
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)
        
        logger.info(f"Loaded model: {model_path}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players and ball in frame.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID, self.SPORTS_BALL_CLASS_ID],
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Map class ID to name
                if class_id == self.PERSON_CLASS_ID:
                    class_name = "player"
                elif class_id == self.SPORTS_BALL_CLASS_ID:
                    class_name = "ball"
                else:
                    class_name = "unknown"
                
                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect in multiple frames (batch processing).
        
        Args:
            frames: List of BGR images
        
        Returns:
            List of detection lists for each frame
        """
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID, self.SPORTS_BALL_CLASS_ID],
            verbose=False
        )
        
        all_detections = []
        
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                if class_id == self.PERSON_CLASS_ID:
                    class_name = "player"
                elif class_id == self.SPORTS_BALL_CLASS_ID:
                    class_name = "ball"
                else:
                    class_name = "unknown"
                
                frame_detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name
                ))
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def get_players(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to only players."""
        return [d for d in detections if d.class_name == "player"]
    
    def get_ball(self, detections: List[Detection]) -> Optional[Detection]:
        """Get ball detection (highest confidence if multiple)."""
        balls = [d for d in detections if d.class_name == "ball"]
        if balls:
            return max(balls, key=lambda x: x.confidence)
        return None


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    colors: Dict[str, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw detection boxes on frame.
    
    Args:
        frame: BGR image
        detections: List of detections to draw
        colors: Optional color map for classes
    
    Returns:
        Annotated frame
    """
    if colors is None:
        colors = {
            "player": (0, 255, 0),
            "ball": (0, 165, 255),
            "referee": (255, 255, 0),
            "unknown": (128, 128, 128)
        }
    
    output = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        color = colors.get(det.class_name, (128, 128, 128))
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(
            output,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        cv2.putText(
            output,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return output


if __name__ == "__main__":
    # Test detection
    detector = PlayerBallDetector(device="cpu")
    
    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame[:] = (34, 139, 34)  # Green field
    
    detections = detector.detect(test_frame)
    print(f"Detected {len(detections)} objects")
