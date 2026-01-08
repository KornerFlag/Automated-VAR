"""
Roboflow Detection Module

Integrates with Roboflow API for cloud-based inference.
Supports custom trained models and hybrid local/cloud detection.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import cv2
import base64
import time

from loguru import logger

# Roboflow SDK
try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_SDK_AVAILABLE = True
except ImportError:
    INFERENCE_SDK_AVAILABLE = False
    logger.warning("inference_sdk not installed. Install with: pip install inference-sdk")

try:
    from roboflow import Roboflow
    ROBOFLOW_SDK_AVAILABLE = True
except ImportError:
    ROBOFLOW_SDK_AVAILABLE = False


@dataclass
class RoboflowDetection:
    """Detection result from Roboflow API."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    class_id: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3]
        )


class RoboflowDetector:
    """
    Roboflow API detector for soccer player/ball detection.
    
    Uses Roboflow's hosted inference API for detection.
    Supports custom trained models.
    """
    
    # Default Roboflow configuration
    DEFAULT_API_KEY = "QEZ7CzEaDFxxXMCWMdLn"
    DEFAULT_MODEL_ID = "my-first-project-mbces/1"
    DEFAULT_API_URL = "https://detect.roboflow.com"
    
    def __init__(
        self,
        api_key: str = None,
        model_id: str = None,
        api_url: str = None,
        confidence_threshold: float = 0.5,
        overlap_threshold: float = 0.5
    ):
        """
        Initialize Roboflow detector.
        
        Args:
            api_key: Roboflow API key
            model_id: Model ID in format "project/version"
            api_url: Roboflow API URL
            confidence_threshold: Minimum confidence for detections
            overlap_threshold: IoU threshold for NMS
        """
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.api_url = api_url or self.DEFAULT_API_URL
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold
        
        self.client = None
        self.model = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Roboflow client."""
        if INFERENCE_SDK_AVAILABLE:
            try:
                self.client = InferenceHTTPClient(
                    api_url=self.api_url,
                    api_key=self.api_key
                )
                logger.info(f"Initialized Roboflow client with model: {self.model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Roboflow client: {e}")
                self.client = None
        
        elif ROBOFLOW_SDK_AVAILABLE:
            try:
                rf = Roboflow(api_key=self.api_key)
                project_name, version = self.model_id.rsplit("/", 1)
                project = rf.workspace().project(project_name)
                self.model = project.version(int(version)).model
                logger.info(f"Initialized Roboflow model: {self.model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Roboflow model: {e}")
                self.model = None
        else:
            logger.warning("No Roboflow SDK available")
    
    def detect(self, frame: np.ndarray) -> List[RoboflowDetection]:
        """
        Detect objects in frame using Roboflow API.
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            List of RoboflowDetection objects
        """
        if self.client is None and self.model is None:
            logger.warning("Roboflow client not initialized")
            return []
        
        try:
            if self.client:
                return self._detect_with_client(frame)
            elif self.model:
                return self._detect_with_model(frame)
        except Exception as e:
            logger.error(f"Roboflow detection error: {e}")
            return []
        
        return []
    
    def _detect_with_client(self, frame: np.ndarray) -> List[RoboflowDetection]:
        """Detect using inference SDK client."""
        # Run inference
        result = self.client.infer(frame, model_id=self.model_id)
        
        detections = []
        
        if "predictions" in result:
            for pred in result["predictions"]:
                # Convert center format to corner format
                x_center = pred["x"]
                y_center = pred["y"]
                width = pred["width"]
                height = pred["height"]
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                confidence = pred.get("confidence", 0)
                class_name = pred.get("class", "unknown")
                
                if confidence >= self.confidence_threshold:
                    detections.append(RoboflowDetection(
                        bbox=np.array([x1, y1, x2, y2]),
                        confidence=confidence,
                        class_name=class_name
                    ))
        
        return detections
    
    def _detect_with_model(self, frame: np.ndarray) -> List[RoboflowDetection]:
        """Detect using Roboflow SDK model."""
        # Save frame temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, frame)
            result = self.model.predict(
                f.name,
                confidence=int(self.confidence_threshold * 100),
                overlap=int(self.overlap_threshold * 100)
            ).json()
        
        detections = []
        
        if "predictions" in result:
            for pred in result["predictions"]:
                x_center = pred["x"]
                y_center = pred["y"]
                width = pred["width"]
                height = pred["height"]
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                detections.append(RoboflowDetection(
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=pred.get("confidence", 0),
                    class_name=pred.get("class", "unknown")
                ))
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        delay: float = 0.1
    ) -> List[List[RoboflowDetection]]:
        """
        Detect in multiple frames with rate limiting.
        
        Args:
            frames: List of BGR images
            delay: Delay between API calls (seconds)
        
        Returns:
            List of detection lists for each frame
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
            time.sleep(delay)  # Rate limiting
        
        return all_detections


class HybridDetector:
    """
    Hybrid detector combining local YOLO and Roboflow.
    
    Uses local YOLO for fast inference and Roboflow
    for validation or specialized detection.
    """
    
    def __init__(
        self,
        local_model_path: str = "yolov8x.pt",
        roboflow_api_key: str = None,
        roboflow_model_id: str = None,
        device: str = "cpu",
        use_roboflow_validation: bool = False
    ):
        """
        Initialize hybrid detector.
        
        Args:
            local_model_path: Path to local YOLO model
            roboflow_api_key: Roboflow API key
            roboflow_model_id: Roboflow model ID
            device: Device for local inference
            use_roboflow_validation: Whether to validate with Roboflow
        """
        from .detection import PlayerBallDetector
        
        self.local_detector = PlayerBallDetector(
            model_path=local_model_path,
            device=device
        )
        
        self.roboflow_detector = None
        self.use_roboflow_validation = use_roboflow_validation
        
        if roboflow_api_key or use_roboflow_validation:
            self.roboflow_detector = RoboflowDetector(
                api_key=roboflow_api_key,
                model_id=roboflow_model_id
            )
    
    def detect(self, frame: np.ndarray, use_roboflow: bool = False):
        """
        Detect objects in frame.
        
        Args:
            frame: BGR image
            use_roboflow: Force use of Roboflow API
        
        Returns:
            List of detections
        """
        if use_roboflow and self.roboflow_detector:
            return self.roboflow_detector.detect(frame)
        
        return self.local_detector.detect(frame)


class RoboflowDatasetManager:
    """
    Manager for Roboflow dataset operations.
    
    Handles uploading images, downloading datasets,
    and managing annotations.
    """
    
    def __init__(
        self,
        api_key: str = None,
        workspace: str = None,
        project: str = None
    ):
        """
        Initialize dataset manager.
        
        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Project name
        """
        self.api_key = api_key or RoboflowDetector.DEFAULT_API_KEY
        self.workspace = workspace
        self.project = project
        self.rf = None
        
        if ROBOFLOW_SDK_AVAILABLE:
            self.rf = Roboflow(api_key=self.api_key)
    
    def upload_image(
        self,
        image_path: str,
        annotation_path: str = None,
        split: str = "train"
    ) -> bool:
        """
        Upload image to Roboflow dataset.
        
        Args:
            image_path: Path to image file
            annotation_path: Optional path to annotation file
            split: Dataset split (train/valid/test)
        
        Returns:
            Success status
        """
        if not self.rf:
            logger.error("Roboflow SDK not available")
            return False
        
        try:
            project = self.rf.workspace(self.workspace).project(self.project)
            project.upload(
                image_path,
                annotation_path=annotation_path,
                split=split
            )
            logger.info(f"Uploaded {image_path} to {self.project}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download_dataset(
        self,
        version: int,
        format: str = "yolov8",
        location: str = "./dataset"
    ) -> str:
        """
        Download dataset from Roboflow.
        
        Args:
            version: Dataset version number
            format: Export format (yolov8/coco/voc)
            location: Download location
        
        Returns:
            Path to downloaded dataset
        """
        if not self.rf:
            logger.error("Roboflow SDK not available")
            return ""
        
        try:
            project = self.rf.workspace(self.workspace).project(self.project)
            dataset = project.version(version).download(format, location=location)
            logger.info(f"Downloaded dataset to {location}")
            return location
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return ""
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get project information."""
        if not self.rf:
            return {}
        
        try:
            project = self.rf.workspace(self.workspace).project(self.project)
            return {
                "name": project.name,
                "type": project.type,
                "classes": project.classes,
                "versions": len(project.versions)
            }
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            return {}


# Roboflow training configuration
ROBOFLOW_CONFIG = {
    "api_key": "QEZ7CzEaDFxxXMCWMdLn",
    "model_id": "my-first-project-mbces/1",
    "workspace": "var-project",
    "project": "soccer-player-detection",
    "classes": ["player", "ball", "referee", "goalkeeper"],
    "augmentation": {
        "flip_horizontal": True,
        "flip_vertical": False,
        "rotation": 15,
        "brightness": 0.2,
        "blur": 1.5,
        "noise": 1.5
    },
    "preprocessing": {
        "resize": 640,
        "auto_orient": True,
        "grayscale": False
    }
}


if __name__ == "__main__":
    print("Roboflow Integration Module")
    print(f"Inference SDK available: {INFERENCE_SDK_AVAILABLE}")
    print(f"Roboflow SDK available: {ROBOFLOW_SDK_AVAILABLE}")
    print(f"Default API Key: {RoboflowDetector.DEFAULT_API_KEY[:10]}...")
    print(f"Default Model: {RoboflowDetector.DEFAULT_MODEL_ID}")
