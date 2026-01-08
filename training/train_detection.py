#!/usr/bin/env python3
"""
Detection Model Training Script

Train YOLOv8 model for soccer player/ball detection.
Supports both local datasets and Roboflow integration.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from loguru import logger

# Roboflow configuration
ROBOFLOW_CONFIG = {
    "api_key": "QEZ7CzEaDFxxXMCWMdLn",
    "workspace": "var-project",
    "project": "soccer-player-detection",
    "version": 1,
    "format": "yolov8"
}


def download_roboflow_dataset(
    api_key: str = None,
    workspace: str = None,
    project: str = None,
    version: int = 1,
    format: str = "yolov8",
    location: str = "./data/roboflow"
) -> str:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version
        format: Export format
        location: Download location
    
    Returns:
        Path to downloaded dataset
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Roboflow not installed. Install with: pip install roboflow")
        return None
    
    api_key = api_key or ROBOFLOW_CONFIG["api_key"]
    workspace = workspace or ROBOFLOW_CONFIG["workspace"]
    project = project or ROBOFLOW_CONFIG["project"]
    
    logger.info(f"Downloading dataset from Roboflow...")
    logger.info(f"  Workspace: {workspace}")
    logger.info(f"  Project: {project}")
    logger.info(f"  Version: {version}")
    
    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(format, location=location)
        
        logger.info(f"Dataset downloaded to: {location}")
        return location
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return None


def upload_to_roboflow(
    images_dir: str,
    api_key: str = None,
    workspace: str = None,
    project: str = None,
    split: str = "train"
):
    """
    Upload images to Roboflow for annotation.
    
    Args:
        images_dir: Directory containing images
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        split: Dataset split (train/valid/test)
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Roboflow not installed")
        return
    
    api_key = api_key or ROBOFLOW_CONFIG["api_key"]
    workspace = workspace or ROBOFLOW_CONFIG["workspace"]
    project = project or ROBOFLOW_CONFIG["project"]
    
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    logger.info(f"Uploading {len(image_files)} images to Roboflow...")
    
    for i, img_path in enumerate(image_files):
        try:
            proj.upload(str(img_path), split=split)
            if (i + 1) % 10 == 0:
                logger.info(f"  Uploaded {i + 1}/{len(image_files)}")
        except Exception as e:
            logger.warning(f"  Failed to upload {img_path.name}: {e}")
    
    logger.info("Upload complete!")


def train_detection_model(
    data_path: str,
    model: str = "yolov8x.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = None,
    project_name: str = "var_detection",
    experiment_name: str = None,
    resume: bool = False,
    patience: int = 50,
    save_period: int = 10,
    workers: int = 8,
    augment: bool = True
):
    """
    Train YOLOv8 detection model.
    
    Args:
        data_path: Path to data.yaml file
        model: Base model to use
        epochs: Number of training epochs
        batch_size: Training batch size
        image_size: Input image size
        device: Training device (cuda/mps/cpu)
        project_name: MLflow project name
        experiment_name: Experiment name
        resume: Resume from last checkpoint
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        workers: Number of data loader workers
        augment: Enable data augmentation
    
    Returns:
        Path to best model weights
    """
    # Auto-detect device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info("=" * 60)
    logger.info("YOLOv8 Detection Training")
    logger.info("=" * 60)
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Image Size: {image_size}")
    logger.info("=" * 60)
    
    # Generate experiment name
    if experiment_name is None:
        experiment_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load model
    yolo = YOLO(model)
    
    # Training arguments
    train_args = {
        "data": data_path,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": image_size,
        "device": device,
        "project": project_name,
        "name": experiment_name,
        "patience": patience,
        "save_period": save_period,
        "workers": workers,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": True,
        "close_mosaic": 10,
        "resume": resume,
        "amp": True,  # Automatic mixed precision
        
        # Augmentation
        "augment": augment,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }
    
    # Train
    logger.info("Starting training...")
    results = yolo.train(**train_args)
    
    # Get best model path
    best_model = Path(project_name) / experiment_name / "weights" / "best.pt"
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Best model: {best_model}")
    logger.info("=" * 60)
    
    return str(best_model)


def validate_model(
    model_path: str,
    data_path: str,
    device: str = None,
    batch_size: int = 16,
    image_size: int = 640
):
    """
    Validate trained model.
    
    Args:
        model_path: Path to model weights
        data_path: Path to data.yaml
        device: Validation device
        batch_size: Batch size
        image_size: Image size
    
    Returns:
        Validation metrics
    """
    logger.info("Validating model...")
    
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_path,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        verbose=True
    )
    
    logger.info("Validation Results:")
    logger.info(f"  mAP50: {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


def export_model(
    model_path: str,
    format: str = "onnx",
    image_size: int = 640,
    half: bool = False
):
    """
    Export model to different formats.
    
    Args:
        model_path: Path to model weights
        format: Export format (onnx, torchscript, coreml, etc.)
        image_size: Export image size
        half: Use FP16 half precision
    
    Returns:
        Path to exported model
    """
    logger.info(f"Exporting model to {format}...")
    
    model = YOLO(model_path)
    
    export_path = model.export(
        format=format,
        imgsz=image_size,
        half=half
    )
    
    logger.info(f"Model exported to: {export_path}")
    return export_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 detection model for VAR system"
    )
    
    # Data arguments
    parser.add_argument(
        "--data", 
        default="configs/detection_data.yaml",
        help="Path to data.yaml file"
    )
    parser.add_argument(
        "--roboflow",
        action="store_true",
        help="Download dataset from Roboflow"
    )
    parser.add_argument(
        "--roboflow-version",
        type=int,
        default=1,
        help="Roboflow dataset version"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        default="yolov8x.pt",
        help="Base model (yolov8n/s/m/l/x.pt)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    
    # Output arguments
    parser.add_argument(
        "--project",
        default="var_detection",
        help="Project name for saving results"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name"
    )
    
    # Actions
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export format (onnx, torchscript, etc.)"
    )
    parser.add_argument(
        "--upload",
        default=None,
        help="Upload images directory to Roboflow"
    )
    
    args = parser.parse_args()
    
    # Handle upload
    if args.upload:
        upload_to_roboflow(args.upload)
        return
    
    # Handle Roboflow download
    data_path = args.data
    if args.roboflow:
        data_path = download_roboflow_dataset(version=args.roboflow_version)
        if data_path:
            data_path = f"{data_path}/data.yaml"
        else:
            logger.error("Failed to download Roboflow dataset")
            return
    
    # Validate only
    if args.validate_only:
        if not args.model:
            logger.error("--model required for validation")
            return
        validate_model(args.model, data_path, args.device, args.batch, args.imgsz)
        return
    
    # Export only
    if args.export:
        if not args.model:
            logger.error("--model required for export")
            return
        export_model(args.model, args.export, args.imgsz)
        return
    
    # Train
    best_model = train_detection_model(
        data_path=data_path,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        project_name=args.project,
        experiment_name=args.name,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers
    )
    
    # Validate best model
    if best_model and Path(best_model).exists():
        validate_model(best_model, data_path, args.device)


if __name__ == "__main__":
    main()
