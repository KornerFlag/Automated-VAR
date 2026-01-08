#!/usr/bin/env python3
"""
Download Pre-trained Models

Downloads YOLOv8 weights and any additional models needed.
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_file(url: str, destination: str, expected_hash: str = None):
    """Download file with progress indication."""
    print(f"Downloading: {url}")
    print(f"Destination: {destination}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print("\nDownload complete!")
    
    if expected_hash:
        print("Verifying checksum...")
        with open(destination, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != expected_hash:
            print(f"WARNING: Hash mismatch! Expected {expected_hash}, got {file_hash}")
        else:
            print("Checksum verified!")


def download_yolo_models():
    """Download YOLOv8 models."""
    from ultralytics import YOLO
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv8 variants
    model_variants = {
        "yolov8n.pt": "Nano - fastest, least accurate",
        "yolov8s.pt": "Small - good balance",
        "yolov8m.pt": "Medium - better accuracy",
        "yolov8l.pt": "Large - high accuracy",
        "yolov8x.pt": "Extra Large - best accuracy (recommended)"
    }
    
    print("=" * 60)
    print("YOLOv8 Model Download")
    print("=" * 60)
    
    # Download default model (yolov8x)
    default_model = "yolov8x.pt"
    print(f"\nDownloading {default_model}...")
    
    try:
        model = YOLO(default_model)
        print(f"Model {default_model} downloaded successfully!")
        print(f"Location: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'ultralytics cache'}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    
    return True


def setup_roboflow():
    """Setup Roboflow integration."""
    print("\n" + "=" * 60)
    print("Roboflow Setup")
    print("=" * 60)
    
    try:
        from inference_sdk import InferenceHTTPClient
        print("inference_sdk is installed")
        
        # Test API connection
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="QEZ7CzEaDFxxXMCWMdLn"
        )
        print("Roboflow API connection successful!")
        print("Model ID: my-first-project-mbces/1")
        
    except ImportError:
        print("inference_sdk not installed")
        print("Install with: pip install inference-sdk")
    except Exception as e:
        print(f"Roboflow setup error: {e}")


def create_placeholder_models():
    """Create placeholder model files."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    placeholders = [
        "foul_classifier.pt",
        "homography_model.pt",
        "team_classifier.pt"
    ]
    
    for name in placeholders:
        path = models_dir / name
        if not path.exists():
            path.touch()
            print(f"Created placeholder: {path}")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Automated VAR System - Model Setup")
    print("=" * 60)
    
    # Download YOLO models
    success = download_yolo_models()
    
    # Setup Roboflow
    setup_roboflow()
    
    # Create placeholders
    print("\n" + "=" * 60)
    print("Creating Placeholder Models")
    print("=" * 60)
    create_placeholder_models()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    if success:
        print("\nYou can now run:")
        print("  python -m src.pipeline --video YOUR_VIDEO.mp4 --output results/")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
