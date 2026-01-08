#!/usr/bin/env python3
"""
Foul Classifier Training Script

Train a classifier to detect fouls from video clips.
Uses transfer learning with pre-trained action recognition models.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

# Roboflow configuration for foul dataset
ROBOFLOW_FOUL_CONFIG = {
    "api_key": "QEZ7CzEaDFxxXMCWMdLn",
    "workspace": "var-project",
    "project": "soccer-foul-detection",
    "version": 1
}


def prepare_foul_dataset(
    video_dir: str,
    output_dir: str,
    clip_length: int = 32,
    frame_rate: int = 15
):
    """
    Prepare foul detection dataset from video clips.
    
    Args:
        video_dir: Directory containing labeled video clips
        output_dir: Output directory for processed data
        clip_length: Number of frames per clip
        frame_rate: Target frame rate
    
    Expected structure:
        video_dir/
            foul/
                clip1.mp4
                clip2.mp4
            no_foul/
                clip3.mp4
                clip4.mp4
    """
    import cv2
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_path = Path(video_dir)
    
    dataset = {
        "clips": [],
        "labels": [],
        "metadata": {
            "clip_length": clip_length,
            "frame_rate": frame_rate,
            "classes": ["no_foul", "foul"]
        }
    }
    
    for label_idx, label in enumerate(["no_foul", "foul"]):
        label_dir = video_path / label
        if not label_dir.exists():
            continue
        
        videos = list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.avi"))
        
        for video_file in videos:
            logger.info(f"Processing: {video_file.name}")
            
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame skip
            skip = max(1, int(fps / frame_rate))
            
            frames = []
            frame_idx = 0
            
            while len(frames) < clip_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % skip == 0:
                    # Resize to standard size
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
            
            if len(frames) == clip_length:
                # Save frames as numpy array
                clip_name = f"{label}_{video_file.stem}.npy"
                clip_path = output_path / clip_name
                np.save(clip_path, np.array(frames))
                
                dataset["clips"].append(clip_name)
                dataset["labels"].append(label_idx)
    
    # Save dataset metadata
    with open(output_path / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Dataset prepared: {len(dataset['clips'])} clips")
    return str(output_path)


def create_foul_classifier(
    num_classes: int = 2,
    pretrained: bool = True
):
    """
    Create foul classification model.
    
    Uses a simple CNN-based approach for compatibility.
    For production, consider using video transformers or 3D CNNs.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError:
        logger.error("PyTorch not installed")
        return None
    
    class FoulClassifier(nn.Module):
        def __init__(self, num_classes=2, clip_length=32):
            super().__init__()
            
            # Use ResNet18 as backbone
            self.backbone = models.resnet18(pretrained=pretrained)
            
            # Remove final FC layer
            self.features = nn.Sequential(*list(self.backbone.children())[:-1])
            
            # Temporal pooling
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            # x shape: (batch, frames, channels, height, width)
            batch_size, num_frames = x.shape[:2]
            
            # Process each frame
            x = x.view(-1, *x.shape[2:])  # (batch*frames, C, H, W)
            features = self.features(x)  # (batch*frames, 512, 1, 1)
            features = features.view(batch_size, num_frames, -1)  # (batch, frames, 512)
            
            # Temporal pooling
            features = features.permute(0, 2, 1)  # (batch, 512, frames)
            features = self.temporal_pool(features)  # (batch, 512, 1)
            features = features.squeeze(-1)  # (batch, 512)
            
            # Classify
            output = self.classifier(features)
            return output
    
    model = FoulClassifier(num_classes=num_classes)
    return model


def train_foul_classifier(
    data_dir: str,
    output_dir: str = "models/foul_classifier",
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = None
):
    """
    Train foul classifier.
    
    Args:
        data_dir: Directory with prepared dataset
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Training device
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        logger.error("PyTorch not installed")
        return None
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info("=" * 60)
    logger.info("Foul Classifier Training")
    logger.info("=" * 60)
    logger.info(f"  Data: {data_dir}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info("=" * 60)
    
    # Custom dataset
    class FoulDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            
            with open(self.data_dir / "dataset.json") as f:
                self.metadata = json.load(f)
            
            self.clips = self.metadata["clips"]
            self.labels = self.metadata["labels"]
        
        def __len__(self):
            return len(self.clips)
        
        def __getitem__(self, idx):
            clip_path = self.data_dir / self.clips[idx]
            frames = np.load(clip_path)
            
            # Normalize and convert to tensor
            frames = frames.astype(np.float32) / 255.0
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
            
            label = self.labels[idx]
            return frames, label
    
    # Create dataset and loader
    dataset = FoulDataset(data_dir)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = create_foul_classifier()
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                labels = labels.to(device)
                
                outputs = model(clips)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")
            logger.info(f"  Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # Save final model
    torch.save(model.state_dict(), output_path / "final_model.pt")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    logger.info(f"  Model saved to: {output_path}")
    logger.info("=" * 60)
    
    return str(output_path / "best_model.pt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train foul classifier for VAR system"
    )
    
    parser.add_argument(
        "--prepare",
        default=None,
        help="Prepare dataset from video directory"
    )
    parser.add_argument(
        "--data",
        default="data/foul_clips",
        help="Path to prepared dataset"
    )
    parser.add_argument(
        "--output",
        default="models/foul_classifier",
        help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    
    args = parser.parse_args()
    
    # Prepare dataset
    if args.prepare:
        prepare_foul_dataset(args.prepare, args.data)
        return
    
    # Train
    train_foul_classifier(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
