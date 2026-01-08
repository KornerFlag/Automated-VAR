# Automated VAR System

**Version 0.2.0** | Computer Vision for Soccer Referee Assistance

A comprehensive video analysis platform that automatically detects offside positions, fouls, and penalty incidents in soccer matches using deep learning and computer vision.

---

## Features

| Feature | Description |
|---------|-------------|
| **Offside Detection** | Tracks player positions and detects offside at the moment of a pass |
| **Foul Detection** | Identifies player contacts and falls to detect fouls |
| **Penalty Detection** | Flags fouls occurring inside the penalty area |
| **Team Classification** | Automatically classifies players by jersey color |
| **Field Mapping** | Maps camera view to 2D pitch coordinates |
| **Real-time Tracking** | Multi-object tracking with consistent player IDs |

---

## System Architecture

```
Video Input
    |
    v
+-------------------+
|  Detection        |  YOLOv8 / Roboflow API
|  (Players, Ball)  |
+-------------------+
    |
    v
+-------------------+
|  Tracking         |  Multi-object tracking
|  (ByteTrack)      |
+-------------------+
    |
    v
+-------------------+
|  Team             |  Color-based clustering
|  Classification   |
+-------------------+
    |
    v
+-------------------+
|  Homography       |  Perspective transform
|  (Field Mapping)  |
+-------------------+
    |
    v
+-------------------+
|  VAR Engine       |  Offside, Foul, Penalty
|  Detection        |
+-------------------+
    |
    v
Output (Video + JSON Reports)
```

---

## Installation

### Prerequisites

- Python 3.9+
- pip
- Git
- ffmpeg (for video processing)

### Quick Install

```bash
# Clone or extract the project
cd automated_var

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### GPU Support

**NVIDIA GPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon (M1/M2/M3):**
```bash
# MPS support is built into standard PyTorch
pip install torch torchvision
```

---

## Quick Start

### Command Line

```bash
# Process a video
python -m src.pipeline \
    --video match.mp4 \
    --output results/ \
    --device mps  # or cuda, cpu

# Using Roboflow API (no GPU required)
python -m src.pipeline \
    --video match.mp4 \
    --output results/ \
    --use-roboflow
```

### Web Interface

```bash
streamlit run demo/app.py
```

### Python API

```python
from src import VARPipeline

# Initialize pipeline
pipeline = VARPipeline(device='mps')

# Process video
result = pipeline.process_video(
    video_path='match.mp4',
    output_dir='results/'
)

# Access results
print(f"Offsides: {len(result.offside_incidents)}")
print(f"Fouls: {len(result.foul_incidents)}")
print(f"Penalties: {len(result.penalty_incidents)}")
```

---

## Roboflow Integration

### API Configuration

The system includes Roboflow integration for cloud-based detection:

```yaml
# configs/config.yaml
roboflow:
  api_key: "QEZ7CzEaDFxxXMCWMdLn"
  model_id: "my-first-project-mbces/1"
  workspace: "var-project"
  project: "soccer-player-detection"
```

### Training with Roboflow

1. **Upload training data:**
```bash
python training/train_detection.py --upload data/raw/frames/
```

2. **Download annotated dataset:**
```bash
python training/train_detection.py --roboflow --roboflow-version 1
```

3. **Train model:**
```bash
python training/train_detection.py \
    --data configs/detection_data.yaml \
    --model yolov8x.pt \
    --epochs 100 \
    --batch 16 \
    --device cuda
```

---

## Output Files

| File | Description |
|------|-------------|
| `annotated_video.mp4` | Video with detection overlays |
| `pitch_view.mp4` | 2D top-down tactical view |
| `incidents.json` | Complete incident summary |
| `offsides.json` | Offside detections |
| `fouls.json` | Foul detections |
| `penalties.json` | Penalty incidents |

### Incident JSON Format

```json
{
  "type": "offside",
  "frame": 1523,
  "timestamp": 50.77,
  "confidence": 0.87,
  "position": [35.2, 12.5],
  "players": [7],
  "description": "Player 7 is 45.2cm offside"
}
```

---

## Configuration

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_confidence` | 0.5 | Minimum detection confidence |
| `offside_tolerance_cm` | 15 | Offside tolerance in cm |
| `contact_threshold` | 1.5 | Contact distance in meters |
| `fall_threshold` | 0.3 | Height drop ratio for falls |

### Device Selection

| Device | Flag | Use Case |
|--------|------|----------|
| NVIDIA GPU | `--device cuda` | Windows/Linux with NVIDIA |
| Apple Silicon | `--device mps` | Mac M1/M2/M3 |
| CPU | `--device cpu` | Universal (slower) |

---

## Project Structure

```
automated_var/
├── src/                        # Source code
│   ├── __init__.py
│   ├── detection.py            # YOLOv8 detection
│   ├── tracking.py             # Multi-object tracking
│   ├── team_classifier.py      # Team classification
│   ├── homography.py           # Field mapping
│   ├── offside.py              # Offside detection
│   ├── foul_detection.py       # Foul detection
│   ├── var_detection.py        # VAR engine
│   ├── roboflow_detector.py    # Roboflow integration
│   └── pipeline.py             # Main pipeline
├── training/                   # Training scripts
│   ├── train_detection.py      # Detection training
│   └── train_foul_classifier.py # Foul classifier
├── demo/                       # Web interface
│   └── app.py                  # Streamlit dashboard
├── configs/                    # Configuration files
│   ├── config.yaml
│   └── detection_data.yaml
├── scripts/                    # Utility scripts
│   ├── download_models.py
│   ├── extract_frames.py
│   └── quick_test.py
├── data/                       # Training data
├── models/                     # Trained models
└── requirements.txt
```

---

## Troubleshooting

### CUDA not available
```
Error: Torch not compiled with CUDA enabled
Solution: Use --device cpu or --device mps (Mac)
```

### MediaPipe error
```
Error: module 'mediapipe' has no attribute 'solutions'
Solution: Set use_pose = False in src/foul_detection.py
```

### Out of memory
```
Error: CUDA out of memory
Solution: Reduce batch size or use smaller model (yolov8m.pt)
```

---

## Technical Details

### Pitch Dimensions
- Length: 105m (configurable)
- Width: 68m (configurable)
- Penalty area: 16.5m x 40.3m

### Detection Thresholds
- Offside tolerance: 15cm
- Contact threshold: 1.5m
- Fall detection: 30% height decrease
- Pass velocity: 8 m/s

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- YOLOv8 by Ultralytics
- Roboflow for dataset management
- OpenCV for computer vision
- Streamlit for web interface
# Automated-VAR
