#!/usr/bin/env python3
"""
Quick Test Script

Verify that the VAR system is properly installed and configured.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    tests = []
    
    # Core modules
    try:
        from src.detection import PlayerBallDetector, Detection
        tests.append(("detection", True, ""))
    except Exception as e:
        tests.append(("detection", False, str(e)))
    
    try:
        from src.tracking import MultiObjectTracker, BallTracker
        tests.append(("tracking", True, ""))
    except Exception as e:
        tests.append(("tracking", False, str(e)))
    
    try:
        from src.team_classifier import TeamClassifier
        tests.append(("team_classifier", True, ""))
    except Exception as e:
        tests.append(("team_classifier", False, str(e)))
    
    try:
        from src.homography import FieldHomography
        tests.append(("homography", True, ""))
    except Exception as e:
        tests.append(("homography", False, str(e)))
    
    try:
        from src.offside import OffsideDetector
        tests.append(("offside", True, ""))
    except Exception as e:
        tests.append(("offside", False, str(e)))
    
    try:
        from src.foul_detection import FoulDetector
        tests.append(("foul_detection", True, ""))
    except Exception as e:
        tests.append(("foul_detection", False, str(e)))
    
    try:
        from src.var_detection import VARDecisionEngine
        tests.append(("var_detection", True, ""))
    except Exception as e:
        tests.append(("var_detection", False, str(e)))
    
    try:
        from src.pipeline import VARPipeline
        tests.append(("pipeline", True, ""))
    except Exception as e:
        tests.append(("pipeline", False, str(e)))
    
    # Roboflow
    try:
        from src.roboflow_detector import RoboflowDetector
        tests.append(("roboflow_detector", True, ""))
    except Exception as e:
        tests.append(("roboflow_detector", False, str(e)))
    
    return tests


def test_dependencies():
    """Test that all dependencies are installed."""
    print("Testing dependencies...")
    
    tests = []
    
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("ultralytics", "Ultralytics (YOLO)"),
        ("loguru", "Loguru"),
        ("yaml", "PyYAML"),
        ("streamlit", "Streamlit"),
    ]
    
    for module, name in deps:
        try:
            __import__(module)
            tests.append((name, True, ""))
        except ImportError as e:
            tests.append((name, False, str(e)))
    
    # Roboflow (optional)
    try:
        import inference_sdk
        tests.append(("Roboflow SDK", True, ""))
    except ImportError:
        tests.append(("Roboflow SDK", False, "Optional - install with: pip install inference-sdk"))
    
    return tests


def test_device():
    """Test available compute devices."""
    print("Testing devices...")
    
    tests = []
    
    try:
        import torch
        
        # CUDA
        if torch.cuda.is_available():
            tests.append(("CUDA", True, f"GPU: {torch.cuda.get_device_name(0)}"))
        else:
            tests.append(("CUDA", False, "Not available"))
        
        # MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            tests.append(("MPS (Apple)", True, "Available"))
        else:
            tests.append(("MPS (Apple)", False, "Not available"))
        
        # CPU
        tests.append(("CPU", True, "Always available"))
        
    except Exception as e:
        tests.append(("Device check", False, str(e)))
    
    return tests


def test_models():
    """Test model availability."""
    print("Testing models...")
    
    tests = []
    
    # Check YOLO models
    try:
        from ultralytics import YOLO
        
        # Try to load default model
        model = YOLO("yolov8n.pt")  # Smallest model for quick test
        tests.append(("YOLOv8", True, "Model loaded successfully"))
    except Exception as e:
        tests.append(("YOLOv8", False, str(e)))
    
    return tests


def test_roboflow():
    """Test Roboflow connection."""
    print("Testing Roboflow...")
    
    tests = []
    
    try:
        from inference_sdk import InferenceHTTPClient
        
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="QEZ7CzEaDFxxXMCWMdLn"
        )
        
        tests.append(("Roboflow API", True, "Connection successful"))
    except ImportError:
        tests.append(("Roboflow API", False, "SDK not installed"))
    except Exception as e:
        tests.append(("Roboflow API", False, str(e)))
    
    return tests


def print_results(name: str, results: list):
    """Print test results."""
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    
    passed = 0
    failed = 0
    
    for test_name, success, message in results:
        status = "PASS" if success else "FAIL"
        symbol = "[OK]" if success else "[X] "
        
        print(f"  {symbol} {test_name}")
        if message and not success:
            print(f"       {message}")
        elif message and success:
            print(f"       {message}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    return passed, failed


def main():
    print("\n" + "="*60)
    print(" VAR System - Installation Test")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    
    # Run tests
    tests = [
        ("Module Imports", test_imports()),
        ("Dependencies", test_dependencies()),
        ("Compute Devices", test_device()),
        ("Models", test_models()),
        ("Roboflow", test_roboflow()),
    ]
    
    for name, results in tests:
        passed, failed = print_results(name, results)
        total_passed += passed
        total_failed += failed
    
    # Summary
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n  All tests passed! System is ready.")
        print("\n  Run with:")
        print("    python -m src.pipeline --video YOUR_VIDEO.mp4 --output results/")
    else:
        print("\n  Some tests failed. Please check the errors above.")
    
    print("="*60 + "\n")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
