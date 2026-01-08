"""
Automated VAR System - Source Package

A computer vision system for automated referee assistance in soccer.

Features:
- Player and ball detection (YOLOv8 / Roboflow)
- Multi-object tracking  
- Team classification
- Field homography mapping
- OFFSIDE detection
- FOUL detection
- PENALTY detection
"""

from .detection import PlayerBallDetector, Detection
from .tracking import MultiObjectTracker, BallTracker, TrackedObject
from .team_classifier import TeamClassifier, get_team_colors
from .homography import FieldHomography, PitchDimensions
from .pipeline import VARPipeline, FrameResult, VideoResult

# VAR Detection Engine
try:
    from .var_detection import (
        VARDecisionEngine,
        VARIncident,
        IncidentType,
        EnhancedOffsideDetector,
        EnhancedFoulDetector,
        PassDetector as VARPassDetector
    )
    VAR_ENGINE_AVAILABLE = True
except ImportError:
    VAR_ENGINE_AVAILABLE = False

# Legacy detection
try:
    from .offside import OffsideDetector, OffsideResult, AttackingDirection
    from .offside import PassDetector as LegacyPassDetector
except ImportError:
    pass

try:
    from .foul_detection import FoulDetector, FoulResult, FoulSeverity
except ImportError:
    pass

# Roboflow integration
try:
    from .roboflow_detector import RoboflowDetector, ROBOFLOW_CONFIG
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

__version__ = "0.2.0"
__author__ = "Automated VAR Team"

__all__ = [
    'PlayerBallDetector',
    'Detection',
    'MultiObjectTracker',
    'BallTracker',
    'TrackedObject',
    'TeamClassifier',
    'FieldHomography',
    'PitchDimensions',
    'VARPipeline',
    'FrameResult', 
    'VideoResult',
    'VARDecisionEngine',
    'VARIncident',
    'IncidentType',
    'OffsideDetector',
    'FoulDetector',
    'RoboflowDetector',
    'VAR_ENGINE_AVAILABLE',
    'ROBOFLOW_AVAILABLE',
]
