"""
Training Module

Scripts for training detection and classification models.
"""

from .train_detection import (
    train_detection_model,
    download_roboflow_dataset,
    upload_to_roboflow,
    validate_model,
    export_model,
    ROBOFLOW_CONFIG
)

from .train_foul_classifier import (
    train_foul_classifier,
    prepare_foul_dataset,
    create_foul_classifier
)

__all__ = [
    'train_detection_model',
    'download_roboflow_dataset',
    'upload_to_roboflow',
    'validate_model',
    'export_model',
    'train_foul_classifier',
    'prepare_foul_dataset',
    'create_foul_classifier',
    'ROBOFLOW_CONFIG'
]
