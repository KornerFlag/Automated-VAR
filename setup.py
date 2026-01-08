#!/usr/bin/env python3
"""
Automated VAR System - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="automated-var",
    version="0.2.0",
    author="VAR Development Team",
    description="Automated Video Assistant Referee System for Soccer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/var-project/automated-var",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "roboflow": [
            "roboflow>=1.1.0",
            "inference-sdk>=0.9.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "var-analyze=src.pipeline:main",
            "var-train=training.train_detection:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision soccer var object-detection yolo tracking",
)
