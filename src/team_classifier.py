"""
Team Classification Module

Classifies players into teams based on jersey colors.
Uses color histogram clustering and temporal smoothing.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from loguru import logger


class TeamClassifier:
    """
    Classifies players into teams based on jersey color.
    
    Uses K-means clustering on color histograms with
    temporal smoothing for consistent assignments.
    """
    
    def __init__(
        self,
        n_teams: int = 2,
        method: str = "clustering",
        temporal_smoothing: bool = True,
        smoothing_window: int = 10,
        color_space: str = "hsv"
    ):
        """
        Initialize classifier.
        
        Args:
            n_teams: Number of teams to classify
            method: Classification method (clustering/predefined)
            temporal_smoothing: Whether to smooth assignments over time
            smoothing_window: Number of frames for smoothing
            color_space: Color space for features (hsv/lab/rgb)
        """
        self.n_teams = n_teams
        self.method = method
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        self.color_space = color_space
        
        # Color model
        self.kmeans: Optional[KMeans] = None
        self.team_colors: Dict[str, np.ndarray] = {}
        
        # Temporal smoothing
        self.assignment_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )
        
        # Predefined colors (if using predefined method)
        self.predefined_colors = {
            "team_a": np.array([0, 0, 255]),    # Red
            "team_b": np.array([255, 255, 255]), # White
            "referee": np.array([0, 255, 255])   # Yellow
        }
    
    def extract_color_features(
        self,
        frame: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """
        Extract color features from player region.
        
        Args:
            frame: BGR image
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Color feature vector
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return np.zeros(12)
        
        # Focus on upper body (jersey area)
        h = region.shape[0]
        upper_region = region[int(h * 0.1):int(h * 0.5), :]
        
        if upper_region.size == 0:
            upper_region = region
        
        # Convert color space
        if self.color_space == "hsv":
            color_region = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            color_region = cv2.cvtColor(upper_region, cv2.COLOR_BGR2LAB)
        else:
            color_region = upper_region
        
        # Calculate histogram features
        features = []
        for channel in range(3):
            hist = cv2.calcHist(
                [color_region], [channel], None, [8], [0, 256]
            )
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist[:4])  # Use first 4 bins
        
        return np.array(features)
    
    def classify(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int] = None
    ) -> Dict[int, str]:
        """
        Classify players into teams.
        
        Args:
            frame: BGR image
            bboxes: List of bounding boxes
            track_ids: Optional list of track IDs
        
        Returns:
            Dictionary mapping track_id to team name
        """
        if len(bboxes) < self.n_teams:
            logger.warning(f"Not enough players ({len(bboxes)}) for clustering")
            return {}
        
        if track_ids is None:
            track_ids = list(range(len(bboxes)))
        
        # Extract features
        features = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            feat = self.extract_color_features(frame, bbox)
            if feat is not None and not np.all(feat == 0):
                features.append(feat)
                valid_indices.append(i)
        
        if len(features) < self.n_teams:
            return {}
        
        features = np.array(features)
        
        # Cluster
        if self.method == "clustering":
            assignments = self._cluster_classify(features)
        else:
            assignments = self._predefined_classify(features)
        
        # Build result
        result = {}
        for i, idx in enumerate(valid_indices):
            track_id = track_ids[idx]
            team = assignments[i]
            
            # Apply temporal smoothing
            if self.temporal_smoothing:
                self.assignment_history[track_id].append(team)
                team = self._smooth_assignment(track_id)
            
            result[track_id] = team
        
        return result
    
    def _cluster_classify(self, features: np.ndarray) -> List[str]:
        """Classify using K-means clustering."""
        n_clusters = min(self.n_teams + 1, len(features))  # +1 for referee
        
        # Fit or update K-means
        if self.kmeans is None or len(features) > 10:
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            labels = self.kmeans.fit_predict(features)
        else:
            labels = self.kmeans.predict(features)
        
        # Map cluster labels to team names
        team_names = ["team_a", "team_b", "referee"]
        return [team_names[min(label, len(team_names) - 1)] for label in labels]
    
    def _predefined_classify(self, features: np.ndarray) -> List[str]:
        """Classify based on predefined team colors."""
        assignments = []
        
        for feat in features:
            # Simple comparison with predefined colors
            min_dist = float('inf')
            best_team = "unknown"
            
            for team, color in self.predefined_colors.items():
                dist = np.linalg.norm(feat[:3] - color / 255.0)
                if dist < min_dist:
                    min_dist = dist
                    best_team = team
            
            assignments.append(best_team)
        
        return assignments
    
    def _smooth_assignment(self, track_id: int) -> str:
        """Get smoothed team assignment using voting."""
        history = self.assignment_history[track_id]
        
        if not history:
            return "unknown"
        
        # Count votes
        votes = defaultdict(int)
        for team in history:
            votes[team] += 1
        
        # Return most common
        return max(votes.keys(), key=lambda x: votes[x])
    
    def set_team_colors(
        self,
        team_a_color: Tuple[int, int, int],
        team_b_color: Tuple[int, int, int],
        referee_color: Tuple[int, int, int] = (0, 255, 255)
    ):
        """Set predefined team colors."""
        self.predefined_colors = {
            "team_a": np.array(team_a_color),
            "team_b": np.array(team_b_color),
            "referee": np.array(referee_color)
        }
    
    def reset(self):
        """Reset classifier state."""
        self.kmeans = None
        self.assignment_history.clear()


def get_team_colors() -> Dict[str, Tuple[int, int, int]]:
    """Get default team colors for visualization."""
    return {
        "team_a": (255, 0, 0),      # Blue (BGR)
        "team_b": (0, 0, 255),      # Red (BGR)
        "referee": (0, 255, 255),   # Yellow (BGR)
        "unknown": (128, 128, 128)  # Gray (BGR)
    }


def extract_dominant_colors(
    image: np.ndarray,
    n_colors: int = 3,
    mask: np.ndarray = None
) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from image region.
    
    Args:
        image: BGR image
        n_colors: Number of colors to extract
        mask: Optional mask for region of interest
    
    Returns:
        List of dominant colors (BGR)
    """
    # Reshape image
    pixels = image.reshape(-1, 3)
    
    if mask is not None:
        mask_flat = mask.flatten()
        pixels = pixels[mask_flat > 0]
    
    if len(pixels) < n_colors:
        return [(0, 0, 0)] * n_colors
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Sort by frequency
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    colors = []
    for idx in sorted_indices:
        center = kmeans.cluster_centers_[idx]
        colors.append(tuple(map(int, center)))
    
    return colors
