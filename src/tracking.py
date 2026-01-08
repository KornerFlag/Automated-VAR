"""
Multi-Object Tracking Module

Implements ByteTrack-style tracking for players and ball.
Maintains consistent IDs across frames.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .detection import Detection

from loguru import logger


@dataclass
class TrackedObject:
    """A tracked object with history."""
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_name: str
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
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
    
    def update(self, detection: Detection):
        """Update track with new detection."""
        # Calculate velocity
        old_center = np.array(self.center)
        new_center = np.array(detection.center)
        self.velocity = new_center - old_center
        
        # Update state
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
        # Add to history
        self.history.append(self.center)
    
    def predict(self):
        """Predict next position using velocity."""
        self.bbox[0] += self.velocity[0]
        self.bbox[2] += self.velocity[0]
        self.bbox[1] += self.velocity[1]
        self.bbox[3] += self.velocity[1]
        self.time_since_update += 1
        self.age += 1


class MultiObjectTracker:
    """
    Multi-object tracker using Hungarian algorithm.
    
    Maintains consistent track IDs across frames using
    IoU-based matching and Kalman filtering concepts.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching
            track_thresh: Confidence threshold for new tracks
            match_thresh: Threshold for matching score
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        
        self.tracks: List[TrackedObject] = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray = None
    ) -> List[TrackedObject]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections for current frame
            frame: Optional frame for appearance features
        
        Returns:
            List of active tracked objects
        """
        self.frame_count += 1
        
        # Filter detections by class (only track players)
        player_detections = [d for d in detections if d.class_name == "player"]
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        if len(self.tracks) > 0 and len(player_detections) > 0:
            matched, unmatched_dets, unmatched_tracks = self._match(
                player_detections
            )
        else:
            matched = []
            unmatched_dets = list(range(len(player_detections)))
            unmatched_tracks = list(range(len(self.tracks)))
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(player_detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = player_detections[det_idx]
            if det.confidence >= self.track_thresh:
                new_track = TrackedObject(
                    track_id=self.next_id,
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    class_name=det.class_name
                )
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update < self.max_age
        ]
        
        # Return confirmed tracks
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]
    
    def _match(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            matched: List of (track_idx, det_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array([d.bbox for d in detections])
        
        iou_matrix = self._calculate_iou_matrix(track_boxes, det_boxes)
        
        # Hungarian algorithm for optimal assignment
        if iou_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched.append((row, col))
                    if col in unmatched_dets:
                        unmatched_dets.remove(col)
                    if row in unmatched_tracks:
                        unmatched_tracks.remove(row)
            
            return matched, unmatched_dets, unmatched_tracks
        
        return [], list(range(len(detections))), list(range(len(self.tracks)))
    
    def _calculate_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes."""
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._calculate_iou(boxes1[i], boxes2[j])
        
        return iou_matrix
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0


class BallTracker:
    """
    Specialized tracker for ball with smoothing.
    
    Uses exponential smoothing and prediction for
    handling occlusions and fast movement.
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.7,
        max_missing_frames: int = 10,
        velocity_smoothing: float = 0.5
    ):
        """
        Initialize ball tracker.
        
        Args:
            smoothing_factor: Weight for exponential smoothing
            max_missing_frames: Maximum frames without detection
            velocity_smoothing: Smoothing factor for velocity
        """
        self.smoothing_factor = smoothing_factor
        self.max_missing_frames = max_missing_frames
        self.velocity_smoothing = velocity_smoothing
        
        self.position: Optional[np.ndarray] = None
        self.velocity: np.ndarray = np.zeros(2)
        self.missing_frames = 0
        self.history: deque = deque(maxlen=30)
    
    def update(self, detection: Optional[Detection]) -> Optional[Tuple[float, float]]:
        """
        Update ball position with new detection.
        
        Args:
            detection: Ball detection or None if not detected
        
        Returns:
            Estimated ball position or None
        """
        if detection is not None:
            new_pos = np.array(detection.center)
            
            if self.position is not None:
                # Update velocity
                new_velocity = new_pos - self.position
                self.velocity = (
                    self.velocity_smoothing * self.velocity +
                    (1 - self.velocity_smoothing) * new_velocity
                )
                
                # Smooth position
                self.position = (
                    self.smoothing_factor * new_pos +
                    (1 - self.smoothing_factor) * self.position
                )
            else:
                self.position = new_pos
            
            self.missing_frames = 0
            self.history.append(tuple(self.position))
            
            return tuple(self.position)
        
        else:
            self.missing_frames += 1
            
            if self.position is not None and self.missing_frames <= self.max_missing_frames:
                # Predict position using velocity
                self.position = self.position + self.velocity
                self.history.append(tuple(self.position))
                return tuple(self.position)
            
            if self.missing_frames > self.max_missing_frames:
                self.position = None
                self.velocity = np.zeros(2)
            
            return None
    
    def reset(self):
        """Reset tracker state."""
        self.position = None
        self.velocity = np.zeros(2)
        self.missing_frames = 0
        self.history.clear()


def draw_tracks(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    team_assignments: Dict[int, str] = None,
    team_colors: Dict[str, Tuple[int, int, int]] = None,
    draw_trails: bool = True
) -> np.ndarray:
    """
    Draw tracked objects on frame.
    
    Args:
        frame: BGR image
        tracks: List of tracked objects
        team_assignments: Optional team assignment for each track
        team_colors: Colors for each team
        draw_trails: Whether to draw movement trails
    
    Returns:
        Annotated frame
    """
    if team_colors is None:
        team_colors = {
            "team_a": (255, 0, 0),
            "team_b": (0, 0, 255),
            "referee": (255, 255, 0),
            "unknown": (128, 128, 128)
        }
    
    output = frame.copy()
    
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        
        # Get color
        if team_assignments and track.track_id in team_assignments:
            team = team_assignments[track.track_id]
            color = team_colors.get(team, (128, 128, 128))
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID
        label = f"ID:{track.track_id}"
        cv2.putText(
            output, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        
        # Draw trail
        if draw_trails and len(track.history) > 1:
            points = list(track.history)
            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))
                alpha = i / len(points)
                thickness = max(1, int(alpha * 3))
                cv2.line(output, pt1, pt2, color, thickness)
    
    return output
