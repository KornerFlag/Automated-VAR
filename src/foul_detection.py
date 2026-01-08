"""
Foul Detection Module

Detects fouls, tackles, and card-worthy incidents.
Uses contact detection and fall analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import cv2

from loguru import logger


class FoulSeverity(Enum):
    """Severity level of foul."""
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    VIOLENT = 4


@dataclass
class FoulResult:
    """Result of foul detection."""
    is_foul: bool
    severity: FoulSeverity
    fouling_player_id: Optional[int]
    fouled_player_id: Optional[int]
    position: Optional[Tuple[float, float]]
    is_penalty: bool
    confidence: float
    frame_number: int
    timestamp: float
    description: str = ""


@dataclass
class ContactEvent:
    """A detected contact between players."""
    player1_id: int
    player2_id: int
    position: Tuple[float, float]
    distance: float
    frame_number: int
    duration: int = 1


class FallDetector:
    """
    Detects player falls using bounding box analysis.
    
    Monitors height changes and aspect ratio to detect falls.
    """
    
    def __init__(
        self,
        height_drop_threshold: float = 0.3,
        aspect_ratio_threshold: float = 1.5,
        min_fall_frames: int = 3
    ):
        """
        Initialize fall detector.
        
        Args:
            height_drop_threshold: Minimum height drop ratio for fall
            aspect_ratio_threshold: Aspect ratio indicating lying down
            min_fall_frames: Minimum frames for confirmed fall
        """
        self.height_drop_threshold = height_drop_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.min_fall_frames = min_fall_frames
        
        # Height tracking per player
        self.player_heights: Dict[int, deque] = {}
        
        # Fall state per player
        self.fall_state: Dict[int, int] = {}  # frames in fall state
        
        # Use pose estimation (disabled for compatibility)
        self.use_pose = False
    
    def update(
        self,
        player_bboxes: Dict[int, np.ndarray]
    ) -> Dict[int, bool]:
        """
        Update fall detection with new bounding boxes.
        
        Args:
            player_bboxes: Dictionary mapping player ID to bbox
        
        Returns:
            Dictionary mapping player ID to fall status
        """
        falls = {}
        
        for player_id, bbox in player_bboxes.items():
            # Initialize tracking if needed
            if player_id not in self.player_heights:
                self.player_heights[player_id] = deque(maxlen=15)
                self.fall_state[player_id] = 0
            
            # Calculate height and aspect ratio
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            aspect_ratio = width / max(height, 1)
            
            self.player_heights[player_id].append(height)
            
            # Check for fall indicators
            is_falling = False
            
            # Check height drop
            if len(self.player_heights[player_id]) >= 5:
                initial_height = np.mean(list(self.player_heights[player_id])[:3])
                if initial_height > 0:
                    height_ratio = height / initial_height
                    if height_ratio < (1 - self.height_drop_threshold):
                        is_falling = True
            
            # Check aspect ratio (lying down)
            if aspect_ratio > self.aspect_ratio_threshold:
                is_falling = True
            
            # Update fall state
            if is_falling:
                self.fall_state[player_id] += 1
            else:
                self.fall_state[player_id] = max(0, self.fall_state[player_id] - 1)
            
            # Confirm fall
            falls[player_id] = self.fall_state[player_id] >= self.min_fall_frames
        
        return falls
    
    def detect_falls(
        self,
        frame: np.ndarray,
        frame_number: int,
        player_bboxes: Dict[int, np.ndarray]
    ) -> Dict[int, bool]:
        """
        Detect falls in frame.
        
        Args:
            frame: BGR image
            frame_number: Current frame number
            player_bboxes: Player bounding boxes
        
        Returns:
            Dictionary mapping player ID to fall status
        """
        return self.update(player_bboxes)
    
    def reset(self):
        """Reset detector state."""
        self.player_heights.clear()
        self.fall_state.clear()


class ContactDetector:
    """
    Detects contact between players.
    
    Uses distance thresholds and trajectory analysis.
    """
    
    def __init__(
        self,
        contact_threshold: float = 1.5,
        min_contact_frames: int = 2
    ):
        """
        Initialize contact detector.
        
        Args:
            contact_threshold: Maximum distance for contact (meters)
            min_contact_frames: Minimum frames for confirmed contact
        """
        self.contact_threshold = contact_threshold
        self.min_contact_frames = min_contact_frames
        
        # Contact tracking
        self.contact_history: Dict[Tuple[int, int], int] = {}
    
    def detect(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str]
    ) -> List[ContactEvent]:
        """
        Detect contacts between players.
        
        Args:
            player_positions: Player positions in pitch coordinates
            team_assignments: Team assignments
        
        Returns:
            List of contact events
        """
        contacts = []
        player_ids = list(player_positions.keys())
        
        for i, p1_id in enumerate(player_ids):
            for p2_id in player_ids[i + 1:]:
                # Skip same team
                team1 = team_assignments.get(p1_id, "unknown")
                team2 = team_assignments.get(p2_id, "unknown")
                
                if team1 == team2:
                    continue
                if "referee" in [team1, team2]:
                    continue
                if "unknown" in [team1, team2]:
                    continue
                
                # Check distance
                pos1 = np.array(player_positions[p1_id])
                pos2 = np.array(player_positions[p2_id])
                distance = np.linalg.norm(pos1 - pos2)
                
                pair_key = (min(p1_id, p2_id), max(p1_id, p2_id))
                
                if distance < self.contact_threshold:
                    self.contact_history[pair_key] = \
                        self.contact_history.get(pair_key, 0) + 1
                    
                    if self.contact_history[pair_key] >= self.min_contact_frames:
                        contact_pos = tuple((pos1 + pos2) / 2)
                        
                        contacts.append(ContactEvent(
                            player1_id=p1_id,
                            player2_id=p2_id,
                            position=contact_pos,
                            distance=distance,
                            frame_number=0,  # Set by caller
                            duration=self.contact_history[pair_key]
                        ))
                else:
                    self.contact_history[pair_key] = 0
        
        return contacts
    
    def reset(self):
        """Reset detector state."""
        self.contact_history.clear()


class FoulDetector:
    """
    Main foul detection system.
    
    Combines contact detection, fall detection, and
    ball proximity analysis to identify fouls.
    """
    
    def __init__(
        self,
        model_path: str = None,
        penalty_area_only: bool = False,
        contact_threshold: float = 1.5,
        device: str = "cpu"
    ):
        """
        Initialize foul detector.
        
        Args:
            model_path: Optional path to foul classifier model
            penalty_area_only: Only detect fouls in penalty area
            contact_threshold: Distance threshold for contact
            device: Device for inference
        """
        self.penalty_area_only = penalty_area_only
        self.device = device
        
        # Sub-detectors
        self.fall_detector = FallDetector()
        self.contact_detector = ContactDetector(
            contact_threshold=contact_threshold
        )
        
        # Results history
        self.foul_history: List[FoulResult] = []
        
        # FPS for timestamps
        self.fps = 30.0
    
    def update(
        self,
        frame: np.ndarray,
        frame_number: int,
        player_bboxes: Dict[int, np.ndarray],
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        ball_position: Optional[Tuple[float, float]] = None,
        is_penalty_area_func=None
    ) -> List[FoulResult]:
        """
        Process frame and detect fouls.
        
        Args:
            frame: BGR image
            frame_number: Current frame number
            player_bboxes: Player bounding boxes
            player_positions: Player pitch positions
            team_assignments: Team assignments
            ball_position: Ball position
            is_penalty_area_func: Function to check penalty area
        
        Returns:
            List of detected fouls
        """
        timestamp = frame_number / self.fps
        fouls = []
        
        # Detect falls
        falls = self.fall_detector.detect_falls(frame, frame_number, player_bboxes)
        
        # Detect contacts
        contacts = self.contact_detector.detect(
            player_positions, team_assignments
        )
        
        # Analyze contacts for fouls
        for contact in contacts:
            contact.frame_number = frame_number
            
            # Check if either player fell
            player1_fell = falls.get(contact.player1_id, False)
            player2_fell = falls.get(contact.player2_id, False)
            
            if not (player1_fell or player2_fell):
                continue
            
            # Check if in penalty area
            is_penalty = False
            if is_penalty_area_func and contact.position:
                is_penalty = is_penalty_area_func(
                    contact.position[0], contact.position[1]
                )
            
            if self.penalty_area_only and not is_penalty:
                continue
            
            # Calculate foul probability
            confidence = self._calculate_foul_probability(
                contact, player1_fell, player2_fell, ball_position
            )
            
            if confidence < 0.5:
                continue
            
            # Determine severity
            severity = self._determine_severity(
                confidence, contact.duration
            )
            
            # Determine fouler and fouled
            if player1_fell and not player2_fell:
                fouled_id = contact.player1_id
                fouling_id = contact.player2_id
            elif player2_fell and not player1_fell:
                fouled_id = contact.player2_id
                fouling_id = contact.player1_id
            else:
                # Both fell - use ball proximity
                fouling_id = contact.player1_id
                fouled_id = contact.player2_id
            
            foul = FoulResult(
                is_foul=True,
                severity=severity,
                fouling_player_id=fouling_id,
                fouled_player_id=fouled_id,
                position=contact.position,
                is_penalty=is_penalty,
                confidence=confidence,
                frame_number=frame_number,
                timestamp=timestamp,
                description=self._generate_description(
                    severity, is_penalty, contact
                )
            )
            
            fouls.append(foul)
            self.foul_history.append(foul)
        
        return fouls
    
    def _calculate_foul_probability(
        self,
        contact: ContactEvent,
        player1_fell: bool,
        player2_fell: bool,
        ball_position: Optional[Tuple[float, float]]
    ) -> float:
        """Calculate probability that contact was a foul."""
        prob = 0.3  # Base probability
        
        # Increase if someone fell
        if player1_fell or player2_fell:
            prob += 0.3
        if player1_fell and player2_fell:
            prob += 0.1
        
        # Increase if ball is far
        if ball_position and contact.position:
            ball_dist = np.linalg.norm(
                np.array(ball_position) - np.array(contact.position)
            )
            if ball_dist > 3.0:
                prob += 0.2
        
        # Increase if contact duration is long
        if contact.duration > 5:
            prob += 0.1
        
        return min(0.95, prob)
    
    def _determine_severity(
        self,
        confidence: float,
        duration: int
    ) -> FoulSeverity:
        """Determine foul severity."""
        if confidence > 0.85 and duration > 8:
            return FoulSeverity.SEVERE
        elif confidence > 0.7:
            return FoulSeverity.MODERATE
        else:
            return FoulSeverity.MINOR
    
    def _generate_description(
        self,
        severity: FoulSeverity,
        is_penalty: bool,
        contact: ContactEvent
    ) -> str:
        """Generate foul description."""
        location = "in penalty area" if is_penalty else "on field"
        
        if severity == FoulSeverity.SEVERE:
            return f"Severe foul {location}"
        elif severity == FoulSeverity.MODERATE:
            return f"Moderate foul {location}"
        else:
            return f"Minor foul {location}"
    
    def reset(self):
        """Reset detector state."""
        self.fall_detector.reset()
        self.contact_detector.reset()
        self.foul_history.clear()


if __name__ == "__main__":
    # Test foul detection
    detector = FoulDetector()
    
    print("Foul detector initialized")
    print(f"Penalty area only: {detector.penalty_area_only}")
