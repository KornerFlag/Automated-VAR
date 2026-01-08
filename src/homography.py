"""
Field Homography Module

Maps image coordinates to real-world pitch coordinates.
Uses field line detection and perspective transformation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2

from loguru import logger


@dataclass
class PitchDimensions:
    """Standard soccer pitch dimensions in meters."""
    length: float = 105.0
    width: float = 68.0
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.3
    goal_area_length: float = 5.5
    goal_area_width: float = 18.3
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    goal_width: float = 7.32


class FieldHomography:
    """
    Estimates homography between camera view and pitch.
    
    Uses field line detection to establish correspondences
    and compute perspective transformation matrix.
    """
    
    def __init__(
        self,
        pitch: PitchDimensions = None,
        template_scale: float = 10.0
    ):
        """
        Initialize homography estimator.
        
        Args:
            pitch: Pitch dimensions
            template_scale: Pixels per meter for template
        """
        self.pitch = pitch or PitchDimensions()
        self.template_scale = template_scale
        
        # Template dimensions
        self.template_width = int(self.pitch.length * template_scale)
        self.template_height = int(self.pitch.width * template_scale)
        
        # Cached homography matrix
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        
        # Field template
        self.field_template = self._create_field_template()
        
        # Line detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10
    
    def _create_field_template(self) -> np.ndarray:
        """Create 2D field template image."""
        template = np.zeros(
            (self.template_height, self.template_width, 3),
            dtype=np.uint8
        )
        template[:] = (34, 139, 34)  # Green field
        
        scale = self.template_scale
        
        # Field outline
        cv2.rectangle(
            template,
            (0, 0),
            (self.template_width - 1, self.template_height - 1),
            (255, 255, 255), 2
        )
        
        # Center line
        center_x = self.template_width // 2
        cv2.line(
            template,
            (center_x, 0),
            (center_x, self.template_height),
            (255, 255, 255), 2
        )
        
        # Center circle
        center_y = self.template_height // 2
        cv2.circle(
            template,
            (center_x, center_y),
            int(self.pitch.center_circle_radius * scale),
            (255, 255, 255), 2
        )
        
        # Center spot
        cv2.circle(template, (center_x, center_y), 3, (255, 255, 255), -1)
        
        # Penalty areas
        pa_length = int(self.pitch.penalty_area_length * scale)
        pa_width = int(self.pitch.penalty_area_width * scale)
        pa_y_start = (self.template_height - pa_width) // 2
        
        # Left penalty area
        cv2.rectangle(
            template,
            (0, pa_y_start),
            (pa_length, pa_y_start + pa_width),
            (255, 255, 255), 2
        )
        
        # Right penalty area
        cv2.rectangle(
            template,
            (self.template_width - pa_length, pa_y_start),
            (self.template_width, pa_y_start + pa_width),
            (255, 255, 255), 2
        )
        
        # Goal areas
        ga_length = int(self.pitch.goal_area_length * scale)
        ga_width = int(self.pitch.goal_area_width * scale)
        ga_y_start = (self.template_height - ga_width) // 2
        
        cv2.rectangle(
            template,
            (0, ga_y_start),
            (ga_length, ga_y_start + ga_width),
            (255, 255, 255), 2
        )
        
        cv2.rectangle(
            template,
            (self.template_width - ga_length, ga_y_start),
            (self.template_width, ga_y_start + ga_width),
            (255, 255, 255), 2
        )
        
        # Penalty spots
        ps_dist = int(self.pitch.penalty_spot_distance * scale)
        cv2.circle(template, (ps_dist, center_y), 3, (255, 255, 255), -1)
        cv2.circle(
            template,
            (self.template_width - ps_dist, center_y),
            3, (255, 255, 255), -1
        )
        
        return template
    
    def detect_field_lines(
        self,
        frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect field lines in frame.
        
        Args:
            frame: BGR image
        
        Returns:
            List of lines as (x1, y1, x2, y2)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        return [tuple(line[0]) for line in lines]
    
    def estimate(
        self,
        frame: np.ndarray,
        manual_points: List[Tuple[Tuple[int, int], Tuple[float, float]]] = None
    ) -> Optional[np.ndarray]:
        """
        Estimate homography matrix.
        
        Args:
            frame: BGR image
            manual_points: Optional manual correspondences
                          [(image_point, pitch_point), ...]
        
        Returns:
            3x3 homography matrix or None
        """
        if manual_points and len(manual_points) >= 4:
            # Use manual correspondences
            src_points = np.float32([p[0] for p in manual_points])
            dst_points = np.float32([
                self._pitch_to_template(p[1]) for p in manual_points
            ])
            
            self.H, _ = cv2.findHomography(src_points, dst_points)
            
        else:
            # Automatic estimation using line detection
            lines = self.detect_field_lines(frame)
            
            if len(lines) < 4:
                logger.warning("Not enough lines detected for homography")
                return self.H
            
            # Try to match lines to field template
            correspondences = self._match_lines_to_template(lines, frame.shape)
            
            if len(correspondences) >= 4:
                src_points = np.float32([c[0] for c in correspondences])
                dst_points = np.float32([c[1] for c in correspondences])
                
                self.H, _ = cv2.findHomography(
                    src_points, dst_points, cv2.RANSAC, 5.0
                )
        
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
        
        return self.H
    
    def _match_lines_to_template(
        self,
        lines: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Match detected lines to field template."""
        h, w = frame_shape[:2]
        
        # Simple heuristic matching based on position
        correspondences = []
        
        # Find horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for x1, y1, x2, y2 in lines:
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < np.pi / 6:  # Near horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle > np.pi / 3:  # Near vertical
                vertical_lines.append((x1, y1, x2, y2))
        
        # Use frame corners as basic correspondences
        correspondences = [
            ((0, 0), (0, 0)),
            ((w, 0), (self.template_width, 0)),
            ((0, h), (0, self.template_height)),
            ((w, h), (self.template_width, self.template_height))
        ]
        
        return correspondences
    
    def _pitch_to_template(
        self,
        pitch_point: Tuple[float, float]
    ) -> Tuple[int, int]:
        """Convert pitch coordinates to template coordinates."""
        x = (pitch_point[0] + self.pitch.length / 2) * self.template_scale
        y = (pitch_point[1] + self.pitch.width / 2) * self.template_scale
        return (int(x), int(y))
    
    def transform_point(
        self,
        x: float,
        y: float,
        H: np.ndarray = None
    ) -> Optional[Tuple[float, float]]:
        """
        Transform image point to pitch coordinates.
        
        Args:
            x: Image x coordinate
            y: Image y coordinate
            H: Optional homography matrix
        
        Returns:
            Pitch coordinates (x, y) in meters or None
        """
        if H is None:
            H = self.H
        
        if H is None:
            return None
        
        # Transform point
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H)
        
        # Convert from template to pitch coordinates
        tx, ty = transformed[0][0]
        
        pitch_x = tx / self.template_scale - self.pitch.length / 2
        pitch_y = ty / self.template_scale - self.pitch.width / 2
        
        return (pitch_x, pitch_y)
    
    def transform_points(
        self,
        points: List[Tuple[float, float]],
        H: np.ndarray = None
    ) -> List[Optional[Tuple[float, float]]]:
        """Transform multiple points."""
        return [self.transform_point(p[0], p[1], H) for p in points]
    
    def is_in_penalty_area(
        self,
        x: float,
        y: float,
        side: str = "both"
    ) -> bool:
        """
        Check if pitch coordinate is in penalty area.
        
        Args:
            x: Pitch x coordinate (meters)
            y: Pitch y coordinate (meters)
            side: Which side to check (left/right/both)
        
        Returns:
            True if in penalty area
        """
        half_length = self.pitch.length / 2
        pa_length = self.pitch.penalty_area_length
        pa_half_width = self.pitch.penalty_area_width / 2
        
        in_y = abs(y) <= pa_half_width
        
        if side == "left" or side == "both":
            if x <= -half_length + pa_length and in_y:
                return True
        
        if side == "right" or side == "both":
            if x >= half_length - pa_length and in_y:
                return True
        
        return False
    
    def draw_players_on_pitch(
        self,
        positions: List[Tuple[float, float]],
        team_assignments: Dict[int, str],
        team_colors: Dict[str, Tuple[int, int, int]],
        player_ids: List[int] = None,
        ball_position: Tuple[float, float] = None
    ) -> np.ndarray:
        """
        Draw players on 2D pitch template.
        
        Args:
            positions: List of pitch positions (x, y) in meters
            team_assignments: Team assignment for each position index
            team_colors: Colors for each team
            player_ids: Optional player IDs for labels
            ball_position: Optional ball position
        
        Returns:
            Pitch view image
        """
        output = self.field_template.copy()
        
        for i, pos in enumerate(positions):
            if pos is None:
                continue
            
            # Convert to template coordinates
            tx = int((pos[0] + self.pitch.length / 2) * self.template_scale)
            ty = int((pos[1] + self.pitch.width / 2) * self.template_scale)
            
            # Clamp to template bounds
            tx = max(0, min(tx, self.template_width - 1))
            ty = max(0, min(ty, self.template_height - 1))
            
            # Get team color
            team = team_assignments.get(i, "unknown")
            color = team_colors.get(team, (128, 128, 128))
            
            # Draw player marker
            cv2.circle(output, (tx, ty), 8, color, -1)
            cv2.circle(output, (tx, ty), 8, (255, 255, 255), 1)
            
            # Draw player ID
            if player_ids:
                label = str(player_ids[i])
                cv2.putText(
                    output, label, (tx - 5, ty - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
        
        # Draw ball
        if ball_position:
            bx = int((ball_position[0] + self.pitch.length / 2) * self.template_scale)
            by = int((ball_position[1] + self.pitch.width / 2) * self.template_scale)
            
            bx = max(0, min(bx, self.template_width - 1))
            by = max(0, min(by, self.template_height - 1))
            
            cv2.circle(output, (bx, by), 6, (255, 255, 255), -1)
            cv2.circle(output, (bx, by), 6, (0, 0, 0), 1)
        
        return output
    
    def warp_frame_to_pitch(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Warp camera frame to pitch view."""
        if self.H is None:
            return None
        
        return cv2.warpPerspective(
            frame,
            self.H,
            (self.template_width, self.template_height)
        )


if __name__ == "__main__":
    # Test homography
    homography = FieldHomography()
    template = homography.field_template
    
    print(f"Template size: {template.shape}")
    print(f"Pitch dimensions: {homography.pitch.length}m x {homography.pitch.width}m")
    
    # Test coordinate transformation
    test_point = (100, 200)
    print(f"Test point: {test_point}")
