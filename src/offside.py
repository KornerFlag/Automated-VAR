"""
Offside Detection Module

Detects offside positions based on player positions,
ball position, and pass events.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

from loguru import logger


class AttackingDirection(Enum):
    """Direction of attack."""
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = -1
    UNKNOWN = 0


@dataclass
class PassEvent:
    """A detected pass event."""
    frame_number: int
    timestamp: float
    ball_position: Tuple[float, float]
    passer_id: Optional[int]
    passer_position: Optional[Tuple[float, float]]
    velocity: float


@dataclass
class OffsideResult:
    """Result of offside analysis."""
    is_offside: bool
    offending_player_id: Optional[int]
    offending_player_position: Optional[Tuple[float, float]]
    offside_line_x: float
    margin_cm: float
    confidence: float
    frame_number: int
    timestamp: float


class PassDetector:
    """
    Detects pass events from ball trajectory.
    
    Monitors ball velocity and acceleration to identify
    when the ball has been kicked.
    """
    
    def __init__(
        self,
        velocity_threshold: float = 8.0,
        min_pass_distance: float = 5.0,
        cooldown_frames: int = 15
    ):
        """
        Initialize pass detector.
        
        Args:
            velocity_threshold: Minimum velocity change for pass (m/s)
            min_pass_distance: Minimum distance for pass (m)
            cooldown_frames: Frames to wait between passes
        """
        self.velocity_threshold = velocity_threshold
        self.min_pass_distance = min_pass_distance
        self.cooldown_frames = cooldown_frames
        
        self.ball_history: deque = deque(maxlen=30)
        self.last_pass_frame = -cooldown_frames
        self.fps = 30.0
    
    def update(
        self,
        frame_number: int,
        ball_position: Optional[Tuple[float, float]],
        player_positions: Dict[int, Tuple[float, float]] = None,
        fps: float = 30.0
    ) -> Optional[PassEvent]:
        """
        Update with new ball position and detect passes.
        
        Args:
            frame_number: Current frame number
            ball_position: Ball position in pitch coordinates
            player_positions: Dictionary of player positions
            fps: Video frame rate
        
        Returns:
            PassEvent if pass detected, None otherwise
        """
        self.fps = fps
        
        if ball_position is None:
            return None
        
        self.ball_history.append({
            'frame': frame_number,
            'position': np.array(ball_position)
        })
        
        if len(self.ball_history) < 5:
            return None
        
        # Check cooldown
        if frame_number - self.last_pass_frame < self.cooldown_frames:
            return None
        
        # Calculate velocities
        recent = list(self.ball_history)[-5:]
        velocities = []
        
        for i in range(1, len(recent)):
            dt = (recent[i]['frame'] - recent[i-1]['frame']) / fps
            if dt > 0:
                dist = np.linalg.norm(
                    recent[i]['position'] - recent[i-1]['position']
                )
                velocities.append(dist / dt)
        
        if len(velocities) < 2:
            return None
        
        # Detect sudden velocity increase
        acceleration = velocities[-1] - velocities[-2]
        
        if acceleration > self.velocity_threshold:
            self.last_pass_frame = frame_number
            
            # Find nearest player (passer)
            passer_id = None
            passer_position = None
            
            if player_positions:
                ball_pos = np.array(ball_position)
                min_dist = float('inf')
                
                for pid, pos in player_positions.items():
                    dist = np.linalg.norm(np.array(pos) - ball_pos)
                    if dist < min_dist:
                        min_dist = dist
                        passer_id = pid
                        passer_position = pos
            
            return PassEvent(
                frame_number=frame_number,
                timestamp=frame_number / fps,
                ball_position=ball_position,
                passer_id=passer_id,
                passer_position=passer_position,
                velocity=velocities[-1]
            )
        
        return None
    
    def reset(self):
        """Reset detector state."""
        self.ball_history.clear()
        self.last_pass_frame = -self.cooldown_frames


class OffsideDetector:
    """
    Detects offside positions.
    
    Analyzes player positions at the moment of a pass
    to determine if any attacking player is offside.
    """
    
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        tolerance_cm: float = 10.0
    ):
        """
        Initialize offside detector.
        
        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            tolerance_cm: Tolerance for offside detection (cm)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.half_length = pitch_length / 2
        self.tolerance_m = tolerance_cm / 100.0
        
        # Track attacking direction per team
        self.team_attacking_direction: Dict[str, AttackingDirection] = {}
        
        # Ball possession tracking
        self.ball_possession: Optional[str] = None
        
        # Results history
        self.offside_history: List[OffsideResult] = []
    
    def set_attacking_direction(
        self,
        team: str,
        direction: AttackingDirection
    ):
        """Set attacking direction for a team."""
        self.team_attacking_direction[team] = direction
    
    def infer_attacking_direction(
        self,
        team_positions: Dict[str, List[Tuple[float, float]]]
    ):
        """Infer attacking direction from team positions."""
        for team, positions in team_positions.items():
            if len(positions) < 3:
                continue
            
            avg_x = np.mean([p[0] for p in positions])
            
            # Team on left half attacks right
            if avg_x < 0:
                self.team_attacking_direction[team] = AttackingDirection.LEFT_TO_RIGHT
            else:
                self.team_attacking_direction[team] = AttackingDirection.RIGHT_TO_LEFT
    
    def analyze_pass(
        self,
        pass_event: PassEvent,
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        frame_number: int = None,
        timestamp: float = None
    ) -> Optional[OffsideResult]:
        """
        Analyze pass event for offside.
        
        Args:
            pass_event: The pass event to analyze
            player_positions: All player positions
            team_assignments: Team assignment for each player
            frame_number: Optional frame number override
            timestamp: Optional timestamp override
        
        Returns:
            OffsideResult if offside detected
        """
        if pass_event.passer_id is None:
            return None
        
        # Get passer's team
        passer_team = team_assignments.get(pass_event.passer_id)
        if not passer_team or passer_team not in self.team_attacking_direction:
            return None
        
        attacking_dir = self.team_attacking_direction[passer_team]
        if attacking_dir == AttackingDirection.UNKNOWN:
            return None
        
        # Get defending team
        defending_team = None
        for team in self.team_attacking_direction:
            if team != passer_team and team != "referee":
                defending_team = team
                break
        
        if not defending_team:
            return None
        
        # Separate players by team
        attackers = []
        defenders = []
        
        for pid, pos in player_positions.items():
            if pid == pass_event.passer_id:
                continue
            
            team = team_assignments.get(pid)
            if team == passer_team:
                attackers.append((pid, pos))
            elif team == defending_team:
                defenders.append((pid, pos))
        
        if len(defenders) < 2:
            return None
        
        # Find offside line (second-last defender)
        ball_x = pass_event.ball_position[0]
        
        if attacking_dir == AttackingDirection.LEFT_TO_RIGHT:
            # Sort defenders by x descending
            defenders.sort(key=lambda x: -x[1][0])
            goal_line_x = self.half_length
        else:
            # Sort defenders by x ascending
            defenders.sort(key=lambda x: x[1][0])
            goal_line_x = -self.half_length
        
        # Second-last defender determines offside line
        if len(defenders) >= 2:
            second_last_defender_x = defenders[1][1][0]
        else:
            second_last_defender_x = goal_line_x
        
        # Offside line is the more restrictive of ball and defender
        if attacking_dir == AttackingDirection.LEFT_TO_RIGHT:
            offside_line_x = min(second_last_defender_x, ball_x)
        else:
            offside_line_x = max(second_last_defender_x, ball_x)
        
        # Check each attacker for offside
        offside_players = []
        
        for pid, pos in attackers:
            player_x = pos[0]
            
            # Can't be offside in own half
            if attacking_dir == AttackingDirection.LEFT_TO_RIGHT:
                if player_x <= 0:
                    continue
                margin = player_x - offside_line_x
            else:
                if player_x >= 0:
                    continue
                margin = offside_line_x - player_x
            
            # Check if offside (beyond line by more than tolerance)
            if margin > self.tolerance_m:
                offside_players.append({
                    'id': pid,
                    'position': pos,
                    'margin_cm': margin * 100
                })
        
        if offside_players:
            # Return most offside player
            worst = max(offside_players, key=lambda x: x['margin_cm'])
            
            result = OffsideResult(
                is_offside=True,
                offending_player_id=worst['id'],
                offending_player_position=worst['position'],
                offside_line_x=offside_line_x,
                margin_cm=worst['margin_cm'],
                confidence=min(0.95, 0.7 + worst['margin_cm'] / 100),
                frame_number=frame_number or pass_event.frame_number,
                timestamp=timestamp or pass_event.timestamp
            )
            
            self.offside_history.append(result)
            return result
        
        return None
    
    def get_offside_line(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        attacking_team: str
    ) -> Optional[float]:
        """Get current offside line x position."""
        if attacking_team not in self.team_attacking_direction:
            return None
        
        attacking_dir = self.team_attacking_direction[attacking_team]
        
        # Find defending team
        defending_team = None
        for team in self.team_attacking_direction:
            if team != attacking_team and team != "referee":
                defending_team = team
                break
        
        if not defending_team:
            return None
        
        # Get defender positions
        defenders = [
            pos for pid, pos in player_positions.items()
            if team_assignments.get(pid) == defending_team
        ]
        
        if len(defenders) < 2:
            return None
        
        if attacking_dir == AttackingDirection.LEFT_TO_RIGHT:
            defenders.sort(key=lambda x: -x[0])
        else:
            defenders.sort(key=lambda x: x[0])
        
        return defenders[1][0]
    
    def reset(self):
        """Reset detector state."""
        self.team_attacking_direction.clear()
        self.ball_possession = None
        self.offside_history.clear()


if __name__ == "__main__":
    # Test offside detection
    detector = OffsideDetector()
    
    # Set attacking direction
    detector.set_attacking_direction("team_a", AttackingDirection.LEFT_TO_RIGHT)
    detector.set_attacking_direction("team_b", AttackingDirection.RIGHT_TO_LEFT)
    
    print("Offside detector initialized")
    print(f"Tolerance: {detector.tolerance_m * 100}cm")
