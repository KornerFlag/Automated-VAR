"""
Enhanced VAR Detection Module

Integrates offside, foul, and penalty detection into a unified system.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import cv2

from loguru import logger


class IncidentType(Enum):
    """Types of VAR incidents."""
    OFFSIDE = "offside"
    FOUL = "foul"
    PENALTY = "penalty"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    GOAL = "goal"
    HANDBALL = "handball"


@dataclass
class VARIncident:
    """A detected VAR incident."""
    incident_type: IncidentType
    frame_number: int
    timestamp: float
    confidence: float
    position: Tuple[float, float]
    players_involved: List[int]
    team_with_ball: Optional[str]
    description: str
    is_in_penalty_area: bool = False
    video_clip_range: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> dict:
        return {
            'type': self.incident_type.value,
            'frame': self.frame_number,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'position': self.position,
            'players': self.players_involved,
            'team_with_ball': self.team_with_ball,
            'description': self.description,
            'is_penalty_area': self.is_in_penalty_area,
            'clip_range': self.video_clip_range
        }


class EnhancedOffsideDetector:
    """Enhanced offside detection with automatic direction tracking."""
    
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        tolerance_cm: float = 15.0
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.half_length = pitch_length / 2
        self.tolerance_m = tolerance_cm / 100.0
        
        self.team_positions_history: Dict[str, deque] = {
            'team_a': deque(maxlen=100),
            'team_b': deque(maxlen=100)
        }
        
        self.attacking_directions: Dict[str, int] = {}
        self.ball_possession: Optional[str] = None
        self.last_ball_position: Optional[Tuple[float, float]] = None
    
    def update_team_positions(
        self,
        team_positions: Dict[str, List[Tuple[float, float]]]
    ):
        """Update team position history to infer attacking direction."""
        for team, positions in team_positions.items():
            if positions:
                avg_x = np.mean([p[0] for p in positions])
                self.team_positions_history[team].append(avg_x)
        
        for team in ['team_a', 'team_b']:
            if len(self.team_positions_history[team]) >= 30:
                avg_pos = np.mean(list(self.team_positions_history[team]))
                self.attacking_directions[team] = 1 if avg_pos < 0 else -1
    
    def update_ball_possession(
        self,
        ball_position: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        possession_radius: float = 3.0
    ):
        """Determine which team has ball possession."""
        if ball_position is None:
            return
        
        self.last_ball_position = ball_position
        ball_pos = np.array(ball_position)
        
        closest_player = None
        min_dist = float('inf')
        
        for player_id, pos in player_positions.items():
            dist = np.linalg.norm(np.array(pos) - ball_pos)
            if dist < min_dist:
                min_dist = dist
                closest_player = player_id
        
        if closest_player and min_dist < possession_radius:
            team = team_assignments.get(closest_player)
            if team and team in ['team_a', 'team_b']:
                self.ball_possession = team
    
    def check_offside(
        self,
        ball_position: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        passing_player_id: Optional[int] = None
    ) -> Optional[VARIncident]:
        """Check for offside at the moment of a pass."""
        if not self.ball_possession or self.ball_possession not in self.attacking_directions:
            return None
        
        attacking_team = self.ball_possession
        attacking_dir = self.attacking_directions.get(attacking_team, 1)
        defending_team = 'team_b' if attacking_team == 'team_a' else 'team_a'
        
        attackers = []
        defenders = []
        
        for player_id, pos in player_positions.items():
            team = team_assignments.get(player_id)
            if team == attacking_team and player_id != passing_player_id:
                attackers.append((player_id, pos))
            elif team == defending_team:
                defenders.append((player_id, pos))
        
        if len(defenders) < 2:
            return None
        
        if attacking_dir == 1:
            defenders.sort(key=lambda x: -x[1][0])
            goal_line_x = self.half_length
        else:
            defenders.sort(key=lambda x: x[1][0])
            goal_line_x = -self.half_length
        
        if len(defenders) >= 2:
            second_last_defender = defenders[1]
            offside_line_x = second_last_defender[1][0]
        else:
            offside_line_x = goal_line_x
        
        ball_x = ball_position[0]
        if attacking_dir == 1:
            offside_line_x = min(offside_line_x, ball_x)
        else:
            offside_line_x = max(offside_line_x, ball_x)
        
        offside_players = []
        
        for player_id, pos in attackers:
            player_x = pos[0]
            
            if attacking_dir == 1 and player_x <= 0:
                continue
            if attacking_dir == -1 and player_x >= 0:
                continue
            
            if attacking_dir == 1:
                margin = player_x - offside_line_x
                is_offside = margin > self.tolerance_m
            else:
                margin = offside_line_x - player_x
                is_offside = margin > self.tolerance_m
            
            if is_offside:
                offside_players.append({
                    'id': player_id,
                    'position': pos,
                    'margin_cm': margin * 100
                })
        
        if offside_players:
            worst = max(offside_players, key=lambda x: x['margin_cm'])
            
            return VARIncident(
                incident_type=IncidentType.OFFSIDE,
                frame_number=0,
                timestamp=0.0,
                confidence=min(0.95, 0.7 + (worst['margin_cm'] / 100)),
                position=worst['position'],
                players_involved=[worst['id']],
                team_with_ball=attacking_team,
                description=f"Player {worst['id']} is {worst['margin_cm']:.1f}cm offside",
                is_in_penalty_area=self._is_in_penalty_area(worst['position'])
            )
        
        return None
    
    def _is_in_penalty_area(self, position: Tuple[float, float]) -> bool:
        x, y = position
        penalty_length = 16.5
        penalty_width = 40.3 / 2
        
        if x <= -self.half_length + penalty_length and abs(y) <= penalty_width:
            return True
        if x >= self.half_length - penalty_length and abs(y) <= penalty_width:
            return True
        
        return False


class EnhancedFoulDetector:
    """Enhanced foul and penalty detection."""
    
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        contact_threshold: float = 1.5,
        fall_threshold: float = 0.3,
        min_contact_frames: int = 2
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.half_length = pitch_length / 2
        self.contact_threshold = contact_threshold
        self.fall_threshold = fall_threshold
        self.min_contact_frames = min_contact_frames
        
        self.player_heights: Dict[int, deque] = {}
        self.player_velocities: Dict[int, deque] = {}
        self.contact_tracking: Dict[Tuple[int, int], int] = {}
        self.recent_contacts: deque = deque(maxlen=50)
    
    def update(
        self,
        frame_number: int,
        fps: float,
        player_bboxes: Dict[int, np.ndarray],
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        ball_position: Optional[Tuple[float, float]] = None
    ) -> List[VARIncident]:
        """Process frame and detect fouls."""
        incidents = []
        timestamp = frame_number / fps if fps > 0 else 0
        
        self._update_player_tracking(player_bboxes, player_positions)
        
        contacts = self._detect_contacts(
            player_positions, team_assignments, ball_position
        )
        
        for contact in contacts:
            fall_detected = self._check_for_fall(
                contact['players'], player_bboxes
            )
            
            is_penalty_area = self._is_in_penalty_area(contact['position'])
            
            foul_confidence = self._calculate_foul_probability(
                contact, fall_detected, ball_position
            )
            
            if foul_confidence > 0.5:
                if is_penalty_area and foul_confidence > 0.6:
                    incident_type = IncidentType.PENALTY
                elif foul_confidence > 0.8:
                    incident_type = IncidentType.YELLOW_CARD
                else:
                    incident_type = IncidentType.FOUL
                
                incident = VARIncident(
                    incident_type=incident_type,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    confidence=foul_confidence,
                    position=contact['position'],
                    players_involved=contact['players'],
                    team_with_ball=contact.get('team_with_ball'),
                    description=self._generate_description(
                        incident_type, contact, fall_detected
                    ),
                    is_in_penalty_area=is_penalty_area,
                    video_clip_range=(
                        max(0, frame_number - int(fps * 2)),
                        frame_number + int(fps * 2)
                    )
                )
                incidents.append(incident)
        
        return incidents
    
    def _update_player_tracking(
        self,
        player_bboxes: Dict[int, np.ndarray],
        player_positions: Dict[int, Tuple[float, float]]
    ):
        for player_id, bbox in player_bboxes.items():
            if player_id not in self.player_heights:
                self.player_heights[player_id] = deque(maxlen=15)
            
            height = bbox[3] - bbox[1]
            self.player_heights[player_id].append(height)
            
            if player_id not in self.player_velocities:
                self.player_velocities[player_id] = deque(maxlen=10)
            
            if player_id in player_positions:
                self.player_velocities[player_id].append(player_positions[player_id])
    
    def _detect_contacts(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        ball_position: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        contacts = []
        player_ids = list(player_positions.keys())
        
        for i, p1_id in enumerate(player_ids):
            for p2_id in player_ids[i + 1:]:
                team1 = team_assignments.get(p1_id, 'unknown')
                team2 = team_assignments.get(p2_id, 'unknown')
                
                if team1 == team2 or 'unknown' in [team1, team2]:
                    continue
                if 'referee' in [team1, team2]:
                    continue
                
                pos1 = np.array(player_positions[p1_id])
                pos2 = np.array(player_positions[p2_id])
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < self.contact_threshold:
                    pair_key = (min(p1_id, p2_id), max(p1_id, p2_id))
                    self.contact_tracking[pair_key] = \
                        self.contact_tracking.get(pair_key, 0) + 1
                    
                    if self.contact_tracking[pair_key] >= self.min_contact_frames:
                        contact_point = tuple((pos1 + pos2) / 2)
                        
                        ball_dist1 = float('inf')
                        ball_dist2 = float('inf')
                        if ball_position:
                            ball_pos = np.array(ball_position)
                            ball_dist1 = np.linalg.norm(pos1 - ball_pos)
                            ball_dist2 = np.linalg.norm(pos2 - ball_pos)
                        
                        team_with_ball = team1 if ball_dist1 < ball_dist2 else team2
                        
                        contacts.append({
                            'players': [p1_id, p2_id],
                            'position': contact_point,
                            'distance': distance,
                            'duration': self.contact_tracking[pair_key],
                            'team_with_ball': team_with_ball,
                            'ball_distance': min(ball_dist1, ball_dist2)
                        })
                else:
                    pair_key = (min(p1_id, p2_id), max(p1_id, p2_id))
                    self.contact_tracking[pair_key] = 0
        
        return contacts
    
    def _check_for_fall(
        self,
        player_ids: List[int],
        player_bboxes: Dict[int, np.ndarray]
    ) -> Dict[int, bool]:
        falls = {}
        
        for player_id in player_ids:
            if player_id not in self.player_heights:
                falls[player_id] = False
                continue
            
            heights = list(self.player_heights[player_id])
            if len(heights) < 5:
                falls[player_id] = False
                continue
            
            initial_height = np.mean(heights[:3])
            current_height = heights[-1]
            
            if initial_height > 0:
                height_ratio = current_height / initial_height
                falls[player_id] = height_ratio < (1 - self.fall_threshold)
            else:
                falls[player_id] = False
        
        return falls
    
    def _calculate_foul_probability(
        self,
        contact: Dict,
        fall_detected: Dict[int, bool],
        ball_position: Optional[Tuple[float, float]]
    ) -> float:
        probability = 0.3
        
        if any(fall_detected.values()):
            probability += 0.3
        
        if ball_position and contact.get('ball_distance', 0) > 3.0:
            probability += 0.2
        
        if contact.get('duration', 0) > 5:
            probability += 0.1
        
        return min(0.95, probability)
    
    def _is_in_penalty_area(self, position: Tuple[float, float]) -> bool:
        x, y = position
        penalty_length = 16.5
        penalty_width = 40.3 / 2
        
        if x <= -self.half_length + penalty_length and abs(y) <= penalty_width:
            return True
        if x >= self.half_length - penalty_length and abs(y) <= penalty_width:
            return True
        
        return False
    
    def _generate_description(
        self,
        incident_type: IncidentType,
        contact: Dict,
        fall_detected: Dict[int, bool]
    ) -> str:
        players = contact['players']
        fell = [p for p, f in fall_detected.items() if f]
        
        if incident_type == IncidentType.PENALTY:
            desc = f"Penalty: Contact between players {players[0]} and {players[1]} in the box"
        elif incident_type == IncidentType.YELLOW_CARD:
            desc = f"Yellow card offense: players {players[0]} and {players[1]}"
        else:
            desc = f"Foul: contact between {players[0]} and {players[1]}"
        
        if fell:
            desc += f". Player(s) {fell} went down"
        
        return desc


class PassDetector:
    """Detects pass events from ball trajectory."""
    
    def __init__(
        self,
        velocity_threshold: float = 8.0,
        min_pass_distance: float = 5.0
    ):
        self.velocity_threshold = velocity_threshold
        self.min_pass_distance = min_pass_distance
        
        self.ball_history: deque = deque(maxlen=30)
        self.last_pass_frame: int = -30
    
    def update(
        self,
        frame_number: int,
        ball_position: Optional[Tuple[float, float]],
        fps: float = 30.0
    ) -> bool:
        if ball_position is None:
            return False
        
        self.ball_history.append({
            'frame': frame_number,
            'position': ball_position
        })
        
        if len(self.ball_history) < 5:
            return False
        
        if frame_number - self.last_pass_frame < 15:
            return False
        
        recent = list(self.ball_history)[-5:]
        velocities = []
        
        for i in range(1, len(recent)):
            dt = (recent[i]['frame'] - recent[i-1]['frame']) / fps
            if dt > 0:
                pos1 = np.array(recent[i-1]['position'])
                pos2 = np.array(recent[i]['position'])
                velocity = np.linalg.norm(pos2 - pos1) / dt
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return False
        
        acceleration = velocities[-1] - velocities[-2]
        
        if acceleration > self.velocity_threshold:
            self.last_pass_frame = frame_number
            return True
        
        return False
    
    def get_last_ball_position(self) -> Optional[Tuple[float, float]]:
        if self.ball_history:
            return self.ball_history[-1]['position']
        return None


class VARDecisionEngine:
    """Main VAR decision engine combining all detectors."""
    
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        fps: float = 30.0
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.fps = fps
        
        self.offside_detector = EnhancedOffsideDetector(pitch_length, pitch_width)
        self.foul_detector = EnhancedFoulDetector(pitch_length, pitch_width)
        self.pass_detector = PassDetector()
        
        self.incidents: List[VARIncident] = []
        self.frame_buffer: deque = deque(maxlen=150)
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        player_bboxes: Dict[int, np.ndarray],
        player_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, str],
        ball_position: Optional[Tuple[float, float]]
    ) -> List[VARIncident]:
        new_incidents = []
        
        self.frame_buffer.append({
            'frame_number': frame_number,
            'frame': frame.copy()
        })
        
        team_positions = {'team_a': [], 'team_b': []}
        for player_id, pos in player_positions.items():
            team = team_assignments.get(player_id)
            if team in team_positions:
                team_positions[team].append(pos)
        
        self.offside_detector.update_team_positions(team_positions)
        self.offside_detector.update_ball_possession(
            ball_position, player_positions, team_assignments
        )
        
        if ball_position:
            pass_detected = self.pass_detector.update(
                frame_number, ball_position, self.fps
            )
            
            if pass_detected:
                offside_incident = self.offside_detector.check_offside(
                    ball_position, player_positions, team_assignments
                )
                
                if offside_incident:
                    offside_incident.frame_number = frame_number
                    offside_incident.timestamp = frame_number / self.fps
                    offside_incident.video_clip_range = (
                        max(0, frame_number - int(self.fps * 2)),
                        frame_number + int(self.fps * 2)
                    )
                    new_incidents.append(offside_incident)
        
        foul_incidents = self.foul_detector.update(
            frame_number,
            self.fps,
            player_bboxes,
            player_positions,
            team_assignments,
            ball_position
        )
        new_incidents.extend(foul_incidents)
        
        self.incidents.extend(new_incidents)
        
        return new_incidents
    
    def get_all_incidents(self) -> List[VARIncident]:
        return self.incidents
    
    def get_incidents_by_type(self, incident_type: IncidentType) -> List[VARIncident]:
        return [i for i in self.incidents if i.incident_type == incident_type]
    
    def get_incident_summary(self) -> Dict[str, int]:
        summary = {}
        for incident in self.incidents:
            key = incident.incident_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary
    
    def reset(self):
        self.incidents.clear()
        self.frame_buffer.clear()
        self.pass_detector.ball_history.clear()


def draw_var_overlay(
    frame: np.ndarray,
    incidents: List[VARIncident],
    player_positions: Dict[int, Tuple[float, float]] = None,
    pitch_to_image_transform = None
) -> np.ndarray:
    """Draw VAR overlay on frame."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    for incident in incidents:
        if incident.incident_type == IncidentType.OFFSIDE:
            color = (0, 165, 255)
            text = "OFFSIDE"
        elif incident.incident_type == IncidentType.PENALTY:
            color = (0, 0, 255)
            text = "PENALTY"
        elif incident.incident_type == IncidentType.YELLOW_CARD:
            color = (0, 255, 255)
            text = "YELLOW CARD"
        elif incident.incident_type == IncidentType.RED_CARD:
            color = (0, 0, 255)
            text = "RED CARD"
        else:
            color = (255, 255, 255)
            text = "FOUL"
        
        cv2.rectangle(output, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.putText(
            output, text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
        )
        cv2.putText(
            output, f"Conf: {incident.confidence:.0%}", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
    
    return output


if __name__ == "__main__":
    print("Enhanced VAR Detection Module")
    print("=" * 50)
    print("Components:")
    print("  - EnhancedOffsideDetector")
    print("  - EnhancedFoulDetector")
    print("  - PassDetector")
    print("  - VARDecisionEngine")
