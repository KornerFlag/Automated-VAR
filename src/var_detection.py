"""
Enhanced VAR Detection Module

Integrates offside, foul, and penalty detection into a unified system.
Includes VAR Declaration System for official decision announcements.
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


class DecisionType(Enum):
    """Official VAR decision types."""
    OFFSIDE_CONFIRMED = "Offside - Attack Nullified"
    OFFSIDE_BORDERLINE = "Offside - Marginal Call"
    NO_OFFSIDE = "No Offside - Play On"
    PENALTY_AWARDED = "Penalty Kick Awarded"
    PENALTY_REVIEW = "Penalty - Review Recommended"
    NO_PENALTY = "No Penalty - Play On"
    FOUL_CONFIRMED = "Foul Confirmed"
    FOUL_REVIEW = "Foul - Review Recommended"
    NO_FOUL = "No Foul - Play On"
    YELLOW_CARD_RECOMMENDED = "Yellow Card Recommended"
    RED_CARD_RECOMMENDED = "Red Card Recommended"
    GOAL_VALID = "Goal Confirmed"
    GOAL_DISALLOWED = "Goal Disallowed"


class ConfidenceLevel(Enum):
    """Confidence level categories for decisions."""
    VERY_HIGH = ("Very High", 0.90, "#22c55e")  # Green
    HIGH = ("High", 0.75, "#84cc16")  # Lime
    MEDIUM = ("Medium", 0.60, "#eab308")  # Yellow
    LOW = ("Low", 0.45, "#f97316")  # Orange
    VERY_LOW = ("Very Low", 0.0, "#ef4444")  # Red

    @classmethod
    def from_confidence(cls, confidence: float) -> 'ConfidenceLevel':
        """Get confidence level from numeric value."""
        if confidence >= 0.90:
            return cls.VERY_HIGH
        elif confidence >= 0.75:
            return cls.HIGH
        elif confidence >= 0.60:
            return cls.MEDIUM
        elif confidence >= 0.45:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class PositionAnalysis:
    """Detailed position analysis for a decision."""
    player_id: int
    player_position: Tuple[float, float]
    reference_line: Optional[float] = None  # e.g., offside line x-coordinate
    distance_to_reference: Optional[float] = None  # in meters
    zone: str = ""  # e.g., "attacking half", "penalty area"
    relative_to_ball: str = ""  # e.g., "ahead of ball", "behind ball"

    def to_dict(self) -> dict:
        return {
            'player_id': self.player_id,
            'position': self.player_position,
            'reference_line': self.reference_line,
            'distance_to_reference_m': self.distance_to_reference,
            'zone': self.zone,
            'relative_to_ball': self.relative_to_ball
        }


@dataclass
class VARDeclaration:
    """
    Official VAR declaration with decision, confidence, and rationale.
    This represents what the VAR would announce to the referee.
    """
    decision: DecisionType
    confidence: float
    confidence_level: ConfidenceLevel
    rationale: str
    short_summary: str  # Brief one-liner for overlay
    position_analysis: List[PositionAnalysis]
    factors: Dict[str, Any]  # Breakdown of what contributed to decision
    recommendation: str  # What should happen next
    review_required: bool  # Whether human review is recommended

    def to_dict(self) -> dict:
        return {
            'decision': self.decision.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value[0],
            'confidence_color': self.confidence_level.value[2],
            'rationale': self.rationale,
            'short_summary': self.short_summary,
            'position_analysis': [p.to_dict() for p in self.position_analysis],
            'factors': self.factors,
            'recommendation': self.recommendation,
            'review_required': self.review_required
        }

    def get_display_confidence(self) -> str:
        """Get formatted confidence for display."""
        return f"{self.confidence:.0%} ({self.confidence_level.value[0]})"

    def get_color(self) -> str:
        """Get color code for this confidence level."""
        return self.confidence_level.value[2]


@dataclass
class VARIncident:
    """A detected VAR incident with official declaration."""
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
    declaration: Optional[VARDeclaration] = None

    def to_dict(self) -> dict:
        result = {
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
        if self.declaration:
            result['declaration'] = self.declaration.to_dict()
        return result


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
            margin_cm = worst['margin_cm']
            confidence = min(0.95, 0.7 + (margin_cm / 100))

            # Generate VAR Declaration
            declaration = self._generate_offside_declaration(
                player_id=worst['id'],
                player_position=worst['position'],
                margin_cm=margin_cm,
                offside_line_x=offside_line_x,
                ball_position=ball_position,
                confidence=confidence,
                second_last_defender_id=second_last_defender[0] if len(defenders) >= 2 else None
            )

            return VARIncident(
                incident_type=IncidentType.OFFSIDE,
                frame_number=0,
                timestamp=0.0,
                confidence=confidence,
                position=worst['position'],
                players_involved=[worst['id']],
                team_with_ball=attacking_team,
                description=f"Player {worst['id']} is {margin_cm:.1f}cm offside",
                is_in_penalty_area=self._is_in_penalty_area(worst['position']),
                declaration=declaration
            )

        return None

    def _generate_offside_declaration(
        self,
        player_id: int,
        player_position: Tuple[float, float],
        margin_cm: float,
        offside_line_x: float,
        ball_position: Tuple[float, float],
        confidence: float,
        second_last_defender_id: Optional[int]
    ) -> VARDeclaration:
        """Generate official VAR declaration for offside."""
        confidence_level = ConfidenceLevel.from_confidence(confidence)

        # Determine decision type based on margin
        if margin_cm > 50:  # Clear offside (> 50cm)
            decision = DecisionType.OFFSIDE_CONFIRMED
            review_required = False
        elif margin_cm > 15:  # Offside but closer
            decision = DecisionType.OFFSIDE_CONFIRMED
            review_required = confidence < 0.85
        else:  # Marginal offside (< 15cm)
            decision = DecisionType.OFFSIDE_BORDERLINE
            review_required = True

        # Build position analysis
        zone = "attacking half" if abs(player_position[0]) > self.half_length / 2 else "midfield"
        if self._is_in_penalty_area(player_position):
            zone = "penalty area"

        position_analysis = [
            PositionAnalysis(
                player_id=player_id,
                player_position=player_position,
                reference_line=offside_line_x,
                distance_to_reference=margin_cm / 100,
                zone=zone,
                relative_to_ball="ahead of ball" if player_position[0] > ball_position[0] else "behind ball"
            )
        ]

        # Build factors dictionary
        factors = {
            'offside_margin_cm': round(margin_cm, 1),
            'tolerance_applied_cm': self.tolerance_m * 100,
            'offside_line_position': round(offside_line_x, 2),
            'ball_position_x': round(ball_position[0], 2),
            'player_position_x': round(player_position[0], 2),
            'second_last_defender': second_last_defender_id,
            'margin_category': 'clear' if margin_cm > 50 else 'moderate' if margin_cm > 15 else 'marginal'
        }

        # Generate rationale
        if margin_cm > 50:
            rationale = (
                f"Player {player_id} is clearly in an offside position, {margin_cm:.1f}cm beyond the "
                f"second-last defender (Player {second_last_defender_id}). The attacking player was ahead of "
                f"the offside line at the moment the ball was played. This is a clear offside decision."
            )
            short_summary = f"OFFSIDE: Player {player_id} - {margin_cm:.0f}cm beyond line"
            recommendation = "Disallow the attack. Free kick to defending team."
        elif margin_cm > 15:
            rationale = (
                f"Player {player_id} is in an offside position, {margin_cm:.1f}cm beyond the offside line "
                f"set by defender {second_last_defender_id}. While not a marginal call, the position is "
                f"clearly offside after accounting for the {self.tolerance_m * 100:.0f}cm tolerance."
            )
            short_summary = f"OFFSIDE: Player {player_id} - {margin_cm:.0f}cm"
            recommendation = "Disallow the attack. Free kick to defending team."
        else:
            rationale = (
                f"Player {player_id} appears to be marginally offside by approximately {margin_cm:.1f}cm. "
                f"This is within the borderline zone and the decision accounts for the {self.tolerance_m * 100:.0f}cm "
                f"tolerance. Human review recommended due to the tight margin."
            )
            short_summary = f"OFFSIDE (marginal): Player {player_id} - {margin_cm:.0f}cm"
            recommendation = "Review recommended. Marginal offside detected."

        return VARDeclaration(
            decision=decision,
            confidence=confidence,
            confidence_level=confidence_level,
            rationale=rationale,
            short_summary=short_summary,
            position_analysis=position_analysis,
            factors=factors,
            recommendation=recommendation,
            review_required=review_required
        )
    
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

                # Generate VAR Declaration for foul/penalty
                declaration = self._generate_foul_declaration(
                    incident_type=incident_type,
                    contact=contact,
                    fall_detected=fall_detected,
                    is_penalty_area=is_penalty_area,
                    confidence=foul_confidence,
                    ball_position=ball_position,
                    player_positions=player_positions
                )

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
                    ),
                    declaration=declaration
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

    def _generate_foul_declaration(
        self,
        incident_type: IncidentType,
        contact: Dict,
        fall_detected: Dict[int, bool],
        is_penalty_area: bool,
        confidence: float,
        ball_position: Optional[Tuple[float, float]],
        player_positions: Dict[int, Tuple[float, float]]
    ) -> VARDeclaration:
        """Generate official VAR declaration for foul/penalty incidents."""
        confidence_level = ConfidenceLevel.from_confidence(confidence)
        players = contact['players']
        fell = [p for p, f in fall_detected.items() if f]
        contact_pos = contact['position']

        # Determine decision type
        if incident_type == IncidentType.PENALTY:
            if confidence >= 0.85:
                decision = DecisionType.PENALTY_AWARDED
                review_required = False
            else:
                decision = DecisionType.PENALTY_REVIEW
                review_required = True
        elif incident_type == IncidentType.YELLOW_CARD:
            decision = DecisionType.YELLOW_CARD_RECOMMENDED
            review_required = confidence < 0.80
        elif incident_type == IncidentType.RED_CARD:
            decision = DecisionType.RED_CARD_RECOMMENDED
            review_required = True
        else:
            if confidence >= 0.75:
                decision = DecisionType.FOUL_CONFIRMED
                review_required = False
            else:
                decision = DecisionType.FOUL_REVIEW
                review_required = True

        # Build position analysis
        position_analysis = []
        for player_id in players:
            if player_id in player_positions:
                pos = player_positions[player_id]
                zone = "penalty area" if is_penalty_area else "midfield"
                if abs(pos[0]) > self.half_length * 0.75:
                    zone = "attacking third" if not is_penalty_area else zone

                relative_ball = "near ball"
                if ball_position:
                    dist = np.linalg.norm(np.array(pos) - np.array(ball_position))
                    if dist > 5.0:
                        relative_ball = "away from ball"
                    elif dist > 2.0:
                        relative_ball = "close to ball"

                position_analysis.append(PositionAnalysis(
                    player_id=player_id,
                    player_position=pos,
                    zone=zone,
                    relative_to_ball=relative_ball
                ))

        # Build factors dictionary
        factors = {
            'contact_detected': True,
            'contact_distance_m': round(contact.get('distance', 0), 2),
            'contact_duration_frames': contact.get('duration', 0),
            'fall_detected': bool(fell),
            'players_fell': fell,
            'ball_distance_m': round(contact.get('ball_distance', 0), 2) if contact.get('ball_distance') else None,
            'is_penalty_area': is_penalty_area,
            'contact_position': (round(contact_pos[0], 2), round(contact_pos[1], 2)),
            'foul_severity': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
        }

        # Generate rationale based on incident type
        if incident_type == IncidentType.PENALTY:
            rationale = (
                f"Contact detected between Player {players[0]} and Player {players[1]} inside the penalty area. "
            )
            if fell:
                rationale += f"Player {fell[0]} went to ground after the challenge. "
            ball_dist = contact.get('ball_distance', 0)
            if ball_dist and ball_dist > 3.0:
                rationale += f"The ball was {ball_dist:.1f}m away from the contact point, suggesting the foul was not a genuine attempt to play the ball. "
            else:
                rationale += f"The challenge was made in proximity to the ball ({ball_dist:.1f}m). "
            rationale += f"Confidence level: {confidence:.0%}."

            short_summary = f"PENALTY: Contact in box - Players {players[0]} & {players[1]}"
            recommendation = "Award penalty kick. Consider disciplinary action."

        elif incident_type == IncidentType.YELLOW_CARD:
            rationale = (
                f"Significant foul by Player {players[0]} on Player {players[1]}. "
                f"The nature of the challenge warrants a caution. "
            )
            if fell:
                rationale += f"Player {fell[0]} was brought down by the challenge. "
            rationale += f"Contact duration: {contact.get('duration', 0)} frames. Confidence: {confidence:.0%}."

            short_summary = f"YELLOW CARD: Reckless foul - Player {players[0]}"
            recommendation = "Issue yellow card. Award free kick to fouled team."

        else:  # Regular foul
            rationale = (
                f"Foul detected between Player {players[0]} and Player {players[1]}. "
            )
            if fell:
                rationale += f"Player {fell[0]} went to ground. "
            rationale += (
                f"Contact occurred at position ({contact_pos[0]:.1f}m, {contact_pos[1]:.1f}m) on the pitch. "
                f"Confidence level: {confidence:.0%}."
            )

            short_summary = f"FOUL: Contact between Players {players[0]} & {players[1]}"
            recommendation = "Award free kick to the fouled team."

        return VARDeclaration(
            decision=decision,
            confidence=confidence,
            confidence_level=confidence_level,
            rationale=rationale,
            short_summary=short_summary,
            position_analysis=position_analysis,
            factors=factors,
            recommendation=recommendation,
            review_required=review_required
        )


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
    pitch_to_image_transform = None,
    show_declaration: bool = True
) -> np.ndarray:
    """Draw VAR overlay on frame with declaration panel."""
    output = frame.copy()
    h, w = frame.shape[:2]

    for incident in incidents:
        if incident.incident_type == IncidentType.OFFSIDE:
            color = (0, 165, 255)  # Orange
            text = "OFFSIDE"
        elif incident.incident_type == IncidentType.PENALTY:
            color = (0, 0, 255)  # Red
            text = "PENALTY"
        elif incident.incident_type == IncidentType.YELLOW_CARD:
            color = (0, 255, 255)  # Yellow
            text = "YELLOW CARD"
        elif incident.incident_type == IncidentType.RED_CARD:
            color = (0, 0, 255)  # Red
            text = "RED CARD"
        else:
            color = (255, 255, 255)  # White
            text = "FOUL"

        # Draw main decision banner
        banner_height = 90 if show_declaration and incident.declaration else 80
        cv2.rectangle(output, (10, 10), (450, banner_height), (0, 0, 0), -1)
        cv2.rectangle(output, (10, 10), (450, banner_height), color, 2)

        # VAR badge
        cv2.rectangle(output, (15, 15), (70, 45), color, -1)
        cv2.putText(output, "VAR", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Decision text
        cv2.putText(output, text, (80, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        # Confidence bar
        conf_bar_width = int(200 * incident.confidence)
        cv2.rectangle(output, (80, 55), (280, 70), (50, 50, 50), -1)
        cv2.rectangle(output, (80, 55), (80 + conf_bar_width, 70), color, -1)
        cv2.putText(output, f"{incident.confidence:.0%}", (290, 67),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show declaration details if available
        if show_declaration and incident.declaration:
            decl = incident.declaration

            # Draw declaration panel
            panel_y = banner_height + 20
            panel_height = 180
            cv2.rectangle(output, (10, panel_y), (500, panel_y + panel_height), (20, 20, 20), -1)
            cv2.rectangle(output, (10, panel_y), (500, panel_y + panel_height), (60, 60, 60), 1)

            # Declaration header
            cv2.putText(output, "VAR DECLARATION", (20, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            # Decision type
            cv2.putText(output, decl.decision.value, (20, panel_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Confidence level with color
            conf_color = _hex_to_bgr(decl.get_color())
            cv2.putText(output, f"Confidence: {decl.get_display_confidence()}",
                       (20, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)

            # Short summary
            cv2.putText(output, decl.short_summary, (20, panel_y + 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Recommendation
            cv2.putText(output, "Recommendation:", (20, panel_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            cv2.putText(output, decl.recommendation[:60], (20, panel_y + 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Review indicator
            if decl.review_required:
                cv2.rectangle(output, (350, panel_y + 15), (490, panel_y + 40), (0, 100, 255), -1)
                cv2.putText(output, "REVIEW REQ.", (360, panel_y + 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Position factors (if offside, show margin)
            if incident.incident_type == IncidentType.OFFSIDE and decl.factors:
                margin = decl.factors.get('offside_margin_cm', 0)
                cv2.putText(output, f"Margin: {margin:.1f}cm", (350, panel_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    return output


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def draw_declaration_panel(
    frame: np.ndarray,
    declaration: VARDeclaration,
    position: Tuple[int, int] = (10, 100)
) -> np.ndarray:
    """Draw a detailed declaration panel on frame."""
    output = frame.copy()
    x, y = position

    # Panel background
    panel_width = 520
    panel_height = 250
    cv2.rectangle(output, (x, y), (x + panel_width, y + panel_height), (15, 15, 15), -1)
    cv2.rectangle(output, (x, y), (x + panel_width, y + panel_height), (80, 80, 80), 2)

    # Header
    cv2.rectangle(output, (x, y), (x + panel_width, y + 40), (30, 30, 30), -1)
    cv2.putText(output, "VAR OFFICIAL DECLARATION", (x + 15, y + 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Decision with colored background
    conf_color = _hex_to_bgr(declaration.get_color())
    cv2.rectangle(output, (x + 10, y + 50), (x + panel_width - 10, y + 85), conf_color, -1)
    cv2.putText(output, declaration.decision.value, (x + 20, y + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Confidence
    cv2.putText(output, f"Confidence: {declaration.get_display_confidence()}",
               (x + 15, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 1)

    # Summary
    cv2.putText(output, declaration.short_summary[:65], (x + 15, y + 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Factors summary
    y_offset = y + 160
    cv2.putText(output, "Key Factors:", (x + 15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    factors_text = []
    for key, val in list(declaration.factors.items())[:4]:
        if val is not None and val != "":
            factors_text.append(f"{key}: {val}")

    for i, factor in enumerate(factors_text[:3]):
        cv2.putText(output, f"• {factor[:40]}", (x + 20, y_offset + 18 + i * 16),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Recommendation
    cv2.rectangle(output, (x + 10, y + panel_height - 45), (x + panel_width - 10, y + panel_height - 10),
                 (40, 40, 40), -1)
    cv2.putText(output, f">> {declaration.recommendation[:55]}", (x + 20, y + panel_height - 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)

    # Review required badge
    if declaration.review_required:
        cv2.rectangle(output, (x + panel_width - 120, y + 5), (x + panel_width - 5, y + 35),
                     (0, 0, 200), -1)
        cv2.putText(output, "REVIEW", (x + panel_width - 110, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output


if __name__ == "__main__":
    print("Enhanced VAR Detection Module")
    print("=" * 50)
    print("Components:")
    print("  - EnhancedOffsideDetector")
    print("  - EnhancedFoulDetector")
    print("  - PassDetector")
    print("  - VARDecisionEngine")
    print("  - VARDeclaration (Declaration System)")
    print("")
    print("Declaration Types:")
    for dt in DecisionType:
        print(f"  - {dt.value}")
    print("")
    print("Confidence Levels:")
    for cl in ConfidenceLevel:
        print(f"  - {cl.value[0]}: >= {cl.value[1]:.0%} ({cl.value[2]})")
