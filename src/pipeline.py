"""
Main Processing Pipeline - Enhanced VAR System

Integrates all modules to process soccer video for VAR decisions:
- OFFSIDE Detection
- FOUL Detection  
- PENALTY Detection
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import cv2
from loguru import logger
import argparse

# Import core modules
from .detection import PlayerBallDetector, Detection, draw_detections
from .tracking import MultiObjectTracker, BallTracker, TrackedObject, draw_tracks
from .team_classifier import TeamClassifier, get_team_colors
from .homography import FieldHomography, PitchDimensions

# Import VAR detection engine
try:
    from .var_detection import (
        VARDecisionEngine, VARIncident, IncidentType,
        EnhancedOffsideDetector, EnhancedFoulDetector, PassDetector,
        draw_var_overlay
    )
    VAR_ENGINE_AVAILABLE = True
except ImportError as e:
    VAR_ENGINE_AVAILABLE = False
    logger.warning(f"VAR detection engine not available: {e}")

# Optional Roboflow integration
try:
    from .roboflow_detector import RoboflowDetector
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False


@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    team_assignments: Dict[int, str]
    player_pitch_positions: Dict[int, Tuple[float, float]]
    ball_pitch_position: Optional[Tuple[float, float]]
    var_incidents: List[Any] = field(default_factory=list)
    offside_detected: bool = False
    foul_detected: bool = False
    penalty_detected: bool = False


@dataclass 
class VideoResult:
    """Results for entire video processing."""
    video_path: str
    total_frames: int
    processed_frames: int
    fps: float
    resolution: Tuple[int, int]
    offside_incidents: List[Dict]
    foul_incidents: List[Dict]
    penalty_incidents: List[Dict]
    processing_time: float
    incident_summary: Dict[str, int] = field(default_factory=dict)


class VARPipeline:
    """
    Main VAR processing pipeline.
    
    Processes soccer video footage to detect:
    - OFFSIDE positions
    - FOUL incidents
    - PENALTY decisions
    - YELLOW/RED CARDS
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        use_roboflow: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize VAR pipeline.
        
        Args:
            config: Configuration dictionary
            use_roboflow: Whether to use Roboflow for detection
            device: Device for inference (cuda/mps/cpu)
        """
        self.config = config or {}
        self.device = device
        self.use_roboflow = use_roboflow
        self.fps = 30.0
        
        logger.info("=" * 60)
        logger.info("Initializing VAR Pipeline")
        logger.info("=" * 60)
        
        # Detection
        if use_roboflow and ROBOFLOW_AVAILABLE:
            logger.info("Detection: Roboflow API")
            self.detector = RoboflowDetector(
                api_key=self.config.get('roboflow_api_key', 'QEZ7CzEaDFxxXMCWMdLn'),
                model_id=self.config.get('roboflow_model_id', 'my-first-project-mbces/1')
            )
        else:
            logger.info("Detection: Local YOLOv8")
            self.detector = PlayerBallDetector(
                model_path=self.config.get('detection_model', 'yolov8x.pt'),
                device=device,
                confidence_threshold=self.config.get('detection_confidence', 0.5)
            )
        
        # Tracking
        logger.info("Tracking: Multi-object tracker")
        self.tracker = MultiObjectTracker(
            track_thresh=self.config.get('track_thresh', 0.5),
            match_thresh=self.config.get('match_thresh', 0.8)
        )
        self.ball_tracker = BallTracker()
        
        # Team classification
        logger.info("Teams: Color-based classification")
        self.team_classifier = TeamClassifier(
            method=self.config.get('team_method', 'clustering'),
            temporal_smoothing=True
        )
        
        # Homography
        logger.info("Homography: Field mapping")
        self.homography = FieldHomography(
            pitch=PitchDimensions(
                length=self.config.get('pitch_length', 105.0),
                width=self.config.get('pitch_width', 68.0)
            )
        )
        
        # VAR Engine
        if VAR_ENGINE_AVAILABLE:
            logger.info("VAR Engine: Enhanced detection active")
            self.var_engine = VARDecisionEngine(
                pitch_length=self.config.get('pitch_length', 105.0),
                pitch_width=self.config.get('pitch_width', 68.0),
                fps=self.fps
            )
            self.use_var_engine = True
        else:
            logger.warning("VAR Engine not available")
            self.use_var_engine = False
            self._init_basic_detectors()
        
        # Results storage
        self.frame_results: List[FrameResult] = []
        self.offside_incidents: List[Dict] = []
        self.foul_incidents: List[Dict] = []
        self.penalty_incidents: List[Dict] = []
        
        logger.info("=" * 60)
        logger.info(f"VAR Pipeline Ready | Device: {device}")
        logger.info("=" * 60)
    
    def _init_basic_detectors(self):
        """Initialize basic detection when VAR engine unavailable."""
        from collections import deque
        self.ball_history = deque(maxlen=30)
        self.contact_history: Dict[Tuple[int, int], int] = {}
        self.player_heights: Dict[int, deque] = {}
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> FrameResult:
        """Process a single frame for all VAR decisions."""
        timestamp = frame_number / self.fps
        
        # Detection
        if self.use_roboflow and ROBOFLOW_AVAILABLE:
            raw_detections = self.detector.detect(frame)
            detections = [
                Detection(
                    bbox=d.bbox,
                    confidence=d.confidence,
                    class_id=0 if d.class_name.lower() in ['player', 'person'] else 32,
                    class_name='player' if d.class_name.lower() in ['player', 'person'] else 'ball'
                )
                for d in raw_detections
            ]
        else:
            detections = self.detector.detect(frame)
        
        # Tracking
        tracked_objects = self.tracker.update(detections, frame)
        
        # Ball tracking
        ball_dets = [d for d in detections if d.class_name == 'ball']
        ball_det = ball_dets[0] if ball_dets else None
        ball_position = self.ball_tracker.update(ball_det)
        
        # Team classification
        player_bboxes = []
        player_ids = []
        player_bboxes_dict = {}
        
        for obj in tracked_objects:
            if obj.class_name == 'player':
                player_bboxes.append(obj.bbox)
                player_ids.append(obj.track_id)
                player_bboxes_dict[obj.track_id] = obj.bbox
        
        team_assignments = {}
        if player_bboxes:
            team_assignments = self.team_classifier.classify(
                frame, player_bboxes, player_ids
            )
        
        # Homography
        H = self.homography.estimate(frame)
        
        player_pitch_positions = {}
        ball_pitch_position = None
        
        if H is not None:
            for obj in tracked_objects:
                if obj.class_name == 'player':
                    img_pos = obj.bottom_center
                    pitch_pos = self.homography.transform_point(img_pos[0], img_pos[1], H)
                    if pitch_pos is not None:
                        player_pitch_positions[obj.track_id] = pitch_pos
            
            if ball_position:
                ball_pitch_position = self.homography.transform_point(
                    ball_position[0], ball_position[1], H
                )
        
        # VAR Detection
        var_incidents = []
        offside_detected = False
        foul_detected = False
        penalty_detected = False
        
        if self.use_var_engine:
            var_incidents = self.var_engine.process_frame(
                frame=frame,
                frame_number=frame_number,
                player_bboxes=player_bboxes_dict,
                player_positions=player_pitch_positions,
                team_assignments=team_assignments,
                ball_position=ball_pitch_position
            )
            
            for incident in var_incidents:
                self._process_incident(incident, frame_number)
                
                if incident.incident_type == IncidentType.OFFSIDE:
                    offside_detected = True
                elif incident.incident_type == IncidentType.PENALTY:
                    penalty_detected = True
                    foul_detected = True
                elif incident.incident_type in [IncidentType.FOUL, IncidentType.YELLOW_CARD, IncidentType.RED_CARD]:
                    foul_detected = True
        
        result = FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            tracked_objects=tracked_objects,
            team_assignments=team_assignments,
            player_pitch_positions=player_pitch_positions,
            ball_pitch_position=ball_pitch_position,
            var_incidents=var_incidents,
            offside_detected=offside_detected,
            foul_detected=foul_detected,
            penalty_detected=penalty_detected
        )
        
        self.frame_results.append(result)
        return result
    
    def _process_incident(self, incident: 'VARIncident', frame_number: int):
        """Process and store a VAR incident."""
        incident_dict = incident.to_dict()
        
        if incident.incident_type == IncidentType.OFFSIDE:
            self.offside_incidents.append(incident_dict)
            logger.info(f"OFFSIDE @ frame {frame_number}: {incident.description}")
        
        elif incident.incident_type == IncidentType.PENALTY:
            self.penalty_incidents.append(incident_dict)
            self.foul_incidents.append(incident_dict)
            logger.warning(f"PENALTY @ frame {frame_number}: {incident.description}")
        
        elif incident.incident_type == IncidentType.YELLOW_CARD:
            self.foul_incidents.append(incident_dict)
            logger.warning(f"YELLOW CARD @ frame {frame_number}: {incident.description}")
        
        elif incident.incident_type == IncidentType.RED_CARD:
            self.foul_incidents.append(incident_dict)
            logger.error(f"RED CARD @ frame {frame_number}: {incident.description}")
        
        elif incident.incident_type == IncidentType.FOUL:
            self.foul_incidents.append(incident_dict)
            logger.info(f"FOUL @ frame {frame_number}: {incident.description}")
    
    def process_video(
        self,
        video_path: str,
        output_dir: str = None,
        save_video: bool = True,
        save_pitch_view: bool = True,
        max_frames: int = None,
        progress_callback=None
    ) -> VideoResult:
        """Process entire video for VAR analysis."""
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.use_var_engine:
            self.var_engine.fps = self.fps
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Processing: {video_path}")
        logger.info(f"   Resolution: {width}x{height}, FPS: {self.fps}, Frames: {total_frames}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        video_writer = None
        pitch_writer = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if save_video and output_dir:
            video_writer = cv2.VideoWriter(
                str(output_dir / "annotated_video.mp4"),
                fourcc, self.fps, (width, height)
            )
        
        if save_pitch_view and output_dir:
            pitch_size = (self.homography.template_width, self.homography.template_height)
            pitch_writer = cv2.VideoWriter(
                str(output_dir / "pitch_view.mp4"),
                fourcc, self.fps, pitch_size
            )
        
        frame_number = 0
        processed_frames = 0
        team_colors = get_team_colors()
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_number >= max_frames):
                break
            
            result = self.process_frame(frame, frame_number)
            processed_frames += 1
            
            if video_writer:
                annotated = self._draw_frame_annotations(frame, result, team_colors)
                video_writer.write(annotated)
            
            if pitch_writer and result.player_pitch_positions:
                pitch_view = self._draw_pitch_view(result, team_colors)
                if pitch_view is not None:
                    pitch_writer.write(pitch_view)
            
            if progress_callback:
                progress_callback(frame_number, total_frames)
            
            if frame_number % 100 == 0:
                logger.info(f"   Processed: {frame_number}/{total_frames} frames")
            
            frame_number += 1
        
        cap.release()
        if video_writer:
            video_writer.release()
        if pitch_writer:
            pitch_writer.release()
        
        processing_time = time.time() - start_time
        
        incident_summary = {
            'total_offsides': len(self.offside_incidents),
            'total_fouls': len(self.foul_incidents),
            'total_penalties': len(self.penalty_incidents)
        }
        
        if output_dir:
            self._save_results(output_dir, incident_summary)
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"   Time: {processing_time:.1f}s")
        logger.info(f"   Offsides: {len(self.offside_incidents)}")
        logger.info(f"   Fouls: {len(self.foul_incidents)}")
        logger.info(f"   Penalties: {len(self.penalty_incidents)}")
        logger.info("=" * 60)
        
        return VideoResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_frames,
            fps=self.fps,
            resolution=(width, height),
            offside_incidents=self.offside_incidents,
            foul_incidents=self.foul_incidents,
            penalty_incidents=self.penalty_incidents,
            processing_time=processing_time,
            incident_summary=incident_summary
        )
    
    def _draw_frame_annotations(
        self,
        frame: np.ndarray,
        result: FrameResult,
        team_colors: Dict
    ) -> np.ndarray:
        annotated = draw_tracks(
            frame, result.tracked_objects,
            result.team_assignments, team_colors
        )
        
        if self.use_var_engine and result.var_incidents:
            annotated = draw_var_overlay(annotated, result.var_incidents)
        else:
            y_offset = 50
            
            if result.offside_detected:
                cv2.rectangle(annotated, (10, y_offset - 35), (280, y_offset + 10), (0, 0, 0), -1)
                cv2.putText(annotated, "OFFSIDE", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                y_offset += 60
            
            if result.penalty_detected:
                cv2.rectangle(annotated, (10, y_offset - 35), (280, y_offset + 10), (0, 0, 0), -1)
                cv2.putText(annotated, "PENALTY", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                y_offset += 60
            
            elif result.foul_detected:
                cv2.rectangle(annotated, (10, y_offset - 35), (200, y_offset + 10), (0, 0, 0), -1)
                cv2.putText(annotated, "FOUL", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        cv2.putText(annotated, f"Frame: {result.frame_number} | Time: {result.timestamp:.2f}s",
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _draw_pitch_view(self, result: FrameResult, team_colors: Dict) -> Optional[np.ndarray]:
        try:
            return self.homography.draw_players_on_pitch(
                list(result.player_pitch_positions.values()),
                {i: result.team_assignments.get(pid, 'unknown')
                 for i, pid in enumerate(result.player_pitch_positions.keys())},
                team_colors,
                list(result.player_pitch_positions.keys()),
                result.ball_pitch_position
            )
        except Exception:
            return None
    
    def _save_results(self, output_dir: Path, incident_summary: Dict):
        with open(output_dir / "incidents.json", 'w') as f:
            json.dump({
                'summary': incident_summary,
                'offside_incidents': self.offside_incidents,
                'foul_incidents': self.foul_incidents,
                'penalty_incidents': self.penalty_incidents
            }, f, indent=2, default=str)
        
        if self.offside_incidents:
            with open(output_dir / "offsides.json", 'w') as f:
                json.dump(self.offside_incidents, f, indent=2, default=str)
        
        if self.foul_incidents:
            with open(output_dir / "fouls.json", 'w') as f:
                json.dump(self.foul_incidents, f, indent=2, default=str)
        
        if self.penalty_incidents:
            with open(output_dir / "penalties.json", 'w') as f:
                json.dump(self.penalty_incidents, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")
    
    def reset(self):
        self.tracker.reset()
        self.ball_tracker.reset()
        self.team_classifier.reset()
        if self.use_var_engine:
            self.var_engine.reset()
        self.frame_results.clear()
        self.offside_incidents.clear()
        self.foul_incidents.clear()
        self.penalty_incidents.clear()


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='VAR Analysis System - Soccer Video Analysis'
    )
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--use-roboflow', action='store_true', help='Use Roboflow API')
    parser.add_argument('--device', default='cpu', help='Device: cuda, mps, or cpu')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--no-video', action='store_true', help='Skip video output')
    parser.add_argument('--no-pitch-view', action='store_true', help='Skip pitch view')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("VAR ANALYSIS SYSTEM")
    print("=" * 60)
    
    pipeline = VARPipeline(
        use_roboflow=args.use_roboflow,
        device=args.device
    )
    
    result = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output,
        save_video=not args.no_video,
        save_pitch_view=not args.no_pitch_view,
        max_frames=args.max_frames
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Frames: {result.processed_frames}/{result.total_frames}")
    print(f"   Time: {result.processing_time:.1f}s")
    print(f"   Offsides: {len(result.offside_incidents)}")
    print(f"   Fouls: {len(result.foul_incidents)}")
    print(f"   Penalties: {len(result.penalty_incidents)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
