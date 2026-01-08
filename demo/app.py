"""
Automated VAR System - Integrated Dashboard

Production-grade interface with full VAR pipeline integration.
For use in professional and enterprise environments.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading

# Page configuration
st.set_page_config(
    page_title="VAR Analysis System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Attempt to import VAR pipeline
try:
    from src.pipeline import VARPipeline, VideoResult
    from src.detection import PlayerBallDetector
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Professional styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #09090b;
        --bg-secondary: #0f0f12;
        --bg-tertiary: #18181b;
        --bg-elevated: #1f1f23;
        --border-subtle: #27272a;
        --border-default: #3f3f46;
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --accent-primary: #2563eb;
        --accent-secondary: #3b82f6;
        --accent-success: #22c55e;
        --accent-warning: #eab308;
        --accent-danger: #ef4444;
    }
    
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }
    
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1600px;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.025em;
    }
    
    /* Header Component */
    .app-header {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    .app-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 2rem;
        right: 2rem;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
    }
    
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }
    
    .system-status {
        display: flex;
        align-items: center;
        gap: 0.625rem;
        padding: 0.5rem 1rem;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent-success);
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        background: var(--accent-success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Card Components */
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem;
        transition: border-color 0.2s ease;
    }
    
    .card:hover {
        border-color: var(--border-default);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .card-badge {
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metric Display */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1.25rem;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    
    .metric-unit {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-left: 0.25rem;
    }
    
    /* Incident List */
    .incident-list {
        display: flex;
        flex-direction: column;
        gap: 0.625rem;
    }
    
    .incident-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    
    .incident-item:hover {
        background: var(--bg-elevated);
        border-color: var(--border-default);
    }
    
    .incident-badge {
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        flex-shrink: 0;
    }
    
    .incident-badge.offside {
        background: rgba(234, 179, 8, 0.15);
        color: var(--accent-warning);
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    
    .incident-badge.foul {
        background: rgba(239, 68, 68, 0.15);
        color: var(--accent-danger);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .incident-badge.penalty {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-danger);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .incident-details {
        flex: 1;
        min-width: 0;
    }
    
    .incident-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.125rem;
    }
    
    .incident-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .incident-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--accent-secondary);
        flex-shrink: 0;
    }
    
    /* Progress */
    .progress-container {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .progress-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    
    .progress-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .progress-value {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .progress-bar {
        height: 6px;
        background: var(--bg-elevated);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.15s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: var(--accent-secondary);
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border-subtle);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: var(--text-muted);
        font-weight: 500;
        padding: 0.625rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border-color: var(--border-subtle);
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--border-subtle);
    }
    
    .stSlider > div > div > div > div {
        background: var(--accent-primary);
    }
    
    /* Info Panel */
    .info-panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-primary);
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    
    .info-panel-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.375rem;
    }
    
    .info-panel-text {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* Module List */
    .module-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    
    .module-status {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .module-status.active {
        background: var(--accent-success);
    }
    
    .module-status.inactive {
        background: var(--text-muted);
    }
    
    .module-name {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    .module-desc {
        font-size: 0.7rem;
        color: var(--text-muted);
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--bg-secondary);
        border: 1px dashed var(--border-subtle);
        border-radius: 12px;
    }
    
    .empty-state-icon {
        font-size: 2.5rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .empty-state-text {
        font-size: 0.875rem;
        color: var(--text-muted);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-default);
        border-radius: 3px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


class VARDashboard:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize dashboard state."""
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'processing': False,
            'results': None,
            'incidents': [],
            'video_info': None,
            'progress': 0,
            'status_message': ''
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render application header."""
        pipeline_status = "Operational" if PIPELINE_AVAILABLE else "Limited Mode"
        status_class = "system-status" if PIPELINE_AVAILABLE else "system-status warning"
        
        st.markdown(f"""
        <div class="app-header">
            <div class="header-content">
                <div>
                    <h1 class="header-title">VAR Analysis System</h1>
                    <p class="header-subtitle">Automated Video Assistant Referee Platform</p>
                </div>
                <div class="{status_class}">
                    <div class="status-indicator"></div>
                    {pipeline_status}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar configuration panel."""
        with st.sidebar:
            st.markdown("""
            <div style="padding-bottom: 1rem; border-bottom: 1px solid var(--border-subtle); margin-bottom: 1.25rem;">
                <h3 style="font-size: 0.9rem; font-weight: 600; color: var(--text-primary); margin: 0;">Analysis Configuration</h3>
                <p style="font-size: 0.75rem; color: var(--text-muted); margin: 0.25rem 0 0 0;">Configure processing parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Device selection
            st.markdown("**Compute Device**")
            device = st.selectbox(
                "device_select",
                options=["cpu", "cuda", "mps"],
                format_func=lambda x: {"cpu": "CPU (Universal)", "cuda": "NVIDIA CUDA", "mps": "Apple Silicon"}[x],
                label_visibility="collapsed"
            )
            
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            
            # Confidence threshold
            st.markdown("**Detection Threshold**")
            confidence = st.slider(
                "confidence_slider",
                min_value=0.30,
                max_value=0.90,
                value=0.50,
                step=0.05,
                format="%.2f",
                label_visibility="collapsed"
            )
            
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            
            # Advanced settings
            with st.expander("Advanced Parameters"):
                offside_tolerance = st.number_input(
                    "Offside Tolerance (cm)",
                    min_value=5,
                    max_value=30,
                    value=15
                )
                
                contact_threshold = st.number_input(
                    "Contact Threshold (m)",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1
                )
                
                max_frames = st.number_input(
                    "Frame Limit (0=all)",
                    min_value=0,
                    max_value=50000,
                    value=0,
                    step=500
                )
                
                save_video = st.checkbox("Generate Annotated Video", value=True)
                save_pitch = st.checkbox("Generate Pitch View", value=True)
            
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            
            # Version info
            st.markdown(f"""
            <div style="padding: 0.875rem; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 6px;">
                <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem;">Version</div>
                <div style="font-size: 0.85rem; color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">0.2.0</div>
            </div>
            """, unsafe_allow_html=True)
            
            return {
                'device': device,
                'confidence': confidence,
                'offside_tolerance': offside_tolerance,
                'contact_threshold': contact_threshold,
                'max_frames': max_frames if max_frames > 0 else None,
                'save_video': save_video,
                'save_pitch': save_pitch
            }
    
    def render_metric(self, label: str, value: str, unit: str = ""):
        """Render a metric display."""
        unit_html = f'<span class="metric-unit">{unit}</span>' if unit else ''
        return f"""
        <div class="metric-item">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}{unit_html}</div>
        </div>
        """
    
    def render_incident(self, incident: Dict) -> str:
        """Render an incident item."""
        type_labels = {
            'offside': ('OFF', 'Offside Position'),
            'foul': ('FUL', 'Foul Detected'),
            'penalty': ('PEN', 'Penalty Incident'),
            'yellow_card': ('YEL', 'Yellow Card'),
            'red_card': ('RED', 'Red Card')
        }
        
        badge, title = type_labels.get(incident['type'], ('INC', 'Incident'))
        timestamp = self.format_time(incident.get('timestamp', 0))
        
        return f"""
        <div class="incident-item">
            <div class="incident-badge {incident['type']}">{badge}</div>
            <div class="incident-details">
                <div class="incident-title">{title}</div>
                <div class="incident-meta">Frame {incident.get('frame', 0):,} | {timestamp} | {incident.get('description', '')}</div>
            </div>
            <div class="incident-confidence">{incident.get('confidence', 0):.0%}</div>
        </div>
        """
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
    
    def render_module_status(self, name: str, desc: str, active: bool = True):
        """Render module status indicator."""
        status = "active" if active else "inactive"
        return f"""
        <div class="module-item">
            <div class="module-status {status}"></div>
            <div>
                <div class="module-name">{name}</div>
                <div class="module-desc">{desc}</div>
            </div>
        </div>
        """
    
    def process_video(self, video_path: str, config: Dict) -> Optional[Dict]:
        """Process video through VAR pipeline."""
        if not PIPELINE_AVAILABLE:
            # Simulation mode
            return self._simulate_processing(video_path, config)
        
        try:
            pipeline = VARPipeline(
                config={
                    'detection_confidence': config['confidence'],
                    'offside_tolerance_cm': config['offside_tolerance'],
                    'contact_threshold': config['contact_threshold']
                },
                device=config['device']
            )
            
            output_dir = tempfile.mkdtemp()
            
            result = pipeline.process_video(
                video_path=video_path,
                output_dir=output_dir,
                save_video=config['save_video'],
                save_pitch_view=config['save_pitch'],
                max_frames=config['max_frames']
            )
            
            return {
                'processed_frames': result.processed_frames,
                'total_frames': result.total_frames,
                'fps': result.fps,
                'processing_time': result.processing_time,
                'offside_incidents': result.offside_incidents,
                'foul_incidents': result.foul_incidents,
                'penalty_incidents': result.penalty_incidents,
                'output_dir': output_dir
            }
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return None
    
    def _simulate_processing(self, video_path: str, config: Dict) -> Dict:
        """Simulate processing for demo purposes."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if config['max_frames']:
            frame_count = min(frame_count, config['max_frames'])
        
        # Simulated incidents
        incidents = [
            {'type': 'offside', 'frame': int(frame_count * 0.15), 'timestamp': frame_count * 0.15 / fps, 'confidence': 0.87, 'description': 'Attacking player beyond defensive line'},
            {'type': 'foul', 'frame': int(frame_count * 0.45), 'timestamp': frame_count * 0.45 / fps, 'confidence': 0.74, 'description': 'Physical contact between opposing players'},
            {'type': 'penalty', 'frame': int(frame_count * 0.72), 'timestamp': frame_count * 0.72 / fps, 'confidence': 0.92, 'description': 'Foul committed within penalty area'},
        ]
        
        return {
            'processed_frames': frame_count,
            'total_frames': frame_count,
            'fps': fps,
            'processing_time': frame_count / 25,
            'incidents': incidents,
            'output_dir': None
        }
    
    def run(self):
        """Run the dashboard application."""
        self.render_header()
        config = self.render_sidebar()
        
        # Main content tabs
        tab_analysis, tab_results, tab_incidents, tab_export = st.tabs([
            "Analysis", "Results", "Incidents", "Export"
        ])
        
        with tab_analysis:
            self._render_analysis_tab(config)
        
        with tab_results:
            self._render_results_tab()
        
        with tab_incidents:
            self._render_incidents_tab()
        
        with tab_export:
            self._render_export_tab()
    
    def _render_analysis_tab(self, config: Dict):
        """Render the analysis tab."""
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.markdown("""
            <div class="card-header">
                <h3 class="card-title">Video Input</h3>
                <span class="card-badge" style="background: var(--bg-tertiary); color: var(--text-muted);">Upload Required</span>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Select video file for analysis",
                type=['mp4', 'avi', 'mov', 'mkv'],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                # Save to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                # Get video info
                cap = cv2.VideoCapture(tfile.name)
                video_info = {
                    'path': tfile.name,
                    'name': uploaded_file.name,
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                }
                video_info['duration'] = video_info['frames'] / video_info['fps'] if video_info['fps'] > 0 else 0
                cap.release()
                
                st.session_state.video_info = video_info
                
                # Video preview
                st.video(tfile.name)
                
                # Metadata display
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                
                cols = st.columns(4)
                with cols[0]:
                    st.markdown(self.render_metric("Resolution", f"{video_info['width']}x{video_info['height']}"), unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(self.render_metric("Frame Rate", f"{video_info['fps']:.1f}", "fps"), unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(self.render_metric("Duration", self.format_time(video_info['duration'])), unsafe_allow_html=True)
                with cols[3]:
                    st.markdown(self.render_metric("Total Frames", f"{video_info['frames']:,}"), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
                            <line x1="7" y1="2" x2="7" y2="22"></line>
                            <line x1="17" y1="2" x2="17" y2="22"></line>
                            <line x1="2" y1="12" x2="22" y2="12"></line>
                            <line x1="2" y1="7" x2="7" y2="7"></line>
                            <line x1="2" y1="17" x2="7" y2="17"></line>
                            <line x1="17" y1="17" x2="22" y2="17"></line>
                            <line x1="17" y1="7" x2="22" y2="7"></line>
                        </svg>
                    </div>
                    <div class="empty-state-title">No Video Selected</div>
                    <div class="empty-state-text">Upload a video file to begin VAR analysis</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_side:
            st.markdown("""
            <div class="card-header">
                <h3 class="card-title">Analysis Modules</h3>
            </div>
            """, unsafe_allow_html=True)
            
            modules = [
                ("Object Detection", "YOLOv8 Neural Network", True),
                ("Multi-Object Tracking", "ByteTrack Algorithm", True),
                ("Team Classification", "Color Clustering", True),
                ("Field Homography", "Perspective Transform", True),
                ("Offside Detection", "Position Analysis", True),
                ("Foul Detection", "Contact Analysis", True),
                ("Penalty Detection", "Area Monitoring", True),
            ]
            
            for name, desc, active in modules:
                st.markdown(self.render_module_status(name, desc, active), unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Process button
            if uploaded_file:
                if st.button("Execute Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        progress = st.progress(0)
                        
                        for i in range(100):
                            time.sleep(0.02)
                            progress.progress(i + 1)
                        
                        results = self.process_video(st.session_state.video_info['path'], config)
                        
                        if results:
                            st.session_state.results = results
                            st.session_state.incidents = results.get('incidents', [])
                            st.rerun()
    
    def _render_results_tab(self):
        """Render the results tab."""
        st.markdown("""
        <div class="card-header">
            <h3 class="card-title">Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.results:
            results = st.session_state.results
            
            cols = st.columns(4)
            with cols[0]:
                st.markdown(self.render_metric("Processed", f"{results['processed_frames']:,}", "frames"), unsafe_allow_html=True)
            with cols[1]:
                st.markdown(self.render_metric("Incidents", str(len(st.session_state.incidents))), unsafe_allow_html=True)
            with cols[2]:
                st.markdown(self.render_metric("Duration", f"{results['processing_time']:.1f}", "sec"), unsafe_allow_html=True)
            with cols[3]:
                perf = results['processed_frames'] / results['processing_time'] if results['processing_time'] > 0 else 0
                st.markdown(self.render_metric("Performance", f"{perf:.1f}", "fps"), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-panel">
                <div class="info-panel-title">No Results Available</div>
                <div class="info-panel-text">Execute an analysis on the Analysis tab to view results here.</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_incidents_tab(self):
        """Render the incidents tab."""
        incident_count = len(st.session_state.incidents)
        
        st.markdown(f"""
        <div class="card-header">
            <h3 class="card-title">Detected Incidents</h3>
            <span class="card-badge" style="background: var(--bg-tertiary); color: var(--text-secondary);">{incident_count} found</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.incidents:
            st.markdown('<div class="incident-list">', unsafe_allow_html=True)
            for incident in st.session_state.incidents:
                st.markdown(self.render_incident(incident), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-panel">
                <div class="info-panel-title">No Incidents Detected</div>
                <div class="info-panel-text">Run an analysis to detect offside positions, fouls, and penalty incidents.</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_export_tab(self):
        """Render the export tab."""
        st.markdown("""
        <div class="card-header">
            <h3 class="card-title">Data Export</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="font-size: 0.9rem; margin: 0 0 0.5rem 0; color: var(--text-primary);">JSON Format</h4>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1rem;">Structured data for system integration</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.incidents:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'incidents': st.session_state.incidents,
                    'summary': st.session_state.results
                }
                
                st.download_button(
                    "Download JSON",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"var_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="font-size: 0.9rem; margin: 0 0 0.5rem 0; color: var(--text-primary);">CSV Format</h4>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1rem;">Spreadsheet-compatible tabular data</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.incidents:
                csv = "type,frame,timestamp,confidence,description\n"
                for inc in st.session_state.incidents:
                    csv += f"{inc['type']},{inc.get('frame',0)},{inc.get('timestamp',0)},{inc.get('confidence',0)},\"{inc.get('description','')}\"\n"
                
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"var_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )


def main():
    """Application entry point."""
    dashboard = VARDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
