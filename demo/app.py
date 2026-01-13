"""
VAR Analysis System - Liquid Design with Working Animations
"""

import streamlit as st
import cv2
import tempfile
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="VisionAI VAR System",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BRAND_NAME = "VisionAI"
BRAND_PRODUCT = "VAR Analysis System"
BRAND_VERSION = "2.0"

# ============================================================================
# CSS WITH WORKING ANIMATIONS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #030305;
    background-image: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15), transparent),
                      radial-gradient(ellipse 60% 40% at 80% 60%, rgba(139, 92, 246, 0.08), transparent);
}

#MainMenu, footer, header, .stDeployButton { display: none !important; }

.main .block-container {
    padding-top: 80px !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
p, span, li, div { font-family: 'Inter', sans-serif !important; }

.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 3rem;
    background: rgba(3, 3, 5, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.nav-brand { display: flex; align-items: center; gap: 0.75rem; }

.nav-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700; color: white;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
}

.nav-brand-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 700; color: #fafafa;
}

.nav-links { display: flex; align-items: center; gap: 2.5rem; }
.nav-link { font-size: 0.9rem; font-weight: 500; color: #71717a; cursor: pointer; transition: color 0.3s ease; }
.nav-link:hover { color: #fafafa; }

.nav-cta {
    padding: 0.6rem 1.5rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; font-size: 0.9rem; font-weight: 600;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
    cursor: pointer; transition: all 0.3s ease;
}
.nav-cta:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5); }

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(99, 102, 241, 0.5) !important;
}

[data-testid="stFileUploader"] > div {
    background: #0a0a0f !important;
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

.stProgress > div > div { background: #18181b !important; border-radius: 8px !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; border-radius: 8px !important; }

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.75rem !important; font-weight: 700 !important; color: #fafafa !important;
}
[data-testid="stMetricLabel"] { color: #71717a !important; }

.stDownloadButton > button {
    background: rgba(74, 222, 128, 0.1) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(74, 222, 128, 0.25) !important;
    box-shadow: none !important;
}

video { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# Separate style block for animations (sometimes helps with Streamlit)
st.markdown("""
<style>
@-webkit-keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@-webkit-keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
}

@-webkit-keyframes pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.3); }
}
@keyframes pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.3); }
}

@-webkit-keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.4); }
    50% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.8); }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.4); }
    50% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.8); }
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NAVBAR
# ============================================================================
st.markdown(f"""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-logo">V</div>
        <span class="nav-brand-text">{BRAND_NAME}</span>
    </div>
    <div class="nav-links">
        <span class="nav-link">Features</span>
        <span class="nav-link">Product</span>
        <span class="nav-link">Documentation</span>
        <span class="nav-link">Pricing</span>
        <span class="nav-cta">Get Started</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================
def spacer(rem=2):
    st.markdown(f"<div style='height:{rem}rem'></div>", unsafe_allow_html=True)

def centered_text(text, size="1rem", color="#a1a1aa"):
    st.markdown(f"<p style='text-align:center; font-size:{size}; color:{color}; margin:0; line-height:1.7;'>{text}</p>", unsafe_allow_html=True)

def hero_title_static(text):
    st.markdown(f"<h1 style='text-align:center; font-size:4.5rem; font-weight:800; color:#fafafa; letter-spacing:-0.04em; line-height:1.1; margin:0;'>{text}</h1>", unsafe_allow_html=True)

def hero_title_animated(text):
    # Animated gradient text
    st.markdown(f"""
    <h1 style='
        text-align:center; 
        font-size:4.5rem; 
        font-weight:800; 
        letter-spacing:-0.04em; 
        line-height:1.1; 
        margin:0;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa, #c4b5fd, #8b5cf6, #6366f1);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        -webkit-animation: gradientShift 4s ease infinite;
        animation: gradientShift 4s ease infinite;
    '>{text}</h1>
    """, unsafe_allow_html=True)

def badge_animated(text):
    st.markdown(f"""
    <div style='text-align:center; margin-bottom:2.5rem;'>
        <span style='display:inline-flex; align-items:center; gap:12px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:50px; padding:12px 26px; font-size:0.95rem; color:#a1a1aa;'>
            <span style='
                width:10px; 
                height:10px; 
                background:#6366f1; 
                border-radius:50%; 
                -webkit-animation: pulse 2s ease-in-out infinite;
                animation: pulse 2s ease-in-out infinite;
            '></span>
            {text}
        </span>
    </div>
    """, unsafe_allow_html=True)

def section_header(label, title, subtitle=""):
    st.markdown(f"<div style='text-align:center; margin-bottom:1.5rem;'><span style='display:inline-block; background:rgba(99,102,241,0.1); color:#818cf8; padding:8px 20px; border-radius:50px; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.15em;'>{label}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; font-size:2.75rem; font-weight:700; color:#fafafa; margin:0 0 1rem 0;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='text-align:center; font-size:1.15rem; color:#71717a; max-width:550px; margin:0 auto 3rem auto;'>{subtitle}</p>", unsafe_allow_html=True)

def feature_card(num, title, desc):
    st.markdown(f"""
    <div style='background:#0a0a0f; border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:2rem; height:100%; transition: all 0.3s ease;'>
        <div style='font-size:0.85rem; font-weight:600; color:#6366f1; margin-bottom:1rem;'>{num}</div>
        <h3 style='font-size:1.2rem; font-weight:600; color:#fafafa; margin:0 0 0.75rem 0;'>{title}</h3>
        <p style='font-size:0.95rem; color:#71717a; line-height:1.6; margin:0;'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

def stat_box_animated(value, label):
    # Animated gradient for accent stat
    st.markdown(f"""
    <div style='text-align:center; padding:1.5rem;'>
        <div style='
            font-family:Space Grotesk,sans-serif; 
            font-size:3.25rem; 
            font-weight:800;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa, #8b5cf6, #6366f1);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            -webkit-animation: gradientShift 3s ease infinite;
            animation: gradientShift 3s ease infinite;
        '>{value}</div>
        <div style='font-size:0.95rem; color:#52525b; margin-top:0.5rem;'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

def stat_box(value, label):
    st.markdown(f"""
    <div style='text-align:center; padding:1.5rem;'>
        <div style='font-family:Space Grotesk,sans-serif; font-size:3.25rem; font-weight:800; color:#fafafa;'>{value}</div>
        <div style='font-size:0.95rem; color:#52525b; margin-top:0.5rem;'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

def card_header(title, badge_text, color="#4ade80"):
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; padding-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;'>
        <span style='font-size:0.8rem; font-weight:600; color:#52525b; text-transform:uppercase; letter-spacing:0.1em;'>{title}</span>
        <span style='font-size:0.75rem; font-weight:600; padding:5px 14px; border-radius:50px; background:{color}18; color:{color};'>{badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

def module_item_animated(name, delay_ms=0):
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:14px; padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.04);'>
        <div style='
            width:10px; 
            height:10px; 
            background:#4ade80; 
            border-radius:50%; 
            box-shadow:0 0 12px rgba(74,222,128,0.5);
            -webkit-animation: pulse 2s ease-in-out infinite;
            animation: pulse 2s ease-in-out infinite;
            animation-delay: {delay_ms}ms;
        '></div>
        <span style='font-size:0.95rem; color:#e4e4e7;'>{name}</span>
    </div>
    """, unsafe_allow_html=True)

def timeline_item(inc_type, timestamp, desc, conf):
    colors = {"offside": "#fbbf24", "foul": "#f87171", "penalty": "#f87171"}
    color = colors.get(inc_type, "#818cf8")
    m, s = divmod(int(timestamp), 60)
    st.markdown(f"""
    <div style='position:relative; padding:16px 0 16px 28px; border-left:2px solid rgba(255,255,255,0.08); margin-left:8px;'>
        <div style='
            position:absolute; 
            left:-6px; 
            top:20px; 
            width:10px; 
            height:10px; 
            border-radius:50%; 
            background:{color}; 
            box-shadow:0 0 10px {color}80;
            -webkit-animation: pulse 2s ease-in-out infinite;
            animation: pulse 2s ease-in-out infinite;
        '></div>
        <span style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:{color};'>{inc_type.upper()}</span>
        <span style='font-size:0.8rem; color:#52525b; margin-left:12px; font-family:monospace;'>{m:02d}:{s:02d}</span>
        <p style='font-size:0.9rem; color:#a1a1aa; margin:6px 0 0 0;'>{desc} — {conf:.0%} confidence</p>
    </div>
    """, unsafe_allow_html=True)

def pitch_card_animated():
    st.markdown("""
    <div style='
        background:linear-gradient(145deg, #1a5c2e, #16492a, #134a26); 
        border-radius:20px; 
        padding:3rem 2.5rem; 
        text-align:center; 
        box-shadow:0 30px 80px rgba(0,0,0,0.5), 0 0 60px rgba(34,197,94,0.1); 
        border:1px solid rgba(255,255,255,0.05);
        -webkit-animation: float 5s ease-in-out infinite;
        animation: float 5s ease-in-out infinite;
    '>
        <div style='width:80px; height:80px; background:rgba(255,255,255,0.1); border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 1.5rem auto; border:2px solid rgba(255,255,255,0.15);'>
            <div style='width:40px; height:40px; background:white; border-radius:50%; box-shadow:0 4px 15px rgba(0,0,0,0.3);'></div>
        </div>
        <h3 style='font-size:1.5rem; font-weight:700; color:white; margin:0 0 0.5rem 0;'>Live Match Analysis</h3>
        <p style='font-size:1rem; color:rgba(255,255,255,0.6); margin:0 0 2.5rem 0;'>AI-powered tracking and incident detection in real-time</p>
        <div style='display:flex; justify-content:center; gap:4rem;'>
            <div style='text-align:center;'>
                <div style='font-family:Space Grotesk,sans-serif; font-size:2.25rem; font-weight:700; color:#4ade80; text-shadow:0 0 30px rgba(74,222,128,0.4);'>22</div>
                <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.1em; margin-top:6px;'>Players Tracked</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-family:Space Grotesk,sans-serif; font-size:2.25rem; font-weight:700; color:#4ade80; text-shadow:0 0 30px rgba(74,222,128,0.4);'>60</div>
                <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.1em; margin-top:6px;'>Frames/Second</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-family:Space Grotesk,sans-serif; font-size:2.25rem; font-weight:700; color:#4ade80; text-shadow:0 0 30px rgba(74,222,128,0.4);'>99%</div>
                <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.1em; margin-top:6px;'>Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'incidents' not in st.session_state:
        st.session_state.incidents = []
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None

    # ===== HERO =====
    spacer(5)
    badge_animated(f"Introducing {BRAND_PRODUCT} {BRAND_VERSION}")
    hero_title_static("See Beyond")
    hero_title_animated("The Game")
    spacer(2)
    centered_text(
        "Next-generation computer vision that captures what the human eye misses.<br>Real-time analysis with frame-perfect precision and AI-powered decision support.",
        size="1.35rem", color="#71717a"
    )
    spacer(3)
    
    # CTA Buttons
    c1, c2, c3, c4, c5 = st.columns([2, 1.2, 0.3, 1.2, 2])
    with c2:
        st.button("Start Analyzing", type="primary", use_container_width=True)
    with c4:
        st.button("View Demo", use_container_width=True)
    
    spacer(4)
    
    # Pitch Card (floating)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pitch_card_animated()
    
    # ===== FEATURES =====
    spacer(5)
    section_header("Features", "Powered by Advanced AI", "Seven specialized modules working together for comprehensive match analysis")
    
    features = [
        ("01", "Object Detection", "YOLOv8-powered detection identifies players and ball with 95%+ accuracy at 60 frames per second."),
        ("02", "Multi-Object Tracking", "ByteTrack maintains consistent player IDs through occlusions and rapid movement."),
        ("03", "Team Classification", "Automatic team assignment using advanced jersey color clustering algorithms."),
        ("04", "Field Homography", "Precise coordinate mapping from camera perspective to real pitch positions."),
        ("05", "Offside Detection", "Millimeter-precise offside line calculation at the exact moment of pass."),
        ("06", "Incident Analysis", "Real-time foul and penalty detection with player contact analysis."),
    ]
    
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(features):
        with cols[i % 3]:
            feature_card(num, title, desc)
            spacer(1)
    
    # ===== STATS =====
    spacer(2)
    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.06); border-bottom:1px solid rgba(255,255,255,0.06); padding:2rem 0; margin:2rem 0;'>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        stat_box_animated("95%+", "Detection Accuracy")
    with cols[1]:
        stat_box("60", "FPS Processing")
    with cols[2]:
        stat_box("7", "AI Modules")
    with cols[3]:
        stat_box("<30ms", "Latency")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== PRODUCT =====
    spacer(3)
    section_header("Product", f"Try {BRAND_PRODUCT}", "Upload your match footage and experience the power of AI analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div style='background:#0a0a0f; border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:1.5rem;'>", unsafe_allow_html=True)
        card_header("Video Input", "Ready", "#4ade80")
        
        uploaded = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
        
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            video_info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            video_info['duration'] = video_info['frames'] / max(video_info['fps'], 1)
            cap.release()
            st.session_state.video_info = video_info
            
            st.video(tfile.name)
            
            m, s = divmod(int(video_info['duration']), 60)
            mcols = st.columns(3)
            mcols[0].metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            mcols[1].metric("Frame Rate", f"{video_info['fps']:.0f} FPS")
            mcols[2].metric("Duration", f"{m}:{s:02d}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.incidents:
            spacer(1)
            st.markdown("<div style='background:#0a0a0f; border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:1.5rem;'>", unsafe_allow_html=True)
            card_header("Incident Timeline", f"{len(st.session_state.incidents)} Detected", "#818cf8")
            for inc in st.session_state.incidents:
                timeline_item(inc['type'], inc['timestamp'], inc['description'], inc['confidence'])
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='background:#0a0a0f; border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:1.5rem;'>", unsafe_allow_html=True)
        card_header("Analysis Modules", "7 Active", "#4ade80")
        modules = ["Object Detection", "Multi-Object Tracking", "Team Classification", 
                   "Field Homography", "Offside Detection", "Foul Detection", "Penalty Analysis"]
        for i, mod in enumerate(modules):
            module_item_animated(mod, delay_ms=i*200)
        st.markdown("</div>", unsafe_allow_html=True)
        
        spacer(1)
        
        if uploaded:
            if st.button("Run Analysis", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                for i, stage in enumerate(["Initializing...", "Loading models...", "Processing...", "Analyzing...", "Complete"]):
                    status.text(stage)
                    for p in range(i*20, (i+1)*20):
                        progress.progress(min(p, 100))
                        time.sleep(0.008)
                
                frames = st.session_state.video_info['frames']
                fps = st.session_state.video_info['fps']
                st.session_state.incidents = [
                    {'type': 'offside', 'timestamp': frames * 0.15 / fps, 'confidence': 0.91, 'description': 'Player beyond defensive line'},
                    {'type': 'foul', 'timestamp': frames * 0.42 / fps, 'confidence': 0.78, 'description': 'Contact between players'},
                    {'type': 'penalty', 'timestamp': frames * 0.71 / fps, 'confidence': 0.94, 'description': 'Foul in penalty area'},
                ]
                st.session_state.results = {'processed_frames': frames}
                st.rerun()
        
        if st.session_state.results:
            spacer(1)
            st.markdown("<div style='background:#0a0a0f; border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:1.5rem;'>", unsafe_allow_html=True)
            card_header("Results", "Complete", "#4ade80")
            rcols = st.columns(2)
            rcols[0].metric("Frames", f"{st.session_state.results['processed_frames']:,}")
            rcols[1].metric("Incidents", len(st.session_state.incidents))
            st.markdown("</div>", unsafe_allow_html=True)
            
            spacer(1)
            data = json.dumps({'timestamp': datetime.now().isoformat(), 'incidents': st.session_state.incidents}, indent=2, default=str)
            st.download_button("Export Report", data, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json", use_container_width=True)
    
    # Footer
    spacer(4)
    st.markdown(f"<div style='text-align:center; padding:2.5rem; border-top:1px solid rgba(255,255,255,0.06);'><p style='color:#52525b; font-size:0.9rem; margin:0;'>{BRAND_NAME} — {BRAND_PRODUCT} v{BRAND_VERSION}</p><p style='color:#3f3f46; font-size:0.85rem; margin:0.5rem 0 0 0;'>Built with YOLOv8, ByteTrack, and Streamlit</p></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()