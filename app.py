import cv2
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="VisionGuard AI", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. SURGICAL CSS OVERRIDE (ICONS PROTECTED)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen&family=VT323&display=swap');

    /* 1. FORCE VT323 FONT ON TEXT (BUT NOT ICONS) */
    .stApp, p, label, div[data-testid="stMarkdownContainer"] {
        font-family: 'VT323', monospace !important;
    }

    /* Explicitly restore Streamlit's internal icons */
    span[class*="material-icons"],
    span[class*="material-symbols"],
    .material-icons,
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* 2. BACKGROUND & CRT EFFECT */
    .stApp {
        background-color: #020502 !important;
        background-image: repeating-linear-gradient(
            0deg,
            rgba(0, 255, 0, 0.03),
            rgba(0, 255, 0, 0.03) 1px,
            transparent 1px,
            transparent 2px
        ) !important;
        color: #00FF00 !important;
    }

    /* 3. PANELS & SIDEBAR (Glass + Neon) */
    [data-testid="stSidebar"], [data-testid="metric-container"], .telemetry-panel {
        background: rgba(5, 15, 5, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(0, 255, 0, 0.5) !important;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.1) !important;
        border-radius: 0px !important;
        padding: 1.5rem !important;
    }

    /* 4. FIX THE DOUBLE ALERT BOX */
    [data-testid="stAlert"] {
        background: rgba(5, 15, 5, 0.8) !important;
        border: 1px solid #00FF00 !important;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.2) !important;
        border-radius: 0px !important;
    }
    [data-testid="stAlert"] > div { background: transparent !important; }
    [data-testid="stAlert"] * {
        color: #00FF00 !important;
        font-size: 1.4rem !important;
        font-family: 'VT323', monospace !important;
    }

    /* 5. TYPOGRAPHY: TITLES & NUMBERS (SILKSCREEN 8-BIT) */
    h1, h2, h3, [data-testid="stMetricValue"] > div { 
        font-family: 'Silkscreen', cursive !important; 
        color: #00FF00 !important;
        text-shadow: 2px 2px 0px rgba(0, 100, 0, 0.8) !important; 
        text-transform: uppercase;
        font-weight: normal !important;
    }

    [data-testid="stMetricValue"] > div { font-size: 3.5rem !important; }
    
    [data-testid="stMetricLabel"] * { 
        font-size: 1.4rem !important; 
        color: #00CC00 !important; 
        font-family: 'VT323', monospace !important;
    }

    .terminal-header {
        color: #00FF00;
        font-size: 1.4rem;
        font-family: 'VT323', monospace !important;
        border-bottom: 1px dashed rgba(0, 255, 0, 0.4);
        padding-bottom: 5px;
        margin-bottom: 15px;
        text-transform: uppercase;
    }

    /* 6. KILL THE RED SLIDERS (Using Streamlit's baseweb architecture) */
    div[data-baseweb="slider"] > div > div > div {
        background: #00FF00 !important; 
    }
    [data-testid="stThumbValue"] {
        background: #001100 !important;
        color: #00FF00 !important;
        border: 1px solid #00FF00 !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.2rem !important;
    }

    /* 7. FORCE THE SIDEBAR ARROW TO BE VISIBLE */
    [data-testid="collapsedControl"] {
        background: #001100 !important;
        border: 1px solid #00FF00 !important;
        border-radius: 0px !important;
        z-index: 999999 !important;
    }
    [data-testid="collapsedControl"] svg {
        color: #00FF00 !important;
        fill: #00FF00 !important;
    }

    /* CLEANUP */
    header[data-testid="stHeader"] { background: transparent !important; }
    .stDeployButton, #MainMenu { display: none !important; }
    .block-container { padding-top: 1rem !important; max-width: 95% !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>VisionGuard :: Autonomous Agent</h1>", unsafe_allow_html=True)

# ==========================================
# 3. CORE LOGIC 
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()
G = 9.81  # Gravity m/s^2

# ==========================================
# 4. SIDEBAR CONTROLS 
# ==========================================
st.sidebar.markdown("<h2>Agent Telemetry Controls</h2>", unsafe_allow_html=True)
speed_kmh = st.sidebar.slider("VELOCITY VECTOR (km/h)", min_value=30, max_value=160, value=90, step=5)
weather = st.sidebar.selectbox("FRICTION COEFFICIENT (µ)", ["DRY_STEEL", "WET_TRACK", "SEVERE_FROST"])

friction_map = {"DRY_STEEL": 0.3, "WET_TRACK": 0.15, "SEVERE_FROST": 0.08}
mu = friction_map[weather]
speed_ms = speed_kmh / 3.6
stopping_distance = (speed_ms ** 2) / (2 * mu * G)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>Spatial ROI Mask</h2>", unsafe_allow_html=True)
horizon_height = st.sidebar.slider("Horizon Axis Y (%)", 10, 90, 55, step=1) / 100.0
top_width = st.sidebar.slider("Track Top Width X (%)", 5, 50, 12, step=1) / 100.0
bottom_width = st.sidebar.slider("Track Bottom Width X (%)", 20, 100, 85, step=1) / 100.0

# ==========================================
# 5. MAIN LAYOUT (STRICT PLACEHOLDERS)
# ==========================================
col1, col2 = st.columns([5, 4], gap="large")

with col1:
    st.markdown("<div class='terminal-header'>// ANALYTICS.STREAM / PERCEPTION.AGENT_01</div>", unsafe_allow_html=True)
    video_placeholder = st.empty() 

with col2:
    st.markdown("<div class='terminal-header'>// SYSTEM.STATUS / DIGITAL.TWIN</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top:0;'>System State</h3>", unsafe_allow_html=True)
    status_placeholder = st.empty() 
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Telemetry Readouts</h3>", unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Kinetic Speed (km/h)", f"{speed_kmh}")
    with m_col2:
        st.metric("Calc Braking Boundary (m)", f"{stopping_distance:.1f}")

# ==========================================
# 6. VIDEO LOOP
# ==========================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    center_x = w / 2
    
    # Calculate mask geometry
    t_left_x = int(center_x - (w * top_width / 2))
    t_right_x = int(center_x + (w * top_width / 2))
    b_left_x = int(center_x - (w * bottom_width / 2))
    b_right_x = int(center_x + (w * bottom_width / 2))
    horizon_y = int(h * horizon_height)
    base_y = int(h * 0.95) 

    roi_points = np.array([
        [t_left_x, horizon_y], [t_right_x, horizon_y], 
        [b_right_x, base_y], [b_left_x, base_y]    
    ], np.int32)

    results = model(frame, classes=[0, 19, 2, 7], verbose=False)
    
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    danger_detected = False
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            bottom_center = (int((x1 + x2) / 2), int(y2))
            
            if cv2.pointPolygonTest(roi_points, bottom_center, False) >= 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                danger_detected = True
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

    video_placeholder.image(frame, channels="BGR", use_container_width=True)
    
    if danger_detected:
        status_placeholder.error(f"🚨 CRITICAL_OVERRIDE // OBSTACLE_DETECTED // AUTO_BRAKE_ENGAGED // STOPPING_BOUNDARY: {stopping_distance:.1f} M")
    else:
        status_placeholder.success("✅ PATH_CONCURRENCY // SYSTEM_NOMINAL // AGENT_DECISION: NO_OBSTRUCTION")
        
    time.sleep(0.01)

cap.release()