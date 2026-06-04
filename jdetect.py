import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import base64
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import gc  # Garbage collector

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Object Detection for Visually Impaired",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - MINIMIZED
# ============================================================================
st.markdown("""
<style>
    .main, .stApp { background-color: #000000; }
    .stButton > button {
        background-color: #FFD700;
        color: #000000;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #FFA500;
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# OPTIMIZED CONFIGURATION - REDUCED MEMORY FOOTPRINT
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5
CLOSE_THRESHOLD_PERCENT = 15
MEDIUM_THRESHOLD_PERCENT = 5
CENTER_ZONE_WIDTH_PERCENT = 40
COOLDOWN_SECONDS = 3
MAX_OBJECTS_PER_ANNOUNCEMENT = 3
MAX_HISTORY_SIZE = 100  # Limit history size
FRAME_PROCESS_INTERVAL = 5  # Process every 5th frame

# Voice settings
VOICE_RATE = 1.0
VOICE_PITCH = 1.0

# IMPORTANT: Use smaller set of critical objects only
IMPORTANT_OBJECTS = {
    'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle',
    'fire hydrant', 'stop sign', 'dog', 'cat', 'chair', 'bed',
    'cell phone', 'bottle', 'cup', 'book', 'laptop'
}

# ============================================================================
# OPTIMIZED TTS MANAGER
# ============================================================================
class BrowserTTSManager:
    def __init__(self):
        self.muted = False
        self.last_alert_time = defaultdict(float)
        self.rate = 1.0
        self.pitch = 1.0
        
    def speak(self, text):
        if not self.muted and text:
            text_escaped = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            js_code = f"""
            <script>
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var utterance = new SpeechSynthesisUtterance('{text_escaped}');
                    utterance.rate = {self.rate};
                    utterance.pitch = {self.pitch};
                    window.speechSynthesis.speak(utterance);
                }}
            </script>
            """
            st.components.v1.html(js_code, height=0, width=0)
    
    def can_announce(self, object_name, current_time):
        return (current_time - self.last_alert_time[object_name]) >= COOLDOWN_SECONDS
    
    def update_alert_time(self, object_name, current_time):
        self.last_alert_time[object_name] = current_time
    
    def toggle_mute(self):
        self.muted = not self.muted
        return not self.muted
    
    def set_rate(self, rate):
        self.rate = max(0.5, min(2.0, rate))
    
    def set_pitch(self, pitch):
        self.pitch = max(0.5, min(2.0, pitch))
    
    def test_speech(self):
        self.speak("Voice system ready")

# ============================================================================
# OPTIMIZED DETECTION SYSTEM - USING YOLOV8NANO (Smallest Model)
# ============================================================================
class ObjectDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.frame_count = 0
        self.model_loaded = False

    def load_model(self):
        """Load smallest YOLO model to save memory"""
        with st.spinner("Loading model (yolov8nano)..."):
            try:
                # Force garbage collection before loading
                gc.collect()
                # Use the smallest model available
                self.model = YOLO('yolov8n.pt')
                self.model_loaded = True
                return True
            except Exception as e:
                st.error(f"Failed: {e}")
                return False

    def estimate_proximity(self, bbox_area, total_area):
        area_percentage = (bbox_area / total_area) * 100
        if area_percentage >= CLOSE_THRESHOLD_PERCENT:
            return "close", area_percentage
        elif area_percentage >= MEDIUM_THRESHOLD_PERCENT:
            return "medium distance", area_percentage
        return "far", area_percentage

    def is_in_center_zone(self, bbox_center_x, frame_width):
        center_threshold = (frame_width * CENTER_ZONE_WIDTH_PERCENT) / 200
        frame_center = frame_width / 2
        return abs(bbox_center_x - frame_center) <= center_threshold

    def process_frame(self, frame, save_detection=False):
        """Optimized frame processing"""
        if not self.model_loaded or self.model is None:
            return frame, []

        self.frame_count += 1
        
        # Process fewer frames to reduce load
        if self.frame_count % FRAME_PROCESS_INTERVAL != 0:
            return frame, []

        # Resize frame for faster processing (maintain aspect ratio)
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w = 640
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        total_area = frame.shape[0] * frame.shape[1]
        detected_obstacles = []
        detection_results = []

        # Run detection with lower memory footprint
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device='cpu')

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Limit to top 5 detections per frame for performance
                for box in boxes[:5]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox_area = (x2 - x1) * (y2 - y1)
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    if class_name not in IMPORTANT_OBJECTS:
                        continue

                    bbox_center_x = (x1 + x2) / 2

                    if not self.is_in_center_zone(bbox_center_x, frame.shape[1]):
                        continue

                    proximity, area_percent = self.estimate_proximity(bbox_area, total_area)

                    if save_detection and proximity in ['close', 'medium distance']:
                        # Limit history size
                        if len(self.detection_history) < MAX_HISTORY_SIZE:
                            detection_info = {
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'object': class_name,
                                'confidence': round(confidence, 2),
                                'proximity': proximity,
                            }
                            self.detection_history.append(detection_info)

                    # Draw simplified bounding box
                    if proximity == 'close':
                        color = (0, 0, 255)
                    elif proximity == 'medium distance':
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    label = f"{class_name[:3]}"
                    cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    if proximity in ['close', 'medium distance']:
                        detected_obstacles.append((class_name, proximity, area_percent))

        # Announce obstacles
        if detected_obstacles:
            current_time = time.time()
            filtered = []
            for obj_name, proximity, _ in detected_obstacles[:MAX_OBJECTS_PER_ANNOUNCEMENT]:
                if st.session_state.tts_manager.can_announce(obj_name, current_time):
                    filtered.append((obj_name, proximity))
                    st.session_state.tts_manager.update_alert_time(obj_name, current_time)
            
            if filtered:
                if len(filtered) == 1:
                    message = f"{filtered[0][1]} {filtered[0][0]}"
                else:
                    message = f"{len(filtered)} objects ahead"
                st.session_state.tts_manager.speak(message)

        return frame, detection_results

    def process_image(self, image):
        """Process single image - optimized"""
        if not self.model_loaded or self.model is None:
            return None, []

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.process_frame(image, save_detection=True)

    def get_export_data(self):
        if not self.detection_history:
            return pd.DataFrame()
        return pd.DataFrame(self.detection_history)

    def clear_history(self):
        self.detection_history = []
        self.frame_count = 0
        gc.collect()

# ============================================================================
# OPTIMIZED WEBRTC TRANSFORMER
# ============================================================================
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detection_system = None
        self.last_update = 0
        
    def recv(self, frame):
        if self.detection_system is None or not self.detection_system.model_loaded:
            return frame
        
        img = frame.to_ndarray(format="bgr24")
        processed_img, detections = self.detection_system.process_frame(
            img, save_detection=st.session_state.save_detections
        )
        
        # Throttle UI updates
        current_time = time.time()
        if current_time - self.last_update > 0.5 and detections:
            self.last_update = current_time
            for det in detections:
                if det.get('proximity') in ['close', 'medium distance']:
                    st.session_state.detection_count = st.session_state.get('detection_count', 0) + 1
                    log_entry = f"⚠️ {det['object'].upper()}"
                    current_log = st.session_state.get('detection_log', [])
                    current_log.insert(0, log_entry)
                    if len(current_log) > 10:
                        current_log = current_log[:10]
                    st.session_state.detection_log = current_log
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ============================================================================
# INITIALIZATION
# ============================================================================
if 'tts_manager' not in st.session_state:
    st.session_state.tts_manager = BrowserTTSManager()
if 'detection_system' not in st.session_state:
    st.session_state.detection_system = ObjectDetectionSystem()
if 'mode' not in st.session_state:
    st.session_state.mode = "webcam"
if 'save_detections' not in st.session_state:
    st.session_state.save_detections = True
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

detection_system = st.session_state.detection_system

# ============================================================================
# SIMPLIFIED SIDEBAR
# ============================================================================
st.sidebar.title("🎛️ Controls")

# Model loading - Simplified
if st.sidebar.button("📦 Load Model", use_container_width=True):
    if detection_system.load_model():
        st.sidebar.success("✅ Ready!")
        st.session_state.tts_manager.test_speech()
        st.rerun()
    else:
        st.sidebar.error("❌ Failed")

st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.radio("Mode", ["📷 Webcam", "🖼️ Image"], index=0)
st.session_state.mode = "webcam" if mode == "📷 Webcam" else "image"

st.sidebar.markdown("---")

# Detection settings - Simplified
confidence = st.sidebar.slider("Confidence", 0.3, 0.7, 0.5, 0.05)
globals()['CONFIDENCE_THRESHOLD'] = confidence

st.sidebar.markdown("---")

# Voice settings
voice_rate = st.sidebar.slider("Speed", 0.7, 1.5, 1.0, 0.05)
st.session_state.tts_manager.set_rate(voice_rate)

if st.sidebar.button("🔊 Test", use_container_width=True):
    st.session_state.tts_manager.test_speech()

if st.sidebar.button("🔇 Mute" if not st.session_state.tts_manager.muted else "🔊 Unmute", use_container_width=True):
    st.session_state.tts_manager.toggle_mute()
    st.rerun()

st.sidebar.markdown("---")

# Data management
if st.sidebar.button("📊 Export CSV", use_container_width=True):
    df = detection_system.get_export_data()
    if not df.empty:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="detections.csv">Download</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear", use_container_width=True):
    detection_system.clear_history()
    st.session_state.detection_count = 0
    st.session_state.detection_log = []
    st.sidebar.success("Cleared!")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Lower confidence = more detections")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("👁️ Object Detection for Visually Impaired")
st.markdown("*Voice feedback - Works in browser*")
st.markdown("---")

# ============================================================================
# WEBCAM MODE
# ============================================================================
if st.session_state.mode == "webcam":
    if not detection_system.model_loaded:
        st.warning("⚠️ **Load model first!** Click 'Load Model' in sidebar.")
    else:
        st.info("📹 Click **Start** → Allow camera → Point at objects")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            status_placeholder = st.empty()
            transformer = VideoTransformer()
            transformer.detection_system = detection_system
            
            webrtc_ctx = webrtc_streamer(
                key="detection",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=lambda: transformer,
                async_processing=True,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            if webrtc_ctx and webrtc_ctx.state.playing:
                status_placeholder.success("🎥 **Active**")
            else:
                status_placeholder.info("⏸️ Click Start")
        
        with col2:
            st.subheader("Detections")
            if st.session_state.detection_log:
                for log in st.session_state.detection_log[:5]:
                    st.warning(log)
            st.metric("Total", st.session_state.detection_count)

# ============================================================================
# IMAGE MODE
# ============================================================================
else:
    st.subheader("📷 Upload Image")
    
    uploaded_file = st.file_uploader("Choose image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        if not detection_system.model_loaded:
            st.error("⚠️ Load model first!")
        else:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, use_container_width=True)
            
            with st.spinner("Detecting..."):
                processed, detections = detection_system.process_image(image)
                
                if processed is not None:
                    with col2:
                        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                        st.image(processed_rgb, use_container_width=True)
                    
                    if detections:
                        st.success(f"✅ Found {len(detections)} objects!")
                        df = pd.DataFrame([{k:v for k,v in d.items() if k != 'bbox'} for d in detections])
                        st.dataframe(df, use_container_width=True)
                        
                        for det in detections:
                            if det.get('proximity') in ['close', 'medium distance']:
                                st.session_state.tts_manager.speak(f"{det['proximity']} {det['object']}")
                                time.sleep(0.3)
                    else:
                        st.warning("No objects detected")

# ============================================================================
# ANALYTICS (Lightweight)
# ============================================================================
with st.expander("📊 Analytics"):
    if detection_system.detection_history:
        df = detection_system.get_export_data()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(df))
        with col2:
            st.metric("Objects", df['object'].nunique())
        with col3:
            close = len(df[df['proximity'] == 'close'])
            st.metric("Close", close)
        
        if len(df) > 0:
            st.subheader("Top Objects")
            st.bar_chart(df['object'].value_counts().head(5))
    else:
        st.info("No data yet. Run detection with 'Save' enabled.")
