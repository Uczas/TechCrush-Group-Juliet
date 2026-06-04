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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Object Detection",
    page_icon="👁️",
    layout="wide"
)

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5
CLOSE_THRESHOLD_PERCENT = 15
MEDIUM_THRESHOLD_PERCENT = 5
COOLDOWN_SECONDS = 3
MAX_OBJECTS_PER_ANNOUNCEMENT = 3
FRAME_PROCESS_INTERVAL = 3

# Smaller object set for faster processing
IMPORTANT_OBJECTS = {
    'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle',
    'dog', 'cat', 'chair', 'bottle', 'cell phone', 'book'
}

# ============================================================================
# BROWSER TTS MANAGER
# ============================================================================
class BrowserTTSManager:
    def __init__(self):
        self.muted = False
        self.last_alert_time = defaultdict(float)
        self.rate = 1.0
        
    def speak(self, text):
        if not self.muted and text:
            text_escaped = text.replace("'", "\\'").replace('"', '\\"')
            js_code = f"""
            <script>
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var utterance = new SpeechSynthesisUtterance('{text_escaped}');
                    utterance.rate = {self.rate};
                    window.speechSynthesis.speak(utterance);
                }}
            </script>
            """
            st.components.v1.html(js_code, height=0, width=0)
    
    def can_announce(self, obj, current_time):
        return (current_time - self.last_alert_time[obj]) >= COOLDOWN_SECONDS
    
    def update_alert_time(self, obj, current_time):
        self.last_alert_time[obj] = current_time
    
    def toggle_mute(self):
        self.muted = not self.muted
        return not self.muted
    
    def set_rate(self, rate):
        self.rate = max(0.5, min(2.0, rate))
    
    def test_speech(self):
        self.speak("System ready")

# ============================================================================
# DETECTION SYSTEM
# ============================================================================
class ObjectDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.frame_count = 0

    def load_model(self):
        with st.spinner("Loading model..."):
            try:
                self.model = YOLO('yolov8n.pt')
                return True
            except Exception as e:
                st.error(f"Failed: {e}")
                return False

    def estimate_proximity(self, bbox_area, total_area):
        percent = (bbox_area / total_area) * 100
        if percent >= CLOSE_THRESHOLD_PERCENT:
            return "close"
        elif percent >= MEDIUM_THRESHOLD_PERCENT:
            return "medium"
        return "far"

    def process_frame(self, frame, save_detection=False):
        if not self.model:
            return frame, []

        self.frame_count += 1
        if self.frame_count % FRAME_PROCESS_INTERVAL != 0:
            return frame, []

        # Resize for speed
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        total_area = frame.shape[0] * frame.shape[1]
        detected = []
        results = []

        results_yolo = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for r in results_yolo:
            if r.boxes:
                for box in r.boxes[:3]:  # Limit to top 3
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    if class_name not in IMPORTANT_OBJECTS:
                        continue

                    proximity = self.estimate_proximity((x2-x1)*(y2-y1), total_area)

                    if save_detection and proximity in ['close', 'medium']:
                        if len(self.detection_history) < 50:
                            self.detection_history.append({
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'object': class_name,
                                'confidence': round(confidence, 2),
                                'proximity': proximity
                            })

                    # Draw box
                    color = (0, 0, 255) if proximity == 'close' else (0, 165, 255) if proximity == 'medium' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame, class_name[:3], (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    if proximity in ['close', 'medium']:
                        detected.append((class_name, proximity))

        # Announce
        if detected:
            current_time = time.time()
            to_announce = []
            for obj, prox in detected[:MAX_OBJECTS_PER_ANNOUNCEMENT]:
                if st.session_state.tts_manager.can_announce(obj, current_time):
                    to_announce.append((obj, prox))
                    st.session_state.tts_manager.update_alert_time(obj, current_time)
            
            if to_announce:
                if len(to_announce) == 1:
                    st.session_state.tts_manager.speak(f"{to_announce[0][1]} {to_announce[0][0]}")
                else:
                    st.session_state.tts_manager.speak(f"{len(to_announce)} objects ahead")

        return frame, results

    def process_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.process_frame(image, save_detection=True)

    def get_export_data(self):
        if not self.detection_history:
            return pd.DataFrame()
        return pd.DataFrame(self.detection_history)

    def clear_history(self):
        self.detection_history = []
        self.frame_count = 0

# ============================================================================
# WEBRTC TRANSFORMER
# ============================================================================
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detection_system = None
        
    def recv(self, frame):
        if self.detection_system and self.detection_system.model:
            img = frame.to_ndarray(format="bgr24")
            processed, detections = self.detection_system.process_frame(img, 
                save_detection=st.session_state.save_detections)
            
            for det in detections:
                if det.get('proximity') in ['close', 'medium']:
                    st.session_state.detection_count = st.session_state.get('detection_count', 0) + 1
                    log = f"⚠️ {det['object']}"
                    log_list = st.session_state.get('detection_log', [])
                    log_list.insert(0, log)
                    st.session_state.detection_log = log_list[:10]
            
            from av import VideoFrame
            return VideoFrame.from_ndarray(processed, format="bgr24")
        return frame

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'tts_manager' not in st.session_state:
    st.session_state.tts_manager = BrowserTTSManager()
if 'detection_system' not in st.session_state:
    st.session_state.detection_system = ObjectDetectionSystem()
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'save_detections' not in st.session_state:
    st.session_state.save_detections = True

detection_system = st.session_state.detection_system

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🎛️ Controls")

if st.sidebar.button("📦 Load Model", use_container_width=True):
    if detection_system.load_model():
        st.sidebar.success("✅ Ready!")
        st.session_state.tts_manager.test_speech()
        st.rerun()

st.sidebar.markdown("---")

# Settings
confidence = st.sidebar.slider("Confidence", 0.3, 0.7, 0.5, 0.05)
globals()['CONFIDENCE_THRESHOLD'] = confidence

close_threshold = st.sidebar.slider("Close Threshold %", 10, 25, 15, 2)
globals()['CLOSE_THRESHOLD_PERCENT'] = close_threshold

st.sidebar.markdown("---")

# Voice
voice_rate = st.sidebar.slider("Voice Speed", 0.7, 1.5, 1.0, 0.05)
st.session_state.tts_manager.set_rate(voice_rate)

if st.sidebar.button("🔊 Test Voice", use_container_width=True):
    st.session_state.tts_manager.test_speech()

if st.sidebar.button("🔇 Mute" if not st.session_state.tts_manager.muted else "🔊 Unmute", use_container_width=True):
    st.session_state.tts_manager.toggle_mute()
    st.rerun()

st.sidebar.markdown("---")

# Data
save = st.sidebar.checkbox("Save detections", value=True)
st.session_state.save_detections = save

if st.sidebar.button("📊 Export CSV", use_container_width=True):
    df = detection_system.get_export_data()
    if not df.empty:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="detections.csv">Download</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    detection_system.clear_history()
    st.session_state.detection_count = 0
    st.session_state.detection_log = []
    st.sidebar.success("Cleared!")

st.sidebar.markdown("---")
st.sidebar.info("💡 Lower confidence = more detections")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("👁️ Object Detection for Visually Impaired")
st.markdown("*Real-time detection with voice feedback*")
st.markdown("---")

# Mode selection
mode = st.radio("Mode", ["📷 Webcam", "🖼️ Image"], horizontal=True)

# ============================================================================
# WEBCAM MODE
# ============================================================================
if mode == "📷 Webcam":
    if not detection_system.model:
        st.warning("⚠️ **Load model first!** Click 'Load Model' in sidebar")
    else:
        st.info("📹 Click **Start** → Allow camera → Point at objects")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            status = st.empty()
            transformer = VideoTransformer()
            transformer.detection_system = detection_system
            
            ctx = webrtc_streamer(
                key="detect",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=lambda: transformer,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            if ctx and ctx.state.playing:
                status.success("🎥 **Active**")
            else:
                status.info("⏸️ Click Start")
        
        with col2:
            st.subheader("Detections")
            for log in st.session_state.detection_log[:5]:
                st.warning(log)
            st.metric("Total", st.session_state.detection_count)

# ============================================================================
# IMAGE MODE
# ============================================================================
else:
    uploaded = st.file_uploader("Choose image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        if not detection_system.model:
            st.error("Load model first!")
        else:
            image = Image.open(uploaded)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_container_width=True)
            
            with st.spinner("Detecting..."):
                processed, detections = detection_system.process_image(image)
                
                if processed is not None:
                    with col2:
                        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), 
                                use_container_width=True)
                    
                    if detections:
                        st.success(f"Found {len(detections)} objects!")
                        df = pd.DataFrame([{k:v for k,v in d.items() if k != 'bbox'} 
                                          for d in detections])
                        st.dataframe(df, use_container_width=True)
                        
                        for d in detections:
                            if d.get('proximity') in ['close', 'medium']:
                                st.session_state.tts_manager.speak(f"{d['proximity']} {d['object']}")
                                time.sleep(0.3)
                    else:
                        st.warning("No objects detected")

# ============================================================================
# ANALYTICS
# ============================================================================
with st.expander("📊 Analytics"):
    if detection_system.detection_history:
        df = detection_system.get_export_data()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Objects", df['object'].nunique())
        col3.metric("Close", len(df[df['proximity'] == 'close']))
        
        if len(df) > 0:
            st.bar_chart(df['object'].value_counts().head(5))
    else:
        st.info("No data yet")
