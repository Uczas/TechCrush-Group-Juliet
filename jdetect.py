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
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main { background-color: #000000; }
    .stApp { background-color: #000000; }
    .stButton > button {
        background-color: #FFD700;
        color: #000000;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FFA500;
        transform: scale(1.02);
    }
    .detection-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #FFD700;
    }
    .stat-card {
        background: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .big-number {
        font-size: 48px;
        font-weight: bold;
        color: #FFD700;
    }
    .warning { color: #FF4444; font-weight: bold; }
    .success { color: #00FF00; font-weight: bold; }
    .info { color: #44AAFF; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 2
CLOSE_THRESHOLD_PERCENT = 15
MEDIUM_THRESHOLD_PERCENT = 5
CENTER_ZONE_WIDTH_PERCENT = 40
COOLDOWN_SECONDS = 3
MAX_OBJECTS_PER_ANNOUNCEMENT = 3

# Voice settings
VOICE_RATE = 1.0  # Browser TTS uses 0.5-2.0 scale
VOICE_PITCH = 1.0

# Important objects
IMPORTANT_OBJECTS = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'train',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'dog', 'cat',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'keyboard', 'cell phone', 'oven', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
}

# ============================================================================
# BROWSER-BASED TTS MANAGER (Cloud Compatible)
# ============================================================================
class BrowserTTSManager:
    def __init__(self):
        self.muted = False
        self.last_alert_time = defaultdict(float)
        self.rate = 1.0
        self.pitch = 1.0
        self.available = True
        
    def speak(self, text):
        """Send text to browser for speech using JavaScript Web Speech API"""
        if not self.muted and text:
            # Escape special characters for JavaScript
            text_escaped = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            # Inject JavaScript to speak the text
            js_code = f"""
            <script>
                if ('speechSynthesis' in window) {{
                    // Cancel any ongoing speech
                    window.speechSynthesis.cancel();
                    
                    // Create new utterance
                    var utterance = new SpeechSynthesisUtterance('{text_escaped}');
                    utterance.rate = {self.rate};
                    utterance.pitch = {self.pitch};
                    
                    // Optional: Get available voices and try to use a good one
                    if (window.speechSynthesis.getVoices) {{
                        var voices = window.speechSynthesis.getVoices();
                        // Try to find a female voice or Google voice if available
                        var preferredVoice = voices.find(v => v.name.includes('Google') || v.name.includes('Female'));
                        if (preferredVoice) {{
                            utterance.voice = preferredVoice;
                        }}
                    }}
                    
                    // Speak
                    window.speechSynthesis.speak(utterance);
                }} else {{
                    console.warn('Web Speech API not supported in this browser');
                }}
            </script>
            """
            st.components.v1.html(js_code, height=0, width=0)
    
    def can_announce(self, object_name, current_time):
        """Check if enough time has passed since last announcement"""
        last_time = self.last_alert_time[object_name]
        return (current_time - last_time) >= COOLDOWN_SECONDS
    
    def update_alert_time(self, object_name, current_time):
        """Update last announcement time"""
        self.last_alert_time[object_name] = current_time
    
    def toggle_mute(self):
        """Toggle mute state"""
        self.muted = not self.muted
        if not self.muted:
            # Cancel any ongoing speech
            st.components.v1.html("""
                <script>
                    if ('speechSynthesis' in window) {
                        window.speechSynthesis.cancel();
                    }
                </script>
            """, height=0, width=0)
        return not self.muted
    
    def set_rate(self, rate):
        """Set speech rate (expects 0.5 to 2.0)"""
        self.rate = max(0.5, min(2.0, rate))
    
    def set_pitch(self, pitch):
        """Set speech pitch (0.5 to 2.0)"""
        self.pitch = max(0.5, min(2.0, pitch))
    
    def test_speech(self):
        """Test the speech system"""
        self.speak("Voice system is ready. You can hear this message.")

# Initialize TTS once in session state
if 'tts_manager' not in st.session_state:
    st.session_state.tts_manager = BrowserTTSManager()

# ============================================================================
# DETECTION SYSTEM
# ============================================================================
class ObjectDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.frame_count = 0
        self.last_announcements = {}

    def load_model(self):
        """Load YOLO model"""
        with st.spinner("Loading YOLO model (yolov8n.pt)..."):
            try:
                self.model = YOLO('yolov8n.pt')
                return True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return False

    def estimate_proximity(self, bbox_area, total_area):
        area_percentage = (bbox_area / total_area) * 100
        if area_percentage >= CLOSE_THRESHOLD_PERCENT:
            proximity = "close"
        elif area_percentage >= MEDIUM_THRESHOLD_PERCENT:
            proximity = "medium distance"
        else:
            proximity = "far"
        return proximity, area_percentage

    def is_in_center_zone(self, bbox_center_x, frame_width):
        center_threshold = (frame_width * CENTER_ZONE_WIDTH_PERCENT) / 200
        frame_center = frame_width / 2
        return abs(bbox_center_x - frame_center) <= center_threshold

    def process_frame(self, frame, save_detection=False):
        """Process frame and return results"""
        if self.model is None:
            return frame, []

        self.frame_count += 1
        total_area = frame.shape[0] * frame.shape[1]
        detected_obstacles = []
        detection_results = []

        # Skip frames for performance
        if self.frame_count % FRAME_SKIP != 0:
            return frame, []

        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
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

                    detection_info = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'object': class_name,
                        'confidence': confidence,
                        'proximity': proximity,
                        'area_percent': round(area_percent, 2),
                        'bbox': (x1, y1, x2, y2)
                    }

                    detection_results.append(detection_info)

                    if proximity in ['close', 'medium distance']:
                        detected_obstacles.append((class_name, proximity, area_percent))

                        if save_detection:
                            self.detection_history.append(detection_info)

                        # Draw bounding box
                        color = (0, 0, 255) if proximity == 'close' else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        label = f"{class_name}: {confidence:.2f} ({proximity})"
                        cv2.putText(frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Announce obstacles
        if detected_obstacles:
            current_time = time.time()
            filtered_obstacles = []
            
            for obj_name, proximity, area in detected_obstacles:
                if st.session_state.tts_manager.can_announce(obj_name, current_time):
                    filtered_obstacles.append((obj_name, proximity, area))
                    st.session_state.tts_manager.update_alert_time(obj_name, current_time)
            
            filtered_obstacles = filtered_obstacles[:MAX_OBJECTS_PER_ANNOUNCEMENT]
            
            if filtered_obstacles:
                if len(filtered_obstacles) == 1:
                    obj_name, proximity, _ = filtered_obstacles[0]
                    message = f"{proximity} {obj_name} ahead"
                else:
                    parts = [f"{proximity} {obj_name}" for obj_name, proximity, _ in filtered_obstacles]
                    if len(parts) == 2:
                        message = f"{parts[0]} and {parts[1]} ahead"
                    else:
                        message = ", ".join(parts[:-1]) + f", and {parts[-1]} ahead"
                
                # Speak using browser TTS
                st.session_state.tts_manager.speak(message)

        return frame, detection_results

    def process_image(self, image):
        """Process a single image"""
        if self.model is None:
            return None, []

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.process_frame(image, save_detection=True)

    def get_export_data(self):
        """Get detection history for export"""
        if not self.detection_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.detection_history)
        df_export = df.drop(columns=['bbox'] if 'bbox' in df.columns else [])
        return df_export

    def clear_history(self):
        """Clear detection history"""
        self.detection_history = []
        self.frame_count = 0

# ============================================================================
# WEBRTC VIDEO TRANSFORMER
# ============================================================================
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detection_system = None
        self.detection_count = 0
        self.detection_log = []
        
    def recv(self, frame):
        """Process each frame from webcam"""
        if self.detection_system is None or self.detection_system.model is None:
            return frame
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame
        processed_img, detections = self.detection_system.process_frame(
            img, save_detection=st.session_state.save_detections
        )
        
        # Update session state for UI
        for det in detections:
            if det['proximity'] in ['close', 'medium distance']:
                self.detection_count += 1
                log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                self.detection_log.insert(0, log_entry)
                
                if len(self.detection_log) > 10:
                    self.detection_log.pop()
        
        # Update session state
        st.session_state.detection_count = self.detection_count
        st.session_state.detection_log = self.detection_log
        
        # Return processed frame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ============================================================================
# SIDEBAR - ENHANCED CONTROLS
# ============================================================================
st.sidebar.title("🎛️ Controls")

# Initialize session state
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
if 'webrtc_ctx' not in st.session_state:
    st.session_state.webrtc_ctx = None

detection_system = st.session_state.detection_system

# Model loading
if st.sidebar.button("📦 Load YOLO Model", use_container_width=True):
    if detection_system.load_model():
        st.sidebar.success("✅ Model ready!")
        # Test voice system
        st.session_state.tts_manager.test_speech()
    else:
        st.sidebar.error("❌ Failed to load")

st.sidebar.markdown("---")

# Mode selection
st.sidebar.subheader("🎥 Input Mode")
mode = st.sidebar.radio(
    "Select mode:",
    ["📷 Webcam", "🖼️ Image Upload", "🎬 Video Upload"],
    index=0
)

if mode == "📷 Webcam":
    st.session_state.mode = "webcam"
elif mode == "🖼️ Image Upload":
    st.session_state.mode = "image"
else:
    st.session_state.mode = "video"

st.sidebar.markdown("---")

# Detection settings
st.sidebar.subheader("⚙️ Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.3, 0.9, 0.5, 0.05)
close_threshold = st.sidebar.slider("Close Threshold (%)", 5, 30, 15, 5)
medium_threshold = st.sidebar.slider("Medium Threshold (%)", 1, 20, 5, 1)
cooldown = st.sidebar.slider("Announcement Cooldown (s)", 1, 10, 3, 1)

# Update globals
globals()['CONFIDENCE_THRESHOLD'] = confidence
globals()['CLOSE_THRESHOLD_PERCENT'] = close_threshold
globals()['MEDIUM_THRESHOLD_PERCENT'] = medium_threshold
globals()['COOLDOWN_SECONDS'] = cooldown

st.sidebar.markdown("---")

# Voice customization
st.sidebar.subheader("🔊 Voice Settings")

# Show TTS status
st.sidebar.success("✅ Browser Speech ready!")

# Rate and pitch (browser TTS uses different scales)
voice_rate = st.sidebar.slider("Speech Rate", 0.5, 2.0, VOICE_RATE, 0.05)
voice_pitch = st.sidebar.slider("Speech Pitch", 0.5, 2.0, VOICE_PITCH, 0.05)
st.session_state.tts_manager.set_rate(voice_rate)
st.session_state.tts_manager.set_pitch(voice_pitch)

# Test voice button
if st.sidebar.button("🔊 Test Voice", use_container_width=True):
    st.session_state.tts_manager.test_speech()
    st.sidebar.success("Check your speakers! Voice should speak.")

# Mute toggle
current_status = not st.session_state.tts_manager.muted
if st.sidebar.button("🔇 Mute" if current_status else "🔊 Unmute", use_container_width=True):
    st.session_state.tts_manager.toggle_mute()
    st.rerun()

st.sidebar.markdown("---")

# Save settings
st.sidebar.subheader("💾 Recording")
save_detections = st.sidebar.checkbox("Save detections to history", value=True)
st.session_state.save_detections = save_detections

# Export section
st.sidebar.subheader("📤 Export Data")
if st.sidebar.button("📊 Export as CSV", use_container_width=True):
    df = detection_system.get_export_data()
    if not df.empty:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="detection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.success(f"Exported {len(df)} detections!")
    else:
        st.sidebar.warning("No detections to export")

if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    detection_system.clear_history()
    st.session_state.detection_count = 0
    st.session_state.detection_log = []
    st.sidebar.success("History cleared!")

st.sidebar.markdown("---")

# Instructions
st.sidebar.subheader("📖 Quick Guide")
st.sidebar.markdown("""
1. **Load Model** first
2. **Choose input mode**
3. **Allow camera permission** when prompted
4. **Adjust sensitivity**
5. **Listen to voice alerts**

💡 **Note:** Voice works automatically in your browser!
""")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("👁️ Real-time Object Detection for Visually Impaired")
st.markdown("*Enhanced with browser-based voice feedback - No installation needed!*")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "📊 Detection Log", "📈 Analytics"])

# ============================================================================
# TAB 1: LIVE DETECTION
# ============================================================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Feed")
        feed_placeholder = st.empty()
        status_placeholder = st.empty()

    with col2:
        st.subheader("📝 Recent Detections")
        log_placeholder = st.empty()
        st.markdown("---")
        st.subheader("📊 Session Stats")
        stats_placeholder = st.empty()

# ============================================================================
# WEBCAM MODE (Using streamlit-webrtc)
# ============================================================================
if st.session_state.mode == "webcam":
    if detection_system.model is None:
        st.warning("⚠️ Please load the YOLO model first using the button in the sidebar!")
    else:
        st.info("📹 **Webcam Instructions:** Click 'Start' below, then allow camera access when prompted by your browser.")
        
        # Create video transformer with detection system
        transformer = VideoTransformer()
        transformer.detection_system = detection_system
        
        # Start webrtc stream
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: transformer,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Display stats and logs
        if webrtc_ctx.state.playing:
            status_placeholder.success("🎥 Camera active - Detection running")
            
            # Update detection log display
            if st.session_state.detection_log:
                log_text = "\n".join(st.session_state.detection_log)
                log_placeholder.markdown(f"```\n{log_text}\n```")
            else:
                log_placeholder.info("Waiting for detections...")
            
            # Update stats
            stats_placeholder.metric("Total Detections", st.session_state.detection_count)
        else:
            status_placeholder.info("⏸️ Camera stopped. Click 'Start' to begin detection.")
                
# ============================================================================
# IMAGE UPLOAD MODE
# ============================================================================
elif st.session_state.mode == "image":
    st.subheader("🖼️ Upload an Image for Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file:
        if detection_system.model is None:
            st.error("⚠️ Please load the model first!")
        else:
            image = Image.open(uploaded_file)

            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.subheader("Original Image")
                st.image(image)

            with st.spinner("Detecting objects..."):
                processed_image, detections = detection_system.process_image(image)

                if processed_image is not None:
                    with col_img2:
                        st.subheader("Detection Results")
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, channels="RGB")

                    if detections:
                        st.success(f"✅ Found {len(detections)} objects!")
                        
                        # Update detection count for image mode
                        for det in detections:
                            if det['proximity'] in ['close', 'medium distance']:
                                st.session_state.detection_count += 1
                                log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                                st.session_state.detection_log.insert(0, log_entry)

                        detection_data = []
                        for det in detections:
                            detection_data.append({
                                'Object': det['object'],
                                'Confidence': f"{det['confidence']:.1%}",
                                'Proximity': det['proximity'],
                                'Area %': det['area_percent']
                            })

                        st.table(pd.DataFrame(detection_data))

                        for det in detections:
                            if det['proximity'] in ['close', 'medium distance']:
                                st.session_state.tts_manager.speak(
                                    f"{det['proximity']} {det['object']} detected"
                                )
                                time.sleep(1)
                    else:
                        st.info("No important objects detected in this image")

# ============================================================================
# VIDEO UPLOAD MODE
# ============================================================================
elif st.session_state.mode == "video":
    st.subheader("🎬 Upload a Video for Detection")

    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file:
        if detection_system.model is None:
            st.error("⚠️ Please load the model first!")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.info(f"Video loaded: {total_frames} frames at {fps} FPS")

            progress_bar = st.progress(0)
            video_placeholder = st.empty()
            detection_stats = []

            frame_count = 0
            processed_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % (fps // 5) == 0:
                    processed_frame, detections = detection_system.process_frame(
                        frame, save_detection=st.session_state.save_detections
                    )

                    if total_frames > 0:
                        progress = processed_count / total_frames
                        progress_bar.progress(progress)

                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(processed_frame_rgb, channels="RGB")

                    for det in detections:
                        if det['proximity'] in ['close', 'medium distance']:
                            detection_stats.append(det)
                            st.session_state.detection_count += 1
                            log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                            st.session_state.detection_log.insert(0, log_entry)

                    processed_count += 1
                    time.sleep(0.033)

                frame_count += 1

            cap.release()
            st.success(f"✅ Video processing complete!")
            st.metric("Total Detections", len(detection_stats))

            if detection_stats:
                st.subheader("Detection Timeline")
                df_stats = pd.DataFrame(detection_stats)
                if 'bbox' in df_stats.columns:
                    df_stats = df_stats.drop(columns=['bbox'])
                st.dataframe(df_stats)

            os.unlink(tfile.name)

# ============================================================================
# TAB 2: DETECTION LOG
# ============================================================================
with tab2:
    st.subheader("📝 Complete Detection Log")

    if detection_system.detection_history:
        df_log = detection_system.get_export_data()
        st.dataframe(df_log, use_container_width=True)

        st.subheader("🔍 Filter Detections")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            object_filter = st.multiselect(
                "Filter by object type",
                options=sorted(df_log['object'].unique())
            )

        with col_f2:
            proximity_filter = st.multiselect(
                "Filter by proximity",
                options=sorted(df_log['proximity'].unique())
            )

        filtered_df = df_log.copy()
        if object_filter:
            filtered_df = filtered_df[filtered_df['object'].isin(object_filter)]
        if proximity_filter:
            filtered_df = filtered_df[filtered_df['proximity'].isin(proximity_filter)]

        if not filtered_df.empty:
            st.write(f"Showing {len(filtered_df)} detections")
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("No matching detections")
    else:
        st.info("No detections recorded yet. Run detection with 'Save detections' enabled.")

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================
with tab3:
    st.subheader("📊 Detection Analytics")

    if detection_system.detection_history:
        df = detection_system.get_export_data()

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        with col_a1:
            st.metric("Total Detections", len(df))
        with col_a2:
            st.metric("Unique Objects", df['object'].nunique())
        with col_a3:
            close_count = len(df[df['proximity'] == 'close'])
            st.metric("Close Objects", close_count)
        with col_a4:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

        st.subheader("Object Distribution")
        object_counts = df['object'].value_counts().head(10)
        st.bar_chart(object_counts)

        st.subheader("Proximity Distribution")
        proximity_counts = df['proximity'].value_counts()
        st.bar_chart(proximity_counts)

        if 'timestamp' in df.columns:
            st.subheader("Detection Timeline")
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            df_time = df.set_index('timestamp_dt').resample('1S').size()
            st.line_chart(df_time)

        st.subheader("Export Analytics Report")
        if st.button("Generate Analytics Report"):
            report = {
                'summary': {
                    'total_detections': len(df),
                    'unique_objects': df['object'].nunique(),
                    'close_detections': len(df[df['proximity'] == 'close']),
                    'medium_detections': len(df[df['proximity'] == 'medium distance']),
                    'average_confidence': float(df['confidence'].mean()),
                    'top_object': df['object'].value_counts().index[0] if not df.empty else None
                },
                'object_counts': df['object'].value_counts().to_dict(),
                'proximity_counts': df['proximity'].value_counts().to_dict(),
                'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            report_json = json.dumps(report, indent=2)
            b64 = base64.b64encode(report_json.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No data available. Run detection to see analytics.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🎯 <strong>How it works:</strong> YOLOv8 detects objects → Filters important obstacles → Checks if directly ahead → Announces close/medium objects via voice</p>
    <p>💡 <strong>Tips:</strong> Adjust thresholds in sidebar | Export data for analysis | Supports images and videos</p>
    <p>🔊 <strong>Voice feedback helps blind/low-vision users navigate safely</strong> - Works in any modern browser!</p>
    <p>📹 <strong>Webcam:</strong> Click Start and allow camera access when prompted</p>
</div>
""", unsafe_allow_html=True)