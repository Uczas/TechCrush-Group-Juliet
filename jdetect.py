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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import torch

# ============================================================================
# FIX FOR PYTORCH 2.6+ COMPATIBILITY
# ============================================================================
# Monkey patch torch.load to use weights_only=False for YOLO compatibility
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Allow YOLO models to load by setting weights_only=False"""
    # Only apply to YOLO model loading
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load

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
    .stApp { background-color: #000000; }
    .stButton > button { background-color: #FFD700; color: #000000; font-size: 18px; font-weight: bold; padding: 12px 24px; border-radius: 10px; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #FFA500; transform: scale(1.02); }
    .detection-card { background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%); padding: 15px; border-radius: 12px; margin: 10px 0; border-left: 4px solid #FFD700; }
    .big-number { font-size: 48px; font-weight: bold; color: #FFD700; }
    .warning { color: #FF4444; font-weight: bold; }
    .success { color: #00FF00; font-weight: bold; }
    .info { color: #44AAFF; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3
CLOSE_THRESHOLD_PERCENT = 15
MEDIUM_THRESHOLD_PERCENT = 5
CENTER_ZONE_WIDTH_PERCENT = 40
COOLDOWN_SECONDS = 3
MAX_OBJECTS_PER_ANNOUNCEMENT = 3
MAX_HISTORY_SIZE = 500

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
# WEB SPEECH HANDLER (Browser-based)
# ============================================================================
class WebSpeechHandler:
    def __init__(self):
        self.last_alert_time = defaultdict(float)
        self.muted = False
        self.rate = 1.0
        
    def speak(self, text):
        """Send speech command to browser via JavaScript"""
        if not self.muted and text:
            # Escape special characters
            text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            js_code = f"""
            <script>
                (function() {{
                    if (window.speechSynthesis) {{
                        var utterance = new SpeechSynthesisUtterance('{text}');
                        utterance.rate = {self.rate};
                        window.speechSynthesis.cancel();
                        window.speechSynthesis.speak(utterance);
                    }}
                }})();
            </script>
            """
            st.components.v1.html(js_code, height=0, width=0)
    
    def can_announce(self, object_name, current_time):
        """Check cooldown"""
        last_time = self.last_alert_time[object_name]
        return (current_time - last_time) >= COOLDOWN_SECONDS
    
    def update_alert_time(self, object_name, current_time):
        """Update last announcement time"""
        self.last_alert_time[object_name] = current_time
    
    def toggle_mute(self):
        """Toggle mute"""
        self.muted = not self.muted
        if not self.muted:
            # Clear any pending speech
            st.components.v1.html("""
                <script>
                    if (window.speechSynthesis) {
                        window.speechSynthesis.cancel();
                    }
                </script>
            """, height=0, width=0)
        return not self.muted
    
    def set_rate(self, rate):
        """Set speech rate (0.5 to 2)"""
        self.rate = max(0.5, min(2.0, rate / 100))

# ============================================================================
# VIDEO TRANSFORMER FOR WEBCAM
# ============================================================================
class VideoTransformer(VideoTransformerBase):
    def __init__(self, detection_system, save_detections):
        self.detection_system = detection_system
        self.save_detections = save_detections
        self.frame_count = 0
        
    def recv(self, frame):
        """Receive and process frame from webcam"""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Skip frames for performance
            self.frame_count += 1
            if self.frame_count % FRAME_SKIP != 0:
                return frame
            
            # Process frame
            processed_img, detections = self.detection_system.process_frame(
                img, save_detection=self.save_detections
            )
            
            # Update session stats
            for det in detections:
                if det['proximity'] in ['close', 'medium distance']:
                    if 'detection_count' in st.session_state:
                        st.session_state.detection_count += 1
                    if 'detection_log' in st.session_state:
                        log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                        st.session_state.detection_log.insert(0, log_entry)
                        if len(st.session_state.detection_log) > 10:
                            st.session_state.detection_log.pop()
            
            # Convert back to frame
            return frame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

# ============================================================================
# DETECTION SYSTEM
# ============================================================================
class ObjectDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.model_loaded = False

    def load_model(self):
        """Load YOLO model with compatibility fixes"""
        if self.model_loaded:
            return True
            
        with st.spinner("Loading YOLO model (yolov8n.pt)..."):
            try:
                # Set environment variable for compatibility
                os.environ['TORCH_USE_RTLD_GLOBAL'] = '1'
                
                # Load the model
                self.model = YOLO('yolov8n.pt')
                
                # Warm up the model with a dummy frame
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, verbose=False)
                
                self.model_loaded = True
                st.success("✅ Model loaded successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.info("💡 Try restarting the app or check your internet connection.")
                return False

    def estimate_proximity(self, bbox_area, total_area):
        """Estimate object proximity based on area percentage"""
        area_percentage = (bbox_area / total_area) * 100
        if area_percentage >= CLOSE_THRESHOLD_PERCENT:
            proximity = "close"
        elif area_percentage >= MEDIUM_THRESHOLD_PERCENT:
            proximity = "medium distance"
        else:
            proximity = "far"
        return proximity, area_percentage

    def is_in_center_zone(self, bbox_center_x, frame_width):
        """Check if object is in the center zone of the frame"""
        center_threshold = (frame_width * CENTER_ZONE_WIDTH_PERCENT) / 200
        frame_center = frame_width / 2
        return abs(bbox_center_x - frame_center) <= center_threshold

    def process_frame(self, frame, save_detection=False):
        """Process a single frame and return annotated frame and detections"""
        if self.model is None:
            return frame, []

        total_area = frame.shape[0] * frame.shape[1]
        detected_obstacles = []
        detection_results = []

        # Resize frame for better performance if too large
        h, w = frame.shape[:2]
        scale_x = 1
        scale_y = 1
        
        if w > 640:
            scale = 640 / w
            new_w = 640
            new_h = int(h * scale)
            frame_small = cv2.resize(frame, (new_w, new_h))
            scale_x = w / new_w
            scale_y = h / new_h
        else:
            frame_small = frame

        # Run inference
        results = self.model(frame_small, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Scale back coordinates if resized
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    bbox_area = (x2 - x1) * (y2 - y1)
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    # Filter to important objects only
                    if class_name not in IMPORTANT_OBJECTS:
                        continue

                    # Check if in center zone
                    bbox_center_x = (x1 + x2) / 2
                    if not self.is_in_center_zone(bbox_center_x, w):
                        continue

                    # Estimate proximity
                    proximity, area_percent = self.estimate_proximity(bbox_area, total_area)

                    # Create detection record
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
                            # Limit history size
                            if len(self.detection_history) > MAX_HISTORY_SIZE:
                                self.detection_history = self.detection_history[-MAX_HISTORY_SIZE:]

                        # Draw bounding box
                        color = (0, 0, 255) if proximity == 'close' else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Add label
                        label = f"{class_name}: {confidence:.2f} ({proximity})"
                        cv2.putText(frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Announce detected obstacles
        if detected_obstacles and 'tts_handler' in st.session_state:
            current_time = time.time()
            filtered_obstacles = []
            
            for obj_name, proximity, area in detected_obstacles:
                if st.session_state.tts_handler.can_announce(obj_name, current_time):
                    filtered_obstacles.append((obj_name, proximity, area))
                    st.session_state.tts_handler.update_alert_time(obj_name, current_time)
            
            # Limit number of objects announced at once
            filtered_obstacles = filtered_obstacles[:MAX_OBJECTS_PER_ANNOUNCEMENT]
            
            if filtered_obstacles:
                # Create announcement message
                if len(filtered_obstacles) == 1:
                    obj_name, proximity, _ = filtered_obstacles[0]
                    message = f"{proximity} {obj_name} ahead"
                else:
                    parts = [f"{proximity} {obj_name}" for obj_name, proximity, _ in filtered_obstacles]
                    if len(parts) == 2:
                        message = f"{parts[0]} and {parts[1]} ahead"
                    else:
                        message = ", ".join(parts[:-1]) + f", and {parts[-1]} ahead"
                
                # Speak the announcement
                st.session_state.tts_handler.speak(message)

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
        if 'bbox' in df.columns:
            df = df.drop(columns=['bbox'])
        return df

    def clear_history(self):
        """Clear detection history"""
        self.detection_history = []

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'detection_system' not in st.session_state:
    st.session_state.detection_system = ObjectDetectionSystem()
if 'tts_handler' not in st.session_state:
    st.session_state.tts_handler = WebSpeechHandler()
if 'mode' not in st.session_state:
    st.session_state.mode = "webcam"
if 'save_detections' not in st.session_state:
    st.session_state.save_detections = True
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

detection_system = st.session_state.detection_system
tts_handler = st.session_state.tts_handler

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.title("🎛️ Controls")

# Model loading
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📦 Load YOLO Model", use_container_width=True):
        if detection_system.load_model():
            st.rerun()

with col2:
    if st.button("🔄 Clear History", use_container_width=True):
        detection_system.clear_history()
        st.session_state.detection_count = 0
        st.session_state.detection_log = []
        st.success("History cleared!")

# Show model status
if detection_system.model_loaded:
    st.sidebar.success("✅ Model Ready")
else:
    st.sidebar.warning("⚠️ Model Not Loaded")

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

# Voice settings
st.sidebar.subheader("🔊 Voice Settings")
voice_rate = st.sidebar.slider("Speech Rate", 50, 200, 100, 10)
tts_handler.set_rate(voice_rate)

# Mute toggle
if st.sidebar.button("🔇 Mute" if not tts_handler.muted else "🔊 Unmute", use_container_width=True):
    tts_handler.toggle_mute()
    st.rerun()

st.sidebar.markdown("---")

# Recording settings
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

st.sidebar.markdown("---")

# Instructions
st.sidebar.subheader("📖 Quick Guide")
st.sidebar.markdown("""
1. **Load Model** first
2. **Choose input mode**
3. **Adjust sensitivity**
4. **Listen to voice alerts**
5. **Export data** for analysis

💡 **Note:** Voice works automatically in your browser!
""")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("👁️ Real-time Object Detection for Visually Impaired")
st.markdown("*Enhanced with voice feedback, file upload, and data export*")
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

    with col2:
        st.subheader("📝 Recent Detections")
        log_placeholder = st.empty()
        st.markdown("---")
        st.subheader("📊 Session Stats")
        stats_placeholder = st.empty()

# ============================================================================
# WEBCAM MODE
# ============================================================================
if st.session_state.mode == "webcam":
    if not detection_system.model_loaded:
        st.warning("⚠️ Please load the YOLO model first using the button in the sidebar!")
    else:
        st.info("🎥 Webcam mode active. Click 'Start' to begin detection.")
        
        # WebRTC streamer
        webrtc_streamer(
            key="object-detection",
            video_transformer_factory=lambda: VideoTransformer(detection_system, st.session_state.save_detections),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Update stats display
        if st.session_state.detection_log:
            log_text = "\n".join(st.session_state.detection_log[:10])
            log_placeholder.markdown(f"```\n{log_text}\n```")
        
        stats_placeholder.metric("Total Detections", st.session_state.detection_count)

# ============================================================================
# IMAGE UPLOAD MODE
# ============================================================================
elif st.session_state.mode == "image":
    st.subheader("🖼️ Upload an Image for Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

    if uploaded_file:
        if not detection_system.model_loaded:
            st.error("⚠️ Please load the model first!")
        else:
            image = Image.open(uploaded_file)

            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with st.spinner("Detecting objects..."):
                processed_image, detections = detection_system.process_image(image)

                if processed_image is not None:
                    with col_img2:
                        st.subheader("Detection Results")
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, use_container_width=True, channels="RGB")

                    if detections:
                        st.success(f"✅ Found {len(detections)} objects!")

                        detection_data = []
                        for det in detections:
                            detection_data.append({
                                'Object': det['object'],
                                'Confidence': f"{det['confidence']:.1%}",
                                'Proximity': det['proximity'],
                                'Area %': det['area_percent']
                            })

                        st.dataframe(pd.DataFrame(detection_data), use_container_width=True)

                        # Announce detections
                        for det in detections:
                            if det['proximity'] in ['close', 'medium distance']:
                                tts_handler.speak(f"{det['proximity']} {det['object']} detected")
                                time.sleep(0.5)
                    else:
                        st.info("No important objects detected in this image")

# ============================================================================
# VIDEO UPLOAD MODE
# ============================================================================
elif st.session_state.mode == "video":
    st.subheader("🎬 Upload a Video for Detection")

    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv', 'webm'])

    if uploaded_file:
        if not detection_system.model_loaded:
            st.error("⚠️ Please load the model first!")
        else:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()

            # Open video
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.info(f"📹 Video loaded: {total_frames} frames at {fps} FPS")

            progress_bar = st.progress(0)
            video_placeholder = st.empty()
            detection_stats = []

            frame_count = 0
            processed_count = 0
            skip_frames = max(1, fps // 3)  # Process at ~3 FPS

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % skip_frames == 0:
                    processed_frame, detections = detection_system.process_frame(
                        frame, save_detection=st.session_state.save_detections
                    )

                    # Update progress
                    if total_frames > 0:
                        progress = processed_count / (total_frames / skip_frames)
                        progress_bar.progress(min(1.0, progress))

                    # Display frame
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

                    # Record detections
                    for det in detections:
                        if det['proximity'] in ['close', 'medium distance']:
                            detection_stats.append(det)
                            st.session_state.detection_count += 1

                    processed_count += 1

                frame_count += 1

            cap.release()
            os.unlink(tfile.name)
            
            st.success(f"✅ Video processing complete!")
            st.metric("Total Detections", len(detection_stats))

            if detection_stats:
                st.subheader("Detection Timeline")
                df_stats = pd.DataFrame(detection_stats)
                if 'bbox' in df_stats.columns:
                    df_stats = df_stats.drop(columns=['bbox'])
                st.dataframe(df_stats, use_container_width=True)

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

        # Metrics row
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

        st.divider()

        # Charts
        st.subheader("Object Distribution (Top 10)")
        object_counts = df['object'].value_counts().head(10)
        st.bar_chart(object_counts)

        st.subheader("Proximity Distribution")
        proximity_counts = df['proximity'].value_counts()
        st.bar_chart(proximity_counts)

        # Timeline if timestamp available
        if 'timestamp' in df.columns:
            st.subheader("Detection Timeline")
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            df_time = df.set_index('timestamp_dt').resample('1S').size()
            if not df_time.empty:
                st.line_chart(df_time)

        # Export report
        st.subheader("Export Analytics Report")
        if st.button("📑 Generate Analytics Report", use_container_width=True):
            report = {
                'summary': {
                    'total_detections': len(df),
                    'unique_objects': int(df['object'].nunique()),
                    'close_detections': int(len(df[df['proximity'] == 'close'])),
                    'medium_detections': int(len(df[df['proximity'] == 'medium distance'])),
                    'far_detections': int(len(df[df['proximity'] == 'far'])),
                    'average_confidence': float(df['confidence'].mean()),
                    'top_object': str(df['object'].value_counts().index[0]) if not df.empty else None,
                    'top_count': int(df['object'].value_counts().iloc[0]) if not df.empty else 0
                },
                'object_counts': df['object'].value_counts().to_dict(),
                'proximity_counts': df['proximity'].value_counts().to_dict(),
                'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            report_json = json.dumps(report, indent=2)
            b64 = base64.b64encode(report_json.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("Report generated!")
    else:
        st.info("No data available. Run detection to see analytics.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🎯 <strong>How it works:</strong> YOLOv8 detects objects → Filters important obstacles → Checks if directly ahead → Announces close/medium objects via voice</p>
    <p>💡 <strong>Tips:</strong> Adjust thresholds in sidebar | Export data for analysis | Supports images, videos, and webcam</p>
    <p>🔊 <strong>Voice feedback helps blind/low-vision users navigate safely</strong></p>
</div>
""", unsafe_allow_html=True)
