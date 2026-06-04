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
# CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 2
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
# WEB SPEECH HANDLER
# ============================================================================
class WebSpeechHandler:
    def __init__(self):
        self.last_alert_time = defaultdict(float)
        self.muted = False
        self.rate = 1.0
        
    def speak(self, text):
        if not self.muted and text:
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
        last_time = self.last_alert_time[object_name]
        return (current_time - last_time) >= COOLDOWN_SECONDS
    
    def update_alert_time(self, object_name, current_time):
        self.last_alert_time[object_name] = current_time
    
    def toggle_mute(self):
        self.muted = not self.muted
        return not self.muted
    
    def set_rate(self, rate):
        self.rate = max(0.5, min(2.0, rate / 100))

# ============================================================================
# DETECTION SYSTEM
# ============================================================================
class ObjectDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.model_loaded = False

    def load_model(self):
        if self.model_loaded:
            return True
            
        with st.spinner("Loading YOLO model..."):
            try:
                self.model = YOLO('yolov8n.pt')
                # Warm up model
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, verbose=False)
                self.model_loaded = True
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
        if self.model is None:
            return frame, []

        total_area = frame.shape[0] * frame.shape[1]
        detected_obstacles = []
        detection_results = []

        # Resize for performance
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w = 640
            new_h = int(h * scale)
            frame_small = cv2.resize(frame, (new_w, new_h))
            scale_x = w / new_w
            scale_y = h / new_h
        else:
            frame_small = frame
            scale_x = scale_y = 1

        results = self.model(frame_small, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    bbox_area = (x2 - x1) * (y2 - y1)
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    if class_name not in IMPORTANT_OBJECTS:
                        continue

                    bbox_center_x = (x1 + x2) / 2
                    if not self.is_in_center_zone(bbox_center_x, w):
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
                            if len(self.detection_history) > MAX_HISTORY_SIZE:
                                self.detection_history = self.detection_history[-MAX_HISTORY_SIZE:]

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
                if st.session_state.tts_handler.can_announce(obj_name, current_time):
                    filtered_obstacles.append((obj_name, proximity, area))
                    st.session_state.tts_handler.update_alert_time(obj_name, current_time)
            
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
                
                st.session_state.tts_handler.speak(message)

        return frame, detection_results

    def process_image(self, image):
        if self.model is None:
            return None, []

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.process_frame(image, save_detection=True)

    def get_export_data(self):
        if not self.detection_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.detection_history)
        if 'bbox' in df.columns:
            df = df.drop(columns=['bbox'])
        return df

    def clear_history(self):
        self.detection_history = []

# ============================================================================
# CAMERA PROCESSING FUNCTION
# ============================================================================
def process_camera_frame(frame_bytes, detection_system):
    """Process a single camera frame"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame
    processed_frame, detections = detection_system.process_frame(
        frame, save_detection=st.session_state.save_detections
    )
    
    # Update detection log
    for det in detections:
        if det['proximity'] in ['close', 'medium distance']:
            st.session_state.detection_count += 1
            log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
            st.session_state.detection_log.insert(0, log_entry)
            if len(st.session_state.detection_log) > 10:
                st.session_state.detection_log.pop()
    
    # Convert back to bytes
    _, buffer = cv2.imencode('.jpg', processed_frame)
    return buffer.tobytes(), detections

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'detection_system' not in st.session_state:
    st.session_state.detection_system = ObjectDetectionSystem()
if 'tts_handler' not in st.session_state:
    st.session_state.tts_handler = WebSpeechHandler()
if 'save_detections' not in st.session_state:
    st.session_state.save_detections = True
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

detection_system = st.session_state.detection_system
tts_handler = st.session_state.tts_handler

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .stApp { background-color: #000000; }
    .stButton > button { background-color: #FFD700; color: #000000; font-size: 18px; font-weight: bold; padding: 12px 24px; border-radius: 10px; }
    .stButton > button:hover { background-color: #FFA500; transform: scale(1.02); }
    .css-1v0mbdj { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.title("🎛️ Controls")

if st.sidebar.button("📦 Load YOLO Model", use_container_width=True):
    if detection_system.load_model():
        st.sidebar.success("✅ Model ready!")
        st.rerun()

if detection_system.model_loaded:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.warning("⚠️ Click 'Load Model'")

st.sidebar.markdown("---")

# Mode selection
st.sidebar.subheader("🎥 Input Mode")
mode = st.sidebar.radio(
    "Select mode:",
    ["📷 Live Camera", "🖼️ Image Upload", "🎬 Video Upload"],
    index=0
)

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

if st.sidebar.button("🔇 Mute" if not tts_handler.muted else "🔊 Unmute", use_container_width=True):
    tts_handler.toggle_mute()
    st.rerun()

st.sidebar.markdown("---")

# Save settings
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

if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    detection_system.clear_history()
    st.session_state.detection_count = 0
    st.session_state.detection_log = []
    st.sidebar.success("Cleared!")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Instructions:**\n1. Load model first\n2. Select input mode\n3. For camera, click 'Start Camera'\n4. Voice feedback plays automatically")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("👁️ Object Detection for Visually Impaired")
st.markdown("*Real-time detection with voice feedback for accessibility*")
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
# LIVE CAMERA MODE (Using Streamlit's built-in camera)
# ============================================================================
if mode == "📷 Live Camera":
    if not detection_system.model_loaded:
        st.error("⚠️ Please load the YOLO model first using the button in the sidebar!")
    else:
        st.info("📷 **Live Camera Mode** - Click 'Start Camera' to begin real-time detection")
        
        # Camera control buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            start_camera = st.button("▶️ Start Camera", use_container_width=True)
        with col_stop:
            stop_camera = st.button("⏹️ Stop Camera", use_container_width=True)
        
        if start_camera:
            st.session_state.camera_running = True
        if stop_camera:
            st.session_state.camera_running = False
        
        if st.session_state.camera_running:
            # Use Streamlit's built-in camera input
            camera_image = st.camera_input("Take a picture", key="camera")
            
            if camera_image is not None:
                # Process the image
                with st.spinner("Detecting..."):
                    # Convert camera image to PIL
                    image = Image.open(camera_image)
                    
                    # Process frame
                    processed_image, detections = detection_system.process_image(image)
                    
                    if processed_image is not None:
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        feed_placeholder.image(processed_image_rgb, channels="RGB", use_column_width=True)
                        
                        # Update detection log
                        for det in detections:
                            if det['proximity'] in ['close', 'medium distance']:
                                st.session_state.detection_count += 1
                                log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                                st.session_state.detection_log.insert(0, log_entry)
                                if len(st.session_state.detection_log) > 10:
                                    st.session_state.detection_log.pop()
                                
                                # Speak detection
                                tts_handler.speak(f"{det['proximity']} {det['object']} detected")
                        
                        # Auto-refresh for continuous detection
                        time.sleep(0.5)
                        st.rerun()
            else:
                feed_placeholder.info("👆 Click 'Take a picture' to capture and detect objects")
        
        # Display stats
        if st.session_state.detection_log:
            log_text = "\n".join(st.session_state.detection_log[:10])
            log_placeholder.markdown(f"```\n{log_text}\n```")
        
        stats_placeholder.metric("Total Detections", st.session_state.detection_count)

# ============================================================================
# IMAGE UPLOAD MODE
# ============================================================================
elif mode == "🖼️ Image Upload":
    st.subheader("🖼️ Upload an Image for Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file:
        if not detection_system.model_loaded:
            st.error("⚠️ Please load the model first!")
        else:
            image = Image.open(uploaded_file)

            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            with st.spinner("Detecting objects..."):
                processed_image, detections = detection_system.process_image(image)

                if processed_image is not None:
                    with col_img2:
                        st.subheader("Detection Results")
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, use_column_width=True, channels="RGB")

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
                            
                            # Update log
                            if det['proximity'] in ['close', 'medium distance']:
                                st.session_state.detection_count += 1
                                log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                                st.session_state.detection_log.insert(0, log_entry)
                                if len(st.session_state.detection_log) > 10:
                                    st.session_state.detection_log.pop()

                        st.dataframe(pd.DataFrame(detection_data), use_container_width=True)

                        # Speak detections
                        for det in detections:
                            if det['proximity'] in ['close', 'medium distance']:
                                tts_handler.speak(f"{det['proximity']} {det['object']} detected")
                                time.sleep(0.5)
                    else:
                        st.info("No important objects detected in this image")

# ============================================================================
# VIDEO UPLOAD MODE
# ============================================================================
elif mode == "🎬 Video Upload":
    st.subheader("🎬 Upload a Video for Detection")

    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file:
        if not detection_system.model_loaded:
            st.error("⚠️ Please load the model first!")
        else:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()

            # Process video
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"📹 Processing video: {total_frames} frames at {fps} FPS")
            
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
                    video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                    
                    for det in detections:
                        if det['proximity'] in ['close', 'medium distance']:
                            detection_stats.append(det)
                            st.session_state.detection_count += 1
                            log_entry = f"⚠️ {det['object'].upper()} - {det['proximity']} ({det['confidence']:.0%})"
                            st.session_state.detection_log.insert(0, log_entry)
                            if len(st.session_state.detection_log) > 10:
                                st.session_state.detection_log.pop()
                    
                    processed_count += 1
                    
                frame_count += 1
                
            cap.release()
            os.unlink(tfile.name)
            
            st.success(f"✅ Processing complete! Found {len(detection_stats)} objects")
            st.metric("Total Detections", len(detection_stats))
            
            if detection_stats:
                df_stats = pd.DataFrame(detection_stats)
                if 'bbox' in df_stats.columns:
                    df_stats = df_stats.drop(columns=['bbox'])
                st.dataframe(df_stats, use_container_width=True)

# ============================================================================
# Update stats display for all modes
# ============================================================================
if mode != "📷 Live Camera" or not st.session_state.camera_running:
    if st.session_state.detection_log:
        log_text = "\n".join(st.session_state.detection_log[:10])
        with tab1:
            log_placeholder.markdown(f"```\n{log_text}\n```")
    
    with tab1:
        stats_placeholder.metric("Total Detections", st.session_state.detection_count)

# ============================================================================
# TAB 2: DETECTION LOG
# ============================================================================
with tab2:
    if detection_system.detection_history:
        df_log = detection_system.get_export_data()
        st.dataframe(df_log, use_container_width=True)
        
        # Filter options
        st.subheader("🔍 Filter Detections")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if 'object' in df_log.columns:
                object_filter = st.multiselect("Object Type", options=sorted(df_log['object'].unique()))
        with col_f2:
            if 'proximity' in df_log.columns:
                proximity_filter = st.multiselect("Proximity", options=sorted(df_log['proximity'].unique()))
        
        if object_filter or proximity_filter:
            filtered_df = df_log.copy()
            if object_filter:
                filtered_df = filtered_df[filtered_df['object'].isin(object_filter)]
            if proximity_filter:
                filtered_df = filtered_df[filtered_df['proximity'].isin(proximity_filter)]
            st.dataframe(filtered_df, use_container_width=True)
    else:
        st.info("No detections yet. Use Live Camera, Image Upload, or Video Upload to get started!")

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================
with tab3:
    if detection_system.detection_history:
        df = detection_system.get_export_data()
        
        # Metrics
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            st.metric("Total Detections", len(df))
        with col_a2:
            st.metric("Unique Objects", df['object'].nunique())
        with col_a3:
            st.metric("Close Objects", len(df[df['proximity'] == 'close']))
        with col_a4:
            st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
        
        # Charts
        st.subheader("Top Detected Objects")
        if 'object' in df.columns:
            st.bar_chart(df['object'].value_counts().head(10))
        
        st.subheader("Proximity Distribution")
        if 'proximity' in df.columns:
            st.bar_chart(df['proximity'].value_counts())
        
        # Timeline if timestamp available
        if 'timestamp' in df.columns:
            st.subheader("Detection Timeline")
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            df_time = df.set_index('timestamp_dt').resample('1S').size()
            if not df_time.empty:
                st.line_chart(df_time)
        
        # Export report
        st.subheader("Export Analytics Report")
        if st.button("📑 Generate JSON Report", use_container_width=True):
            report = {
                'summary': {
                    'total_detections': len(df),
                    'unique_objects': int(df['object'].nunique()),
                    'close_detections': int(len(df[df['proximity'] == 'close'])),
                    'medium_detections': int(len(df[df['proximity'] == 'medium distance'])),
                    'average_confidence': float(df['confidence'].mean()),
                    'top_object': str(df['object'].value_counts().index[0]) if not df.empty else None
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
        st.info("No data available. Run detection to see analytics!")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🎯 <strong>How it works:</strong> YOLOv8 detects objects → Filters important obstacles → Checks if directly ahead → Announces close/medium objects via voice</p>
    <p>💡 <strong>Tips:</strong> Adjust thresholds in sidebar | Export data for analysis | Works with camera, images, and videos</p>
    <p>🔊 <strong>Voice feedback helps blind/low-vision users navigate safely</strong></p>
</div>
""", unsafe_allow_html=True)
