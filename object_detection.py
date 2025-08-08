import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import av
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------- CONFIG -----------
st.set_page_config(page_title="YOLOv8 - Webcam/Image/Video with gTTS TTS", layout="wide")
st.title("🧠 YOLOv8 Object Detection — Say only on click (gTTS TTS)")

# ----------- Load model once -----------
@st.cache_resource
def load_model(path="yolov8n.pt"):
    return YOLO(path)

model = load_model()
COCO_CLASSES = model.names

# ----------- Text to Speech using gTTS -----------
def speak_browser(text: str):
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            st.audio(tmp.name, format="audio/mp3", autoplay=True)
            # Cleanup tmp file after playback
            # We do not delete immediately as st.audio needs it available
    except Exception as e:
        st.error(f"TTS Error: {e}")

# ----------- Helper to make detection sentence -----------
def make_sentence(detections):
    if not detections:
        return ""
    unique = list(dict.fromkeys(detections))
    if len(unique) == 1:
        return f"This is {unique[0]}."
    elif len(unique) == 2:
        return f"This is {unique[0]} and {unique[1]}."
    else:
        return f"This is {', '.join(unique[:3])}."

# ----------- Webcam video transformer class -----------
class YoloTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.names = COCO_CLASSES
        self.latest = []

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False)[0]
        annotated = img.copy()
        detected = []

        for det in results.boxes:
            cls_id = int(det.cls[0])
            cls_name = self.names[cls_id]
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            detected.append(cls_name)

        self.latest = list(dict.fromkeys(detected))
        st.session_state["last_webcam_detections"] = self.latest
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ----------- Initialize session states for detections -----------
if "last_webcam_detections" not in st.session_state:
    st.session_state["last_webcam_detections"] = []
if "last_image_detections" not in st.session_state:
    st.session_state["last_image_detections"] = []
if "last_video_detections" not in st.session_state:
    st.session_state["last_video_detections"] = []

# ----------- UI: Mode selection -----------
mode = st.radio(
    "Select input source",
    ("📷 Webcam (recommended)", "🖼️ Image upload", "🎥 Video upload"),
    index=0,
    horizontal=True,
)
st.markdown("---")

left_col, right_col = st.columns((2, 1))

with left_col:
    if mode.startswith("📷"):
        st.markdown("#### Webcam")
        st.info("Allow your browser to access the webcam when prompted. Click the ▶︎ start button inside the video box.")

        ctx = webrtc_streamer(
            key="yolo-webcam",
            video_transformer_factory=YoloTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            # recv_timeout=20,
        )
        st.caption("Tip: click ▶︎ inside the video box to start/stop camera.")

        # Show latest detections and Refresh button
        if ctx.video_transformer:
            st.markdown("**Latest detections (press Refresh to update):**")
            if st.button("🔃 Refresh webcam detections"):
                latest = ctx.video_transformer.latest if ctx.video_transformer else []
                st.write(", ".join(latest) if latest else "No objects detected.")

    elif mode.startswith("🖼️"):
        st.markdown("#### Image upload")
        uploaded_image = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            with st.spinner("Detecting..."):
                results = model(img_bgr, verbose=False)[0]

            annotated = img_bgr.copy()
            detected = []
            for det in results.boxes:
                cls_id = int(det.cls[0])
                name = COCO_CLASSES[cls_id]
                conf = float(det.conf[0])
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(annotated, f"{name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                detected.append(name)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            st.session_state["last_image_detections"] = list(dict.fromkeys(detected))

            st.markdown("**Detected objects:**")
            st.write(", ".join(st.session_state["last_image_detections"]) or "No objects detected.")

    else:
        st.markdown("#### Video upload")
        uploaded_video = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)

            frame_display = st.empty()
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
            processed_detections = []

            with st.spinner("Processing video..."):
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, verbose=False)[0]
                    annotated = frame.copy()
                    detected = []

                    for det in results.boxes:
                        cls_id = int(det.cls[0])
                        name = COCO_CLASSES[cls_id]
                        conf = float(det.conf[0])
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(annotated, f"{name} {conf:.2f}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        detected.append(name)

                    processed_detections.extend(detected)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_display.image(annotated_rgb, use_container_width=True)

                    frame_idx += 1
                    if total_frames:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))

                cap.release()

            st.session_state["last_video_detections"] = list(dict.fromkeys(processed_detections))
            st.markdown("**Video processing complete — aggregated detected objects:**")
            st.write(", ".join(st.session_state["last_video_detections"]) or "No objects detected.")

            try:
                os.unlink(tfile.name)
            except Exception:
                pass

with right_col:
    st.markdown("### 🔎 Detections (quick access)")
    if mode.startswith("📷"):
        st.write("Webcam latest:")
        st.write(", ".join(st.session_state["last_webcam_detections"]) or "No objects detected yet.")
    elif mode.startswith("🖼️"):
        st.write("Image latest:")
        st.write(", ".join(st.session_state["last_image_detections"]) or "No objects processed yet.")
    else:
        st.write("Video latest (aggregated):")
        st.write(", ".join(st.session_state["last_video_detections"]) or "No objects processed yet.")

    st.markdown("---")
    st.markdown("### ℹ️ Notes")
    st.write(
        """
        - Press **Start** on the webcam player to enable camera streaming.
        - The **Say** button (top-left) will only speak when you click it.
        - On webcam mode, click **Refresh webcam detections** to update latest detections.
        - `yolov8n.pt` is recommended for local CPU real-time; use a bigger model if you have GPU.
        - This app uses **gTTS** for TTS, so an internet connection is required.
        """
    )

# ----------- Say Button -----------
say_col, _ = st.columns([1, 5])
with say_col:
    say_clicked = st.button("🗣️ Say (speak current detections)")

if say_clicked:
    if mode.startswith("📷"):
        detections = st.session_state.get("last_webcam_detections", [])
    elif mode.startswith("🖼️"):
        detections = st.session_state.get("last_image_detections", [])
    else:
        detections = st.session_state.get("last_video_detections", [])

    if not detections:
        st.info("No detected objects to say.")
    else:
        sentence = make_sentence(detections)
        st.write("🔊 Saying:", sentence)
        speak_browser(sentence)
