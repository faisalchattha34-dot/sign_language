import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from collections import deque, Counter

st.set_page_config(layout="wide", page_title="ASL Detection - YOLOv8")

st.title("🤖 American Sign Language Detection (YOLOv8)")

MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

st.sidebar.header("Settings")
mode = st.sidebar.radio("Choose mode:", ["Image Upload", "Use Camera", "Video Upload"])
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25)
imgsz = st.sidebar.selectbox("Image size", [320, 480, 640, 896], index=2)

def run_inference(img):
    res = model.predict(source=img, conf=conf, imgsz=imgsz)
    out = res[0].plot()
    boxes = res[0].boxes
    labels = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            labels.append((model.names[cls], conf_score))
    return out[:, :, ::-1], labels

# ----- Image upload -----
if mode == "Image Upload":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        st.image(img_np, caption="Uploaded Image", use_column_width=True)
        out_img, labels = run_inference(img_np)
        st.image(out_img, caption="Predicted", use_column_width=True)
        st.write("Detected Signs:", labels)

# ----- Camera -----
elif mode == "Use Camera":
    img_file = st.camera_input("Take a picture")
    if img_file:
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)
        out_img, labels = run_inference(img_np)
        st.image(out_img, caption="Prediction", use_column_width=True)
        st.write("Detected Signs:", labels)

# ----- Video -----
else:
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        sequence = []
        window = deque(maxlen=7)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_img, labels = run_inference(frame_rgb)

            pred = labels[0][0] if labels else None
            if pred:
                window.append(pred)
                most = Counter(window).most_common(1)[0][0]
                if not sequence or sequence[-1] != most:
                    sequence.append(most)

            stframe.image(out_img, use_column_width=True)
            st.markdown(f"### 🧠 Predicted Sequence: {' '.join(sequence)}")

        cap.release()
        os.unlink(tfile.name)
