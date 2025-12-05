# app.py
import streamlit as st
import os
from PIL import Image
import pandas as pd
import json
from io import BytesIO
from tempfile import NamedTemporaryFile

from detecte import load_model, detect
from my_data import prepare_image, draw_on_image
from yolo_func import detect_yolo

# MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Dual Object Detector", page_icon="magnifying glass", layout="centered")

st.title("Object Detection Showdown")
st.markdown("### Faster R-CNN vs YOLOv8n — pick your model and see the difference!")

# Model selector
model_choice = st.radio(
    "Choose detection model",
    ["Faster R-CNN (ResNet50 – accurate)", "YOLOv8n (super fast)"],
    horizontal=True,
    index=0
)

# Load the right detector
if "Faster R-CNN" in model_choice:
    @st.cache_resource
    def get_rcnn():
        with st.spinner("Loading Faster R-CNN from TensorFlow Hub (~30s first time)..."):
            return load_model()
    detector = get_rcnn()
    current_model = "rcnn"
else:
    detector = None
    current_model = "yolo"

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Original Image", use_container_width=True)

    # Run detection button
    if st.button("Run Detection", type="primary", use_container_width=True):
        with st.spinner(f"Detecting with {model_choice.split('(')[0]}..."):
            try:
                if current_model == "yolo":
                    # YOLOv8 path
                    boxes, scores, classes, result_img = detect_yolo(original_image.copy(), confidence_threshold=0.5)

                    st.session_state.results = {
                        "boxes": boxes,
                        "scores": scores,
                        "classes": classes,
                        "original_image": original_image.copy(),
                        "model": "yolo"
                    }
                else:
                    # Faster R-CNN path
                    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        original_image.save(tmp.name)
                        tmp_path = tmp.name

                    image_for_model, input_tensor = prepare_image(tmp_path)
                    boxes, scores, classes = detect(detector, input_tensor)

                    st.session_state.results = {
                        "boxes": boxes,
                        "scores": scores,
                        "classes": classes,
                        "image_for_model": image_for_model.copy(),
                        "original_image": original_image.copy(),
                        "model": "rcnn"
                    }

                    os.unlink(tmp_path)

                st.success("Detection finished! Adjust threshold below.")
                st.rerun()

            except Exception as e:
                st.error(f"Detection failed: {e}")

    # ────────────────────── RESULTS & LIVE THRESHOLD ──────────────────────
    if "results" in st.session_state:
        res = st.session_state.results

        # Load class names
        with open("config.json", "r", encoding="utf-8") as f:
            class_names = json.load(f)["class_names"]

        # Threshold slider
        threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
            help="Live update — no re-run needed!"
        )

        # Filter by threshold
        mask = res["scores"] >= threshold
        boxes = res["boxes"][mask]
        scores = res["scores"][mask]
        classes = res["classes"][mask]

        # Draw image with current threshold
        if res["model"] == "yolo":
            # YOLO redraws everything with new threshold
            _, _, _, display_image = detect_yolo(res["original_image"].copy(), confidence_threshold=threshold)
        else:
            display_image = draw_on_image(
                res["image_for_model"].copy(),
                boxes, scores, classes,
                class_names=class_names,
                threshold=threshold
            )

        # Prepare table
        detections = []
        for s, c in zip(scores, classes):
            idx = int(c) - 1
            name = class_names[idx] if 0 <= idx < len(class_names) else f"Class {int(c)}"
            detections.append({"Object": name.capitalize(), "Confidence": f"{s:.1%}"})

        # Display side-by-side
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(display_image, caption=f"{model_choice.split('(')[0]} ≥ {threshold:.0%}", use_container_width=True)

        with col2:
            st.subheader("Detected Objects")
            if detections:
                df = pd.DataFrame(detections)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.metric("Total objects", len(detections))
            else:
                st.info("No detections above current threshold")

        # Download button
        buf = BytesIO()
        display_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            "Download result image",
            data=buf,
            file_name=f"{current_model}_result.png",
            mime="image/png"
        )

else:
    st.info("Please upload an image to start")

st.caption("Powered by TensorFlow Hub + Ultralytics YOLOv8 + Streamlit")