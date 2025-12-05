# app.py
import streamlit as st
import tempfile
import os
from PIL import Image
from detecte import *
from my_data import *
from constants import *

# Page config
st.set_page_config(page_title="My Object Detector", layout="centered")
st.title("Object Detection Demo")
st.write("Upload an image → Click **Run Detection** → See bounding boxes")

# Load model once and cache it
@st.cache_resource
def get_detector():
    with st.spinner("Loading model (only once)..."):
        return load_model()

detector = get_detector()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width="stretch")  # Updated: use width="stretch"

    # Button to run detection
    if st.button("Run Detection", type="primary"):
        with st.spinner("Detecting objects..."):
            # Save uploaded image to a temporary file (because your function needs a path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name

            try:
                # Your original working code - unchanged!
                image_for_model, input_tensor = prepare_image(temp_path)
                boxes, scores, classes = dectect(detector, input_tensor)
                result_image = draw_on_image(image_for_model, boxes, scores, classes, threshold=CONFIANCE_THRESHOLD)

                # Display result
                st.image(result_image, caption="Result with Bounding Boxes", width="stretch")  # Updated: use width="stretch"
                st.success(f"Detection complete! Found {sum(s >= 0.5 for s in scores)} objects")

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

else:
    st.info("Please upload an image to start")

st.caption("Powered by your custom detector")