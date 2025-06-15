# --- Built-in Libraries ---
import os
import urllib.request

# --- Third-party Libraries ---
import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image

# --- Project Modules ---
from segment_anything import build_sam_vit_t, SamPredictor
from utils import analyze_visual_contrast, overlay_mask
from detect import detect_chairs

# --- Streamlit Config ---
st.set_page_config(page_title="Chair Visibility Analyzer", layout="wide")
st.markdown("""
<style>
body, .stApp {
    background-color: #1e1e1e;
    color: white;
}
.stButton>button, .stFileUploader>div>div {
    background-color: transparent;
    color: white;
    border: 1px solid white;
    padding: 0.5em 1em;
}
.stTextInput>div>div>input, .stNumberInput>div>input {
    background-color: transparent;
    color: white;
    border: 1px solid white;
}
.stImage>div>img {
    border: 1px solid white;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Load SAM Tiny Model ---
MODEL_PATH = "sam_vit_tiny_646f52.pth"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_tiny_646f52.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading SAM Tiny model (~120MB)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = build_sam_vit_t(checkpoint=MODEL_PATH).to(device)
predictor = SamPredictor(sam)

# --- Title ---
st.title("Chair Visibility Analyzer (YOLOv8 + SAM Tiny)")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image = np.array(image_pil)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Resize if needed
    max_dim = 256
    if image.shape[0] > max_dim or image.shape[1] > max_dim:
        scale = max_dim / max(image.shape[:2])
        new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image = cv2.resize(image, new_size)
        st.warning(f"Image resized to {new_size} for performance.")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Step 1: Detect Chairs ---
    st.markdown("### Step 1: Object Detection (YOLOv8)")
    boxes, centers = detect_chairs(image)

    if not boxes:
        st.warning("No chairs detected.")
        st.stop()
    else:
        st.success(f"Detected {len(boxes)} chair(s). Running segmentation...")

    # --- Step 2–4: Segment + Analyze + Display ---
    for idx, (bbox, center) in enumerate(zip(boxes, centers)):
        x, y, w, h = bbox
        cx, cy = center

        try:
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            mask = masks[0]
        except Exception as e:
            st.error(f"Segmentation failed for chair {idx+1}: {e}")
            continue

        results = analyze_visual_contrast(image, mask)

        # --- Show Result ---
        st.markdown(f"#### Chair {idx+1}")
        st.image(overlay_mask(image, mask), caption="Segmented Mask", use_container_width=True)

        is_accessible = (
            results["brightness_result"] == "High contrast"
            and results["color_result"] == "High contrast"
        )
        result_text = "High contrast" if is_accessible else "Low contrast"
        st.markdown(f"**Visibility Result:** {result_text}")

        with st.expander("Show Analysis Details"):
            st.markdown(f"- Brightness Contrast: **{results['brightness_contrast']:.2f}** → {results['brightness_result']}")
            st.markdown(f"- Color Contrast (ΔE): **{results['delta_e']:.2f}** → {results['color_result']}")

        if is_accessible:
            st.success("This chair appears accessible to users with visual impairments.")
        else:
            st.error("This chair may be hard to distinguish from the background.")
