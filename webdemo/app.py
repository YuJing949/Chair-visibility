import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import os
import urllib.request

from segment_anything import sam_model_registry, SamPredictor
from utils import analyze_visual_contrast, overlay_mask
from detect import detect_chairs

# --- Streamlit Page Config ---
st.set_page_config(page_title="Chair Visibility Analyzer", layout="wide")

# --- Custom CSS ---
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

# --- Load SAM model ---
MODEL_PATH = "sam_vit_b_01ec64.pth"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading SAM model (~357MB)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

sam_checkpoint = MODEL_PATH
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# --- Title ---
st.title("Chair Visibility Analyzer (Image Upload + YOLO + SAM)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(image_pil)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Step 1: Detect chairs
    st.markdown("### Step 1: Object Detection (YOLOv8)")
    boxes, centers = detect_chairs(image)

    if not boxes:
        st.warning("No chairs detected in the image.")
    else:
        st.success(f"Detected {len(boxes)} chair(s). Running segmentation...")

        for idx, (bbox, center) in enumerate(zip(boxes, centers)):
            x, y, w, h = bbox
            cx, cy = center

            # Step 2: Segmentation with SAM
            predictor.set_image(image)
            input_point = np.array([[cx, cy]])
            input_label = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            mask = masks[0]

            # Step 3: Analyze contrast
            results = analyze_visual_contrast(image, mask)

            # Step 4: Display results
            st.markdown(f"#### Chair {idx+1}")
            st.image(overlay_mask(image, mask), caption="Segmented Mask", use_container_width=True)

            is_accessible = results['brightness_result'] == "High contrast" and results['color_result'] == "High contrast"
            contrast_summary = "High contrast" if is_accessible else "Low contrast"
            st.markdown(f"**Visibility Result:** {contrast_summary}")

            with st.expander("Show Analysis Details"):
                st.markdown(f"- Brightness Contrast: **{results['brightness_contrast']:.2f}** → {results['brightness_result']}")
                st.markdown(f"- Color Contrast (ΔE): **{results['delta_e']:.2f}** → {results['color_result']}")

            if is_accessible:
                st.success("This chair appears accessible to users with visual impairments.")
            else:
                st.error("This chair may be hard to distinguish from the background. Consider choosing a chair with stronger color contrast.")

