import streamlit as st
import av
import numpy as np
import cv2
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from segment_anything import sam_model_registry, SamPredictor
from utils import analyze_visual_contrast, overlay_mask

# Load SAM model
sam_checkpoint = "D:/cci 2025/segmentation/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Streamlit page
st.title("Chair Visibility Analyzer (Webcam Demo)")

# Webcam feed
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="realtime-analyzer",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

# Capture button
if st.button("Capture Frame"):
    if ctx.video_processor and ctx.video_processor.frame is not None:
        frame_bgr = ctx.video_processor.frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.session_state['frame_rgb'] = frame_rgb
    else:
        st.warning("No frame available.")

# If image captured, show and allow user input
if 'frame_rgb' in st.session_state:
    image = st.session_state['frame_rgb']
    st.image(image, caption="Captured Frame", use_container_width=True)

    # Coordinate input
    h, w, _ = image.shape
    x = st.number_input("X coordinate", min_value=0, max_value=w-1, value=w//2)
    y = st.number_input("Y coordinate", min_value=0, max_value=h-1, value=h//2)

    if x > 0 and y > 0:
        predictor.set_image(image)
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        mask = masks[0]

        result = analyze_visual_contrast(image, mask)
        st.markdown(f"**Brightness Contrast**: {result['brightness_contrast']:.2f} - {result['brightness_result']}")
        st.markdown(f"**Color Contrast (ΔE)**: {result['delta_e']:.2f} - {result['color_result']}")

        overlay = overlay_mask(image, mask)
        st.image(overlay, caption="Segmented Area", use_container_width=True)
