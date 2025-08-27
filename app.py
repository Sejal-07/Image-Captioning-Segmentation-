# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from utils.captioning import generate_caption, generate_multiple_captions
from utils.segmentation import perform_segmentation
from utils.visualization import visualize_segmentation, create_legend_figure

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Image Captioning & Segmentation", page_icon="🖼️", layout="wide"
)

# -------------------------------
# Title and Description
# -------------------------------
st.title("🖼️ Dual Task: Image Captioning & Segmentation")
st.markdown(
    """
This application demonstrates a dual approach to image analysis:
1. **Image Captioning** – Generating descriptive text for images  
2. **Image Segmentation** – Identifying and labeling regions in images  

⚡ *Note*: The first run may take longer as models need to download.
"""
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Model Settings")

caption_model = st.sidebar.selectbox("Captioning Model", ["BLIP", "CLIP"])
seg_model = st.sidebar.selectbox("Segmentation Model", ["DeepLabV3", "Mask R-CNN"])

num_captions = st.sidebar.slider(
    "Number of Captions to Generate", min_value=1, max_value=5, value=1
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5
)

# Sample images
sample_images = {
    "Select an image": None,
    "Sample 1": "data\coco\images\sample_image1.png",
    "Sample 2": "data\coco\images\sample_image2.jpeg",
    "Sample 3": "data\coco\images\sample_image3.jpeg",
}
selected_sample = st.sidebar.selectbox(
    "Or choose a sample image:", list(sample_images.keys())
)

# File uploader
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

# -------------------------------
# Session State
# -------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "segmentation_map" not in st.session_state:
    st.session_state.segmentation_map = None

# -------------------------------
# Load Image
# -------------------------------
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.processed = False
elif selected_sample != "Select an image" and os.path.exists(
    sample_images[selected_sample]
):
    image = Image.open(sample_images[selected_sample]).convert("RGB")
    st.session_state.processed = False

# -------------------------------
# Process Image
# -------------------------------
if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    if st.button("🚀 Process Image") or st.session_state.processed:
        st.session_state.processed = True

        with st.spinner("Processing image..."):

            # -------------------------------
            # Captioning
            # -------------------------------
            with col1:
                st.subheader("📝 Image Captioning")
                if not st.session_state.caption:
                    try:
                        if num_captions > 1:
                            captions = generate_multiple_captions(
                                img_array, num_captions
                            )
                            for i, caption in enumerate(captions):
                                st.success(f"Caption {i+1}: {caption}")
                        else:
                            st.session_state.caption = generate_caption(
                                img_array, model_type=caption_model
                            )
                            st.success(st.session_state.caption)
                    except Exception as e:
                        st.error(f"Captioning failed: {e}")
                else:
                    st.success(st.session_state.caption)

            # -------------------------------
            # Segmentation
            # -------------------------------
            # -------------------------------
            # Segmentation
            # -------------------------------
            with col2:
                st.subheader("🎨 Image Segmentation")
                if st.session_state.segmentation_map is None:
                    try:
                        seg_result = perform_segmentation(
                            img_array, model_type=seg_model, threshold=conf_threshold
                        )
                        st.session_state.segmentation_map = seg_result
                    except Exception as e:
                        st.error(f"Segmentation failed: {e}")
                        st.stop()

                seg_result = st.session_state.segmentation_map

                # Visualization
                try:
                    seg_visualization = visualize_segmentation(img_array, seg_result)
                    st.image(
                        seg_visualization,
                        caption="Segmentation Result",
                        use_column_width=True,
                    )
                except Exception as e:
                    st.error(f"Visualization failed: {e}")

                # -------------------------------
                # Show Segmentation Details (FIXED)
                # -------------------------------
                if isinstance(seg_result, dict):
                    detected_classes = []

                    # Instance segmentation (Mask R-CNN style)
                    if "labels" in seg_result and isinstance(
                        seg_result["labels"], list
                    ):
                        detected_classes = seg_result["labels"]

                        st.write("**Detected objects/classes:**")
                        for i, label in enumerate(seg_result["labels"]):
                            if "scores" in seg_result and i < len(seg_result["scores"]):
                                score = seg_result["scores"][i]
                                st.write(f"- {label} (confidence: {score:.2f})")
                            else:
                                st.write(f"- {label}")

                    # Semantic segmentation (DeepLabV3 style)
                    elif "masks" in seg_result:
                        mask = seg_result["masks"]
                        unique_classes = np.unique(mask)

                        # Ignore background (0)
                        if "labels" in seg_result:
                            detected_classes = [
                                seg_result["labels"][c - 1]
                                for c in unique_classes
                                if c != 0 and (c - 1) < len(seg_result["labels"])
                            ]

                        if detected_classes:
                            st.write("**Detected objects/classes:**")
                            for cls in detected_classes:
                                st.write(f"- {cls}")

                    # Show legend
                    if "legend_patches" in seg_result:
                        legend_fig = create_legend_figure(seg_result["legend_patches"])
                        if legend_fig:
                            st.pyplot(legend_fig)


# -------------------------------
# Instructions
# -------------------------------
with st.expander("📖 How to use this application"):
    st.markdown(
        """
    1. **Upload an image** or select a sample from the sidebar  
    2. **Adjust settings** (models, captions, confidence)  
    3. **Click 'Process Image'** to run captioning & segmentation  
    4. **View results** – captions on the left, segmentation on the right  
    """
    )

# -------------------------------
