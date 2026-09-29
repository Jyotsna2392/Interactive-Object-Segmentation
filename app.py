import os
import io
import urllib.request
import warnings

import numpy as np
import streamlit as st
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHECKPOINT_DIR = "sam2_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sam2.1_hiera_tiny.pt"
)

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_tiny.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ------------------------------------------------------------
# Download checkpoint
# ------------------------------------------------------------

@st.cache_resource
def download_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        with st.spinner("Downloading SAM 2.1 model..."):
            urllib.request.urlretrieve(
                CHECKPOINT_URL,
                CHECKPOINT_PATH
            )

    return CHECKPOINT_PATH


# ------------------------------------------------------------
# Load SAM 2 model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    checkpoint = download_checkpoint()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_sam2(
        MODEL_CFG,
        checkpoint,
        device=device
    )

    model.eval()

    predictor = SAM2ImagePredictor(model)

    return predictor, device


# ------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------

def overlay_mask(image, mask, alpha=0.55):
    image = image.copy()

    overlay = np.zeros_like(image)

    # Cyan mask
    overlay[mask.astype(bool)] = [0, 230, 255]

    result = (
        image.astype(np.float32) * (1 - alpha)
        + overlay.astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def extract_object(image, mask):
    rgba = np.dstack([
        image,
        np.full(image.shape[:2], 255, dtype=np.uint8)
    ])

    rgba[~mask.astype(bool), 3] = 0

    return rgba


def pick_best_mask(masks, scores):
    best_idx = int(np.argmax(scores))

    return (
        masks[best_idx],
        float(scores[best_idx]),
        best_idx
    )


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="SAM 2 Interactive Object Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Interactive Object Segmentation using SAM 2")

st.markdown(
    """
    Upload an image and use either a **Point Prompt** or a
    **Bounding Box Prompt** to segment an object using
    **Meta AI's SAM 2.1 Hiera-Tiny model**.
    """
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

try:
    predictor, device = load_model()

    st.sidebar.success(
        f"SAM 2 loaded successfully\n\nDevice: {device.upper()}"
    )

except Exception as e:
    st.error(f"Could not load SAM 2: {e}")
    st.stop()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Segmentation Settings")

prompt_type = st.sidebar.radio(
    "Choose Prompt Type",
    [
        "Point Prompt",
        "Bounding Box"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    """
    **Point Prompt:** Click approximately on the object.

    **Bounding Box:** Draw a rectangle around the object.
    """
)


# ------------------------------------------------------------
# Upload image
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)

if uploaded_file is None:

    st.info(
        "👆 Upload an image to begin interactive segmentation."
    )

    st.stop()


# ------------------------------------------------------------
# Read image
# ------------------------------------------------------------

image = Image.open(uploaded_file).convert("RGB")

image_rgb = np.array(image)

height, width = image_rgb.shape[:2]

st.subheader("Input Image")

st.image(
    image_rgb,
    caption=f"Image size: {width} × {height}",
    use_container_width=True
)


# ------------------------------------------------------------
# Point Prompt
# ------------------------------------------------------------

if prompt_type == "Point Prompt":

    st.subheader("Point Prompt")

    st.write(
        "Enter the pixel coordinates of the object you want to segment."
    )

    col1, col2 = st.columns(2)

    with col1:
        point_x = st.number_input(
            "X coordinate",
            min_value=0,
            max_value=max(width - 1, 0),
            value=width // 2,
            step=1
        )

    with col2:
        point_y = st.number_input(
            "Y coordinate",
            min_value=0,
            max_value=max(height - 1, 0),
            value=height // 2,
            step=1
        )

    point_x = int(point_x)
    point_y = int(point_y)

    # Show selected point
    preview = image_rgb.copy()

    try:
        import cv2

        cv2.circle(
            preview,
            (point_x, point_y),
            max(5, min(width, height) // 100),
            (0, 255, 255),
            -1
        )

    except Exception:
        pass

    st.image(
        preview,
        caption=f"Selected point: ({point_x}, {point_y})",
        use_container_width=True
    )

    segment_button = st.button(
        "🎯 Segment Object",
        type="primary"
    )

    if segment_button:

        with st.spinner("Running SAM 2 segmentation..."):

            try:

                with torch.inference_mode():

                    predictor.set_image(image_rgb)

                    masks, scores, logits = predictor.predict(
                        point_coords=np.array([
                            [point_x, point_y]
                        ]),
                        point_labels=np.array([1]),
                        multimask_output=True
                    )

                mask, score, index = pick_best_mask(
                    masks,
                    scores
                )

                overlay = overlay_mask(
                    image_rgb,
                    mask
                )

                extracted = extract_object(
                    image_rgb,
                    mask
                )

                st.success(
                    f"Segmentation completed! "
                    f"Best mask score: {score:.4f}"
                )

                # ------------------------------------------------
                # Results
                # ------------------------------------------------

                st.subheader("Segmentation Results")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.image(
                        image_rgb,
                        caption="Original Image",
                        use_container_width=True
                    )

                with c2:
                    st.image(
                        overlay,
                        caption="Mask Overlay",
                        use_container_width=True
                    )

                with c3:
                    st.image(
                        extracted,
                        caption="Extracted Object",
                        use_container_width=True
                    )

                # Mask
                st.subheader("Segmentation Mask")

                st.image(
                    mask.astype(np.uint8) * 255,
                    caption="Binary Segmentation Mask",
                    use_container_width=True
                )

                # Download
                output_buffer = io.BytesIO()

                Image.fromarray(
                    extracted
                ).save(
                    output_buffer,
                    format="PNG"
                )

                st.download_button(
                    "⬇️ Download Extracted Object",
                    data=output_buffer.getvalue(),
                    file_name="extracted_object_point.png",
                    mime="image/png"
                )

            except Exception as e:

                st.error(
                    f"Segmentation failed: {e}"
                )


# ------------------------------------------------------------
# Bounding Box Prompt
# ------------------------------------------------------------

else:

    st.subheader("Bounding Box Prompt")

    st.write(
        "Enter the bounding box coordinates around the object."
    )

    col1, col2 = st.columns(2)

    with col1:

        x_min = st.number_input(
            "X minimum",
            min_value=0,
            max_value=max(width - 1, 0),
            value=int(width * 0.25),
            step=1
        )

        y_min = st.number_input(
            "Y minimum",
            min_value=0,
            max_value=max(height - 1, 0),
            value=int(height * 0.20),
            step=1
        )

    with col2:

        x_max = st.number_input(
            "X maximum",
            min_value=0,
            max_value=max(width - 1, 0),
            value=int(width * 0.80),
            step=1
        )

        y_max = st.number_input(
            "Y maximum",
            min_value=0,
            max_value=max(height - 1, 0),
            value=int(height * 0.85),
            step=1
        )

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    if x_max <= x_min or y_max <= y_min:

        st.warning(
            "Please provide a valid bounding box."
        )

        st.stop()

    # --------------------------------------------------------
    # Draw box preview
    # --------------------------------------------------------

    preview = image_rgb.copy()

    try:

        import cv2

        cv2.rectangle(
            preview,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 255),
            max(2, min(width, height) // 200)
        )

    except Exception:
        pass

    st.image(
        preview,
        caption="Bounding Box Prompt",
        use_container_width=True
    )

    segment_button = st.button(
        "🎯 Segment Object",
        type="primary"
    )

    if segment_button:

        with st.spinner("Running SAM 2 segmentation..."):

            try:

                box = np.array([
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ])

                with torch.inference_mode():

                    predictor.set_image(
                        image_rgb
                    )

                    masks, scores, logits = predictor.predict(
                        box=box[None, :],
                        multimask_output=True
                    )

                mask, score, index = pick_best_mask(
                    masks,
                    scores
                )

                overlay = overlay_mask(
                    image_rgb,
                    mask
                )

                extracted = extract_object(
                    image_rgb,
                    mask
                )

                st.success(
                    f"Segmentation completed! "
                    f"Best mask score: {score:.4f}"
                )

                # ------------------------------------------------
                # Results
                # ------------------------------------------------

                st.subheader("Segmentation Results")

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.image(
                        image_rgb,
                        caption="Original Image",
                        use_container_width=True
                    )

                with c2:

                    st.image(
                        overlay,
                        caption="Mask Overlay",
                        use_container_width=True
                    )

                with c3:

                    st.image(
                        extracted,
                        caption="Extracted Object",
                        use_container_width=True
                    )

                # ------------------------------------------------
                # Mask
                # ------------------------------------------------

                st.subheader("Segmentation Mask")

                st.image(
                    mask.astype(np.uint8) * 255,
                    caption="Binary Segmentation Mask",
                    use_container_width=True
                )

                # ------------------------------------------------
                # Download
                # ------------------------------------------------

                output_buffer = io.BytesIO()

                Image.fromarray(
                    extracted
                ).save(
                    output_buffer,
                    format="PNG"
                )

                st.download_button(
                    "⬇️ Download Extracted Object",
                    data=output_buffer.getvalue(),
                    file_name="extracted_object_box.png",
                    mime="image/png"
                )

            except Exception as e:

                st.error(
                    f"Segmentation failed: {e}"
                )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Interactive Object Segmentation using Meta AI SAM 2.1 Hiera-Tiny"
)
