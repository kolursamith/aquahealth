"""Owner: Student 3"""

import numpy as np
import streamlit as st
from PIL import Image


def render_upload() -> np.ndarray | None:
    """Render a file uploader and return the image as an RGB numpy array, or None."""
    uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return None

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    return np.array(image)
