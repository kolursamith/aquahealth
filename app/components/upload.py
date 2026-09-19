"""Upload zone. Returns raw bytes + name; the backend does the decoding and validation.

Owner: Student 3
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

SUPPORTED = ["jpg", "jpeg", "png", "webp", "bmp"]


def render_upload(key: str) -> tuple[bytes, str] | None:
    uploaded = st.file_uploader(
        "Drop fish image",
        type=SUPPORTED,
        key=key,
        help="One fish per photo, side view, lesion visible, good light. JPG, PNG, WEBP or BMP.",
        label_visibility="collapsed",
    )
    if uploaded is None:
        return None
    return uploaded.getvalue(), uploaded.name


def image_facts(data: bytes) -> dict[str, str]:
    """Real metadata of the uploaded bytes (for the preview panel); empty on decode failure."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return {
                "Resolution": f"{image.width} × {image.height} px",
                "Format": image.format or "unknown",
                "File size": f"{len(data) / 1024:.0f} KB",
            }
    except Exception:  # noqa: BLE001 - the backend reports the real decode error to the user
        return {"File size": f"{len(data) / 1024:.0f} KB"}
