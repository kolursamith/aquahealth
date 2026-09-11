"""Owner: Student 4"""

import streamlit as st

from src.config import CLASS_NAMES


def render_dashboard() -> None:
    st.info("Upload a fish image to run disease detection.")
    with st.expander("Supported disease classes"):
        for name in CLASS_NAMES:
            st.write(f"- {name}")
