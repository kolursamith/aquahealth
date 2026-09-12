"""Owner: Student 4"""

import streamlit as st


def render_dashboard(class_names: list[str], source: str) -> None:
    st.info("Upload a fish image to run disease detection.")
    with st.expander(f"Classes known to the loaded model ({len(class_names)})"):
        st.caption(source)
        for name in class_names:
            st.write(f"- {name}")
