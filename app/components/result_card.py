"""Owner: Student 3"""

import streamlit as st

from src.predict import Prediction


def render_result_card(result: Prediction) -> None:
    st.subheader("Prediction")
    st.metric("Predicted Class", result["predicted_class"])
    st.metric("Confidence", f"{result['confidence']:.1%}")
