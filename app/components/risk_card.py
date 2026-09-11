"""Owner: Student 4"""

import streamlit as st

from src.predict import Prediction

_RISK_COLORS = {"LOW": "blue", "MODERATE": "orange", "HIGH": "red"}


def render_risk_card(result: Prediction) -> None:
    color = _RISK_COLORS.get(result["risk"], "grey")
    st.subheader("Risk Assessment")
    st.markdown(f":{color}[**{result['risk']}**]")
    st.write(result["message"])
