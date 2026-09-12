"""Owner: Student 4"""

from typing import Any

import streamlit as st

_RISK_COLORS = {"LOW": "blue", "MODERATE": "orange", "HIGH": "red"}


def render_risk_card(result: dict[str, Any]) -> None:
    color = _RISK_COLORS.get(result["risk"], "grey")
    st.subheader("Risk Assessment")
    st.markdown(f":{color}[**{result['risk']}**]")
    st.write(result["message"])
