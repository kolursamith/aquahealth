"""Owner: Student 4"""

from typing import Any

import streamlit as st

_RISK_COLORS = {"LOW": "blue", "MODERATE": "orange", "HIGH": "red"}
_HEALTHY_COLORS = {"LOW": "blue", "MODERATE": "green", "HIGH": "green"}


def render_risk_card(result: dict[str, Any]) -> None:
    """Risk level by confidence; a Healthy Fish prediction is shown as a green finding."""
    healthy = bool(result.get("healthy"))
    palette = _HEALTHY_COLORS if healthy else _RISK_COLORS
    color = palette.get(result["risk"], "grey")
    st.subheader("Risk Assessment")
    if healthy:
        st.markdown(f":{color}[**HEALTHY — no disease detected**] (confidence {result['risk']})")
    else:
        st.markdown(f":{color}[**{result['risk']}**]")
    st.write(result["message"])
