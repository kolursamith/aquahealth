"""Owner: Student 3"""

from typing import Any

import streamlit as st


def render_result_card(result: dict[str, Any]) -> None:
    st.subheader("Prediction")
    st.metric("Predicted Class", result["predicted_class"])
    st.metric("Confidence", f"{result['confidence']:.1%}")
    ranked = result.get("ranked_predictions") or []
    if ranked:
        st.table(
            [
                {
                    "rank": r["rank"],
                    "class": r["class_name"],
                    "probability": f"{r['probability']:.1%}",
                }
                for r in ranked
            ]
        )
    for warning in result.get("warnings") or []:
        st.caption(f"Note: {warning}")
