"""Owner: Student 3

Renders the prediction result. Consumes the src.predict.Prediction contract
only -- no model internals, no duplicated class mapping.
"""

from __future__ import annotations

import html

import streamlit as st

from app.mock_prediction import HEALTHY_CLASS
from src.config import RISK_THRESHOLD_MODERATE
from src.predict import Prediction

LOW_CONFIDENCE_TEXT = "Model confidence is low. Manual inspection recommended."


def display_name(class_name: str) -> str:
    """Turn a dataset label such as Bacterial_Gill_Disease into readable text."""
    return class_name.replace("_", " ").strip()


def render_result_card(result: Prediction) -> None:
    """Draw the diagnosis card for one backend result."""
    predicted = str(result["predicted_class"])
    confidence = _as_float(result["confidence"])
    confidence = min(max(confidence, 0.0), 1.0)

    # src.risk_engine treats anything under RISK_THRESHOLD_MODERATE as low
    # confidence, so the UI uses the same line rather than inventing its own.
    low = confidence < RISK_THRESHOLD_MODERATE
    is_healthy = predicted == HEALTHY_CLASS
    headline = "No disease detected" if is_healthy else "AI-based disease indication"
    meter_class = "ah-meter-fill is-low" if low else "ah-meter-fill"

    st.markdown(
        f"""
        <div class="ah-result">
          <div class="ah-result-head">
            <div>
              <div class="ah-eyebrow">{html.escape(headline)}</div>
              <div class="ah-diagnosis">{html.escape(display_name(predicted))}</div>
            </div>
            <div class="ah-conf">
              <div class="ah-conf-value">{confidence * 100:.1f}%</div>
              <div class="ah-conf-label">Confidence</div>
            </div>
          </div>
          <div class="ah-meter">
            <div class="{meter_class}" style="width:{confidence * 100:.1f}%"></div>
          </div>
          {_low_confidence_note(low)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_probabilities(result, predicted)


def _low_confidence_note(low: bool) -> str:
    if not low:
        return ""
    threshold = int(RISK_THRESHOLD_MODERATE * 100)
    return (
        '<div class="ah-note">&#9888;&#65039;<div><strong>'
        f"{html.escape(LOW_CONFIDENCE_TEXT)}</strong><br>"
        f"The model is under {threshold}% sure, so please do not act on this result "
        "alone - take a clearer photo, or ask a fisheries officer to look at the "
        "fish.</div></div>"
    )


def _render_probabilities(result: Prediction, predicted: str) -> None:
    """Compact distribution across every class, when the backend supplies one.

    ``probabilities`` is an optional extra key (see app.mock_prediction), so
    the card simply omits this section when it is absent.
    """
    probabilities = dict(result).get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return

    with st.expander("See how the AI scored every disease class", expanded=False):
        st.markdown(
            '<div style="font-size:13px;color:#4a6474;margin-bottom:10px;">Each bar '
            "shows how strongly the photo matches that condition. The bars add up "
            "to 100%.</div>",
            unsafe_allow_html=True,
        )

        ordered = sorted(probabilities.items(), key=lambda item: _as_float(item[1]), reverse=True)

        rows = []
        for label, value in ordered:
            score = min(max(_as_float(value), 0.0), 1.0)
            top = " is-top" if label == predicted else ""
            rows.append(
                f'<div class="ah-prob-row{top}">'
                f'<div class="ah-prob-name">{html.escape(display_name(str(label)))}</div>'
                f'<div class="ah-prob-track"><div class="ah-prob-bar" '
                f'style="width:{score * 100:.1f}%"></div></div>'
                f'<div class="ah-prob-pct">{score * 100:.1f}%</div>'
                "</div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
