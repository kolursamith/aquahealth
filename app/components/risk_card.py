"""Owner: Student 3 per docs/team/student_3.md.

NOTE(team): the original skeleton header on this file said "Owner: Student 4",
which contradicts docs/team/student_3.md. Flagged rather than silently taken.

Renders the risk level and the recommended action. The risk value and the
message always come from the backend (src.risk_engine via the Prediction
dict); this module only decides how to present them.

Presentation caveat: src.risk_engine.get_risk_level() is derived from
confidence alone, so a confidently *healthy* fish also comes back as
risk="HIGH". Showing that as "HIGH RISK" would badly mislead a farmer, so the
card reads the risk together with the predicted class and labels a healthy
result as confidence rather than danger. See the note in the PR description.
"""

from __future__ import annotations

import html

import streamlit as st

from app.mock_prediction import HEALTHY_CLASS
from src.predict import Prediction
from src.risk_engine import RISK_HIGH, RISK_LOW, RISK_MODERATE

# Styling for a disease indication, keyed by the backend's risk level.
_DISEASE_STYLES = {
    RISK_HIGH: {
        "css": "risk-high",
        "icon": "&#128680;",
        "pill": "HIGH RISK",
        "caption": "Act today",
        "hint": (
            "Keep the affected fish away from healthy stock and avoid moving water "
            "or nets between ponds until someone has checked them."
        ),
    },
    RISK_MODERATE: {
        "css": "risk-moderate",
        "icon": "&#9888;&#65039;",
        "pill": "MODERATE RISK",
        "caption": "Watch closely",
        "hint": (
            "Check the pond twice a day for the next few days and take another photo "
            "if the marks spread."
        ),
    },
    RISK_LOW: {
        "css": "risk-uncertain",
        "icon": "&#10068;",
        "pill": "LOW CONFIDENCE",
        "caption": "Result is uncertain",
        "hint": (
            "Take a new photo in daylight with the whole fish in frame and run the "
            "analysis again before acting on this."
        ),
    },
}

# Styling when the model says the fish is healthy. Here the backend's risk
# level reads as "how sure the model is", not "how dangerous this is".
_HEALTHY_STYLES = {
    RISK_HIGH: {
        "css": "risk-low",
        "icon": "&#9989;",
        "pill": "HIGH CONFIDENCE",
        "caption": "No disease detected",
        "hint": "Keep up your usual feeding, water changes and pond hygiene.",
    },
    RISK_MODERATE: {
        "css": "risk-low",
        "icon": "&#9989;",
        "pill": "MODERATE CONFIDENCE",
        "caption": "No disease detected",
        "hint": ("Nothing to do right now, but keep an eye on the pond over the next few " "days."),
    },
    RISK_LOW: {
        "css": "risk-uncertain",
        "icon": "&#10068;",
        "pill": "LOW CONFIDENCE",
        "caption": "Result is uncertain",
        "hint": (
            "The model is not sure enough to clear this fish. Retake the photo in "
            "daylight and run the analysis again."
        ),
    },
}

_FALLBACK_STYLE = {
    "css": "risk-uncertain",
    "icon": "&#10068;",
    "pill": "UNCERTAIN",
    "caption": "Needs a better photo",
    "hint": "Take a new photo in daylight with the whole fish in frame and try again.",
}


def render_risk_card(result: Prediction) -> None:
    """Draw the risk banner for one backend result."""
    risk = str(result["risk"]).upper().strip()
    is_healthy = str(result["predicted_class"]) == HEALTHY_CLASS
    styles = _HEALTHY_STYLES if is_healthy else _DISEASE_STYLES
    style = styles.get(risk, _FALLBACK_STYLE)

    message = str(result["message"]).strip()
    if not message:
        message = "No recommended action was returned for this image."

    st.markdown(
        f"""
        <div class="ah-risk {style['css']}">
          <div class="ah-risk-top">
            <span class="ah-risk-pill">{style['icon']} {html.escape(style['pill'])}</span>
            <span class="ah-risk-caption">{html.escape(style['caption'])}</span>
          </div>
          <div class="ah-risk-action">{html.escape(message)}</div>
          <div class="ah-risk-hint">{html.escape(style['hint'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
