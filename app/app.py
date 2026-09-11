"""Streamlit entry point.

app.py -> predict(image) -> result

Uses the mock predictor until a trained checkpoint is available; flip
USE_MOCK_BACKEND to False once src/train.py has produced
models/final_model.pth and src.predict.predict is implemented.

This file only orchestrates the UI: no training code, no model internals, and
no duplicated class mapping (that lives in src/config.py).

Owner: Student 3 + Student 4
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent


def _bootstrap_import_path() -> None:
    """Make `app.*` and `src.*` importable under `streamlit run app/app.py`.

    Streamlit puts the script's own folder (app/) first on sys.path, so a plain
    `import app` resolves to *this file* rather than the app/ package, and
    `from app.components... import ...` dies with "'app' is not a package".
    Dropping app/ from sys.path and putting the repo root on instead fixes the
    run command documented in docs/team/student_3.md.
    """
    for entry in list(sys.path):
        try:
            if entry and Path(entry).resolve() == APP_DIR:
                sys.path.remove(entry)
        except (OSError, ValueError):  # unreadable or malformed sys.path entry
            continue

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    # If this file already imported itself as a bare module named "app",
    # discard it so the real package can be imported in its place.
    shadow = sys.modules.get("app")
    if shadow is not None and not hasattr(shadow, "__path__"):
        del sys.modules["app"]


_bootstrap_import_path()

from app.components.dashboard import render_dashboard  # noqa: E402
from app.components.result_card import render_result_card  # noqa: E402
from app.components.risk_card import render_risk_card  # noqa: E402
from app.components.summary_panel import render_summary_panel  # noqa: E402
from app.components.upload import UploadResult, render_upload_card  # noqa: E402
from src.predict import Prediction  # noqa: E402

# Flip to False to use the real backend. Nothing else in the UI changes.
USE_MOCK_BACKEND = True

GENERIC_ERROR = "Analysis failed. Please try another image."
CSS_PATH = ROOT_DIR / "app" / "assets" / "style.css"

st.set_page_config(
    page_title="AquaHealth AI - Smart Fish Disease Detection",
    page_icon=":fish:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_predictor() -> Callable[[np.ndarray], Prediction]:
    """Return the prediction callable.

    Cached as a resource so the real model is loaded into memory once per
    server process. The mock costs nothing to load but takes the same path,
    which keeps the swap to a one-line change.
    """
    if not USE_MOCK_BACKEND:
        from src.predict import predict

        return predict

    from app.mock_prediction import predict as mock_predict

    return mock_predict


@st.cache_data(show_spinner=False)
def load_css(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def inject_css() -> None:
    try:
        st.markdown(f"<style>{load_css(str(CSS_PATH))}</style>", unsafe_allow_html=True)
    except OSError:
        st.warning("Stylesheet not found - the page will render unstyled.")


def render_header() -> None:
    st.markdown(
        """
        <div class="ah-hero">
          <span class="ah-hero-badge">&#128031; AI-based disease indication</span>
          <h1>AQUAHEALTH AI</h1>
          <p class="ah-sub">Smart Fish Disease Detection</p>
          <p class="ah-lede">
            Take a photo of a fish from your pond, upload it here, and get an
            instant indication of whether it shows signs of disease - plus what
            to do next. Built for fish farmers, no technical knowledge needed.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    st.markdown('<div class="ah-section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ah-steps">
          <div class="ah-step">
            <div class="ah-step-num">1</div>
            <div class="ah-step-title">Take a photo</div>
            <div class="ah-step-text">One fish, side view, in daylight. Keep the
              whole body in the picture.</div>
          </div>
          <div class="ah-step">
            <div class="ah-step-num">2</div>
            <div class="ah-step-title">Upload it</div>
            <div class="ah-step-text">Choose the photo from your phone or
              computer. JPG and PNG both work.</div>
          </div>
          <div class="ah-step">
            <div class="ah-step-num">3</div>
            <div class="ah-step-title">AI checks the skin &amp; fins</div>
            <div class="ah-step-text">The model compares your photo against
              known disease patterns in seconds.</div>
          </div>
          <div class="ah-step">
            <div class="ah-step-num">4</div>
            <div class="ah-step-title">Get your action plan</div>
            <div class="ah-step-text">You see the likely condition, a risk level
              and the step to take next.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="ah-footer">
          <strong>Important:</strong> AquaHealth AI provides an
          <strong>AI-based disease indication</strong> from a photograph. It is
          not a certified veterinary diagnosis. For treatment decisions, sick
          stock or any large-scale mortality, always consult a qualified
          aquaculture veterinarian or your local fisheries officer.
          <br><br>
          AquaHealth AI &middot; Hackathon prototype &middot; Results are
          currently generated by a demo backend.
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_analysis() -> None:
    """Clear any previous prediction so a stale result is never shown."""
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)


def run_analysis(image: np.ndarray) -> None:
    predict = load_predictor()
    reset_analysis()

    with st.spinner("Analysing your photo - checking skin, fins and gills"):
        try:
            result = predict(image)
        except Exception:  # any backend failure becomes one calm user message
            st.session_state["error"] = GENERIC_ERROR
            return

    if not isinstance(result, dict) or "predicted_class" not in result:
        st.session_state["error"] = GENERIC_ERROR
        return

    st.session_state["result"] = result


def render_analysis_section(upload: UploadResult) -> None:
    """Analyse button plus the results for the current upload."""
    st.markdown(
        '<div class="ah-section-title">Step 2 &middot; Run the analysis</div>',
        unsafe_allow_html=True,
    )

    if upload.is_blocked:
        st.markdown(
            '<div class="ah-msg err">This image cannot be analysed. Please upload a '
            "valid JPG/PNG photo of a fish.</div>",
            unsafe_allow_html=True,
        )
        return

    action_col, reset_col, _ = st.columns([1.3, 1, 2.2])
    with action_col:
        analyse = st.button("Analyse this fish", type="primary", use_container_width=True)
    with reset_col:
        if st.button("New analysis", use_container_width=True):
            reset_analysis()
            st.rerun()

    if analyse and upload.array is not None:
        run_analysis(upload.array)

    if st.session_state.get("error"):
        st.markdown(
            f'<div class="ah-msg err">{st.session_state["error"]} If it keeps failing, '
            "take a fresh photo in daylight and try once more.</div>",
            unsafe_allow_html=True,
        )
        return

    result = st.session_state.get("result")
    if result:
        st.markdown(
            '<div class="ah-section-title">Step 3 &middot; Your result</div>',
            unsafe_allow_html=True,
        )
        render_result_card(result)
        render_risk_card(result)


def main() -> None:
    inject_css()
    render_header()

    st.markdown('<div class="ah-section-title">Recent activity</div>', unsafe_allow_html=True)
    render_summary_panel()

    render_how_it_works()

    upload = render_upload_card()

    # A different file was chosen: drop the previous prediction immediately.
    if st.session_state.get("last_file_id") != upload.file_id:
        st.session_state["last_file_id"] = upload.file_id
        reset_analysis()

    if upload.has_image or upload.is_blocked:
        render_analysis_section(upload)
    else:
        # Student 4's panel: empty-state help and the supported class list.
        render_dashboard()

    render_footer()


if __name__ == "__main__":
    main()
