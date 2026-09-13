# ruff: noqa: E501  (HTML/CSS string literals; black does not reflow strings)
"""AquaHealth AI — Streamlit entry point (presentation layer only).

    HOME → ANALYZE (upload → preview → scanning) → RESULTS → DASHBOARD → HOW IT WORKS

The app holds no model logic: it loads the frozen final checkpoint once (cached) and renders what
`src.predict.Predictor.predict(image)` returns, plus the Grad-CAM explanation produced by
`src.explainability.generate_gradcam` on the same frozen model. Without a checkpoint it shows a clear
"model not available" state — there is no mock predictor and nothing is ever trained here.

Run:  streamlit run app/main.py
Checkpoint: models/final_model.pth (or the AQUAHEALTH_CHECKPOINT environment variable).

Owner: Student 3 + Student 4
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# `streamlit run app/main.py` puts app/ (not the project root) on sys.path.
sys.path.insert(0, str(ROOT))

from app.components.dashboard import render_dashboard  # noqa: E402
from app.components.how_it_works import render_how_it_works  # noqa: E402
from app.components.result_card import (  # noqa: E402
    render_classes_grid,
    render_explanation,
    render_low_confidence_notice,
    render_probabilities,
    render_quality_notice,
    render_recommendation,
    render_result_panel,
    render_risk_panel,
    render_visual_explanation,
)
from app.components.upload import image_facts, render_upload  # noqa: E402
from app.state import SessionHistory  # noqa: E402
from app.theme import CSS, analyzing_html, hero_html, page_header, scene_html  # noqa: E402
from src.config import CHECKPOINT_PATH, CLASS_NAMES  # noqa: E402
from src.explainability import EXPLANATION_NOTE, generate_gradcam  # noqa: E402
from src.predict import ModelLoadError, Predictor  # noqa: E402

logger = logging.getLogger("aquahealth.app")
CHECKPOINT_ENV = "AQUAHEALTH_CHECKPOINT"
PAGES = [
    ("01", "Home"),
    ("02", "Analyze"),
    ("03", "Results"),
    ("04", "Dashboard"),
    ("05", "How it works"),
]
DISCLAIMER = "AI screening support only — not a veterinary diagnosis."


# --- model ---------------------------------------------------------------------------------------


def checkpoint_location() -> Path:
    return Path(os.environ.get(CHECKPOINT_ENV, CHECKPOINT_PATH))


@st.cache_resource(show_spinner=False)
def load_predictor(path: str) -> Predictor:
    """The final checkpoint is loaded once per process and reused for every analysis."""
    return Predictor(path)


def get_predictor() -> tuple[Predictor | None, str]:
    path = checkpoint_location()
    if not path.is_file():
        return None, "The final model checkpoint is not installed on this machine."
    try:
        return load_predictor(str(path)), ""
    except ModelLoadError as exc:
        logger.exception("checkpoint could not be loaded")
        return None, f"The model checkpoint could not be loaded ({type(exc).__name__})."


# --- state ---------------------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("history", SessionHistory())
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("explanation", None)
    st.session_state.setdefault("image_bytes", None)
    st.session_state.setdefault("image_name", "")
    st.session_state.setdefault("upload_key", 0)
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("view", "Original")


def reset_analysis() -> None:
    """Clear the current image, result and explanation; the session history is preserved."""
    st.session_state["result"] = None
    st.session_state["explanation"] = None
    st.session_state["image_bytes"] = None
    st.session_state["image_name"] = ""
    st.session_state["view"] = "Original"
    st.session_state["upload_key"] += 1


def go(page: str) -> None:
    st.session_state["page"] = page


def start_new_analysis() -> None:
    reset_analysis()
    go("Analyze")


def set_view(view: str) -> None:
    st.session_state["view"] = view


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_analysis(predictor: Predictor) -> None:
    """Exactly one backend call per Analyze click (+ one Grad-CAM call on the same frozen model)."""
    placeholder = st.empty()
    placeholder.markdown(analyzing_html(), unsafe_allow_html=True)
    try:
        result = predictor.predict(st.session_state["image_bytes"]).to_dict()
    except Exception:  # noqa: BLE001 - never show a stack trace to the user
        logger.exception("inference failed")
        result = {"status": "error", "error": "An unexpected error occurred during analysis."}
    explanation: dict[str, Any] | None = None
    if result["status"] == "ok":
        try:
            cam = generate_gradcam(predictor, st.session_state["image_bytes"])
            explanation = {
                "target_class": cam.target_class,
                "overlay_original": _png_bytes(cam.overlay_original),
                "overlay_input": _png_bytes(cam.overlay_input),
                "model_input": _png_bytes(cam.model_input),
                "peak": cam.peak,
                "target_layer": cam.target_layer,
                "note": EXPLANATION_NOTE,
            }
        except Exception:  # noqa: BLE001 - the explanation is optional; prediction must survive
            logger.exception("visual explanation failed")
            explanation = {"error": "AI visual explanation unavailable."}
    placeholder.empty()
    st.session_state["result"] = result
    st.session_state["explanation"] = explanation
    st.session_state["view"] = "Original"
    if result["status"] == "ok":
        st.session_state["history"].add(st.session_state["image_name"], result)
    go("Results")


# --- chrome --------------------------------------------------------------------------------------


def render_nav(predictor: Predictor | None) -> None:
    brand, *slots = st.columns([1.6] + [1] * len(PAGES), gap="small")
    with brand:
        st.markdown(
            '<div class="aq-brand"><span class="logo"></span><div><b>AquaHealth AI</b><br>'
            '<span class="aq-label">Fish health intelligence</span></div></div>',
            unsafe_allow_html=True,
        )
    with st.container(key="aq-nav"):
        cols = st.columns(len(PAGES), gap="small")
        for col, (number, name) in zip(cols, PAGES, strict=True):
            active = st.session_state["page"] == name
            col.button(
                f"{number}  {name}",
                key=f"nav-{name}",
                type="primary" if active else "secondary",
                use_container_width=True,
                on_click=go,
                args=(name,),
            )
    status = (
        f"Model {predictor.model_version} · cached · {predictor.device}"
        if predictor
        else "Model not available"
    )
    st.markdown(
        f'<div class="aq-navline"></div><div class="aq-label" style="margin:-14px 0 14px 0;">{status} · analyses this session: {st.session_state["history"].count}</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f'<div class="aq-footer"><span><b>AquaHealth AI</b> · AI screening support for aquaculture.</span>'
        f"<span>{DISCLAIMER}</span></div>",
        unsafe_allow_html=True,
    )


def render_model_missing(reason: str) -> None:
    st.markdown(
        f'<div class="aq-err"><b>Model not available.</b> {reason}<br>Install the final checkpoint at '
        "<code>models/final_model.pth</code> and reload. No prediction is possible without it — this "
        "application never shows placeholder results.</div>",
        unsafe_allow_html=True,
    )


# --- pages ---------------------------------------------------------------------------------------


def render_home(predictor: Predictor | None, reason: str, class_names: list[str]) -> None:
    st.markdown(
        hero_html(
            "AI-powered aquaculture intelligence",
            "See fish health<br>through AI",
            "Upload a fish image and receive an AI-assisted 8-class visual screening result with confidence, "
            "risk interpretation and a visual model explanation.",
        ),
        unsafe_allow_html=True,
    )
    if predictor is None:
        render_model_missing(reason)
    _, c1, c2, _ = st.columns([1.4, 1, 1, 1.4], gap="small")
    c1.button(
        "Analyze fish",
        type="primary",
        use_container_width=True,
        on_click=start_new_analysis,
        key="hero-analyze",
    )
    c2.button(
        "How it works",
        use_container_width=True,
        on_click=go,
        args=("How it works",),
        key="hero-how",
    )
    st.markdown(scene_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="aq-section"><span class="aq-num">DESIGNED FOR EARLIER FISH HEALTH DECISIONS</span></div>'
        '<div class="aq-grid4" style="margin-top:12px;">'
        '<div class="aq-feature"><span class="aq-num">01 · AI VISION</span><b>EfficientNet-B0</b><small>ImageNet-pretrained backbone fine-tuned on a de-duplicated, leakage-controlled training split</small></div>'
        '<div class="aq-feature"><span class="aq-num">02 · 8-CLASS SCREENING</span><b>7 disease classes + Healthy Fish</b><small>softmax probabilities for every class, selected by the model — never by the interface</small></div>'
        '<div class="aq-feature"><span class="aq-num">03 · VISUAL EXPLANATION</span><b>Grad-CAM activation visualization</b><small>see the regions with stronger model activation for the predicted class</small></div>'
        '<div class="aq-feature"><span class="aq-num">04 · LOCAL SCREENING</span><b>Runs locally through the application</b><small>your image is processed by the frozen model inside this app; no external service</small></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    render_classes_grid(class_names)
    render_footer()


def render_workspace(predictor: Predictor | None, reason: str) -> None:
    st.markdown(
        page_header(
            "02 · ANALYZE",
            "AI inspection console",
            "Fish image analysis",
            "Upload one clear fish image for AI screening.",
        ),
        unsafe_allow_html=True,
    )
    if predictor is None:
        render_model_missing(reason)
        return
    data = st.session_state["image_bytes"]
    if data is None:
        st.markdown(
            '<span class="aq-num">DROP FISH IMAGE · JPG / JPEG / PNG / WEBP / BMP</span>',
            unsafe_allow_html=True,
        )
        upload = render_upload(key=f"uploader-{st.session_state['upload_key']}")
        if upload is not None:
            st.session_state["image_bytes"], st.session_state["image_name"] = upload
            st.rerun()
        st.markdown(scene_html(small=True, nodes=False), unsafe_allow_html=True)
        st.markdown(
            '<div class="aq-grid3" style="margin-top:14px;">'
            '<div class="aq-feature"><span class="aq-num">TIP 01</span><b>One fish per photo</b><small>side view, filling the frame</small></div>'
            '<div class="aq-feature"><span class="aq-num">TIP 02</span><b>Good light, in focus</b><small>no strong shadows or flash glare; keep the lesion, gill or fin sharp</small></div>'
            '<div class="aq-feature"><span class="aq-num">TIP 03</span><b>Any resolution</b><small>224 px or larger is best; the backend checks for dark or blurry images</small></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        render_footer()
        return
    if "Resolution" not in image_facts(data):
        # Not decodable as an image: let the backend produce its canonical error result.
        run_analysis(predictor)
        st.rerun()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<span class="aq-num">FISH IMAGE</span>', unsafe_allow_html=True)
        st.image(data, use_container_width=True)
    with right:
        facts = image_facts(data)
        rows = "".join(
            f'<div class="aq-row" style="grid-template-columns:1fr 1.5fr;margin:6px 0;"><span class="aq-label">{k}</span><span>{v}</span></div>'
            for k, v in {"Filename": st.session_state["image_name"], **facts}.items()
        )
        st.markdown(
            f'<div class="aq-frame"><span class="aq-num">IMAGE INFORMATION</span><div style="margin-top:8px;">{rows}</div>'
            '<p class="aq-disclaimer" style="margin:10px 0 0 0;">Quality (dark / blur) is assessed by the backend during analysis.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Analyze image", type="primary", use_container_width=True, key="ws-analyze"):
            run_analysis(predictor)
            st.rerun()
        st.button(
            "Choose another", use_container_width=True, on_click=reset_analysis, key="ws-change"
        )
    render_footer()


def render_results(predictor: Predictor | None) -> None:
    result: dict[str, Any] | None = st.session_state["result"]
    if result is None:
        st.markdown(
            page_header(
                "03 · RESULTS",
                "AI analysis",
                "No result yet",
                "Analyse a fish image first; the result will appear here.",
            ),
            unsafe_allow_html=True,
        )
        st.button("Go to analyze", type="primary", on_click=start_new_analysis, key="res-go")
        render_footer()
        return
    if result["status"] != "ok":
        st.markdown(
            page_header("03 · RESULTS", "AI analysis", "Image could not be analysed", ""),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="aq-err"><p style="margin:0;">{result.get("error", "unknown error")}</p>'
            "<p class='aq-disclaimer' style='margin:8px 0 0 0;'>Supported formats: JPG, JPEG, PNG, WEBP, BMP; the file must be a readable image of at least 8 × 8 px.</p></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Choose another image", type="primary", on_click=start_new_analysis, key="res-retry"
        )
        render_footer()
        return

    explanation = st.session_state.get("explanation")
    st.markdown(
        page_header(
            "03 · RESULTS",
            "AI analysis complete",
            "AI health assessment",
            "Real output of the frozen final model for the uploaded image.",
        ),
        unsafe_allow_html=True,
    )
    render_low_confidence_notice(result)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        has_cam = bool(explanation and "overlay_original" in explanation)
        with st.container(key="aq-toggle"):
            t1, t2 = st.columns(2, gap="small")
            t1.button(
                "Original",
                use_container_width=True,
                type="primary" if st.session_state["view"] == "Original" else "secondary",
                on_click=set_view,
                args=("Original",),
                key="view-original",
            )
            t2.button(
                "AI explanation",
                use_container_width=True,
                type="primary" if st.session_state["view"] == "AI explanation" else "secondary",
                on_click=set_view,
                args=("AI explanation",),
                key="view-cam",
                disabled=not has_cam,
            )
        if st.session_state["view"] == "AI explanation" and has_cam and explanation:
            st.image(
                explanation["overlay_original"],
                caption=f"AI activation visualization — {st.session_state['image_name']}",
                use_container_width=True,
            )
        elif "Resolution" in image_facts(st.session_state["image_bytes"]):
            st.image(
                st.session_state["image_bytes"],
                caption=st.session_state["image_name"],
                use_container_width=True,
            )
        render_quality_notice(result)
        st.write("")
        st.button(
            "Analyze another fish",
            type="primary",
            use_container_width=True,
            on_click=start_new_analysis,
            key="res-again",
        )
    with right:
        render_result_panel(result)
        render_recommendation(result)
    st.write("")
    render_risk_panel(result)
    render_visual_explanation(result, explanation)
    st.markdown('<div class="aq-section"></div>', unsafe_allow_html=True)
    render_probabilities(result)
    st.write("")
    render_explanation(result, explanation)
    render_footer()


def main() -> None:
    st.set_page_config(
        page_title="AquaHealth AI", page_icon="🐟", layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    predictor, reason = get_predictor()
    render_nav(predictor)
    class_names = predictor.class_names if predictor is not None else list(CLASS_NAMES)
    page = st.session_state["page"]
    if page == "Home":
        render_home(predictor, reason, class_names)
    elif page == "Analyze":
        render_workspace(predictor, reason)
    elif page == "Results":
        render_results(predictor)
    elif page == "Dashboard":
        st.markdown(
            page_header(
                "04 · DASHBOARD",
                "Session intelligence",
                "What this session has analysed",
                "Every number below comes from the analyses you ran in this session.",
            ),
            unsafe_allow_html=True,
        )
        render_dashboard(st.session_state["history"], class_names, start_new_analysis)
        render_footer()
    else:
        st.markdown(
            page_header(
                "05 · HOW IT WORKS",
                "Pipeline",
                "How AquaHealth AI works",
                "From upload to recommendation — every step matches the implementation.",
            ),
            unsafe_allow_html=True,
        )
        render_how_it_works(ROOT, predictor.model_version if predictor else None)
        render_classes_grid(class_names)
        render_footer()


if __name__ == "__main__":
    main()
