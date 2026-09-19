# ruff: noqa: E501  (HTML/CSS string literals; black does not reflow strings)
"""Result presentation. Everything shown comes from the backend `PredictionResult` dict.

Owner: Student 3
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.theme import confidence_label, gauge_svg

RISK_TITLES = {
    "LOW": "LOW CONFIDENCE / UNCERTAIN",
    "MODERATE": "MODERATE RISK",
    "HIGH": "HIGH RISK",
}
RISK_KICKERS = {
    "LOW": "AI confidence is low",
    "MODERATE": "Possible condition detected",
    "HIGH": "High-confidence classification",
}
BADGE_VAR = {"LOW": "--low", "MODERATE": "--moderate", "HIGH": "--high", "HEALTHY": "--healthy"}
GLOW = {
    "HEALTHY": "glow-healthy",
    "HIGH": "glow-high",
    "MODERATE": "glow-moderate",
    "LOW": "glow-low",
}
RISK_MEANING = {
    "LOW": "Confidence below 50 %: the model is uncertain. Manual inspection is recommended.",
    "MODERATE": "Confidence 50–80 %: possible condition detected. Observe, isolate and re-check.",
    "HIGH": "Confidence above 80 %: high-confidence classification. Act and consult a professional.",
    "HEALTHY": "AI classification indicates a healthy appearance. It does not guarantee the fish is disease-free — keep monitoring.",
}


def _esc(text: Any) -> str:
    return html.escape(str(text))


def risk_badge(result: dict[str, Any]) -> str:
    """Badge label: a confident Healthy Fish prediction is HEALTHY, otherwise the backend risk."""
    if result.get("healthy") and result["risk"] != "LOW":
        return "HEALTHY"
    return str(result["risk"])


def quality_summary(result: dict[str, Any]) -> tuple[bool, str]:
    """(ok, text) derived only from the backend `quality` flags."""
    quality = result.get("quality") or {}
    issues = []
    if quality.get("dark"):
        issues.append("too dark")
    if quality.get("blurry"):
        issues.append("blurry")
    if not issues:
        return True, "Suitable for analysis"
    return False, "Image may be " + " and ".join(issues)


def render_result_panel(result: dict[str, Any]) -> None:
    """The main analysis panel: predicted condition, very large confidence, risk badge, message."""
    badge = risk_badge(result)
    kicker = (
        "AI classification indicates a healthy appearance"
        if badge == "HEALTHY"
        else RISK_KICKERS[result["risk"]]
    )
    title = "HEALTHY" if badge == "HEALTHY" else RISK_TITLES[result["risk"]]
    st.markdown(
        f"""
<div class="aq-frame {GLOW[badge]}">
  <span class="aq-num">AI ANALYSIS</span>
  <div class="aq-label" style="margin-top:6px;">Predicted condition · {_esc(kicker)}</div>
  <div class="aq-big">{_esc(result["predicted_class"])}</div>
  <div class="aq-pct" style="color:var({BADGE_VAR[badge]});text-shadow:0 0 30px var({BADGE_VAR[badge]});">{confidence_label(result["confidence"])}</div>
  <div class="aq-label">Model confidence</div>
  <div style="margin:14px 0 10px 0;"><span class="aq-badge {badge}">{_esc(title)}</span></div>
  <p style="margin:0;">{_esc(result["message"])}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_recommendation(result: dict[str, Any]) -> None:
    css = "aq-warn" if result["risk"] == "LOW" else "aq-note"
    st.markdown(
        f'<div class="{css}"><span class="aq-num">RECOMMENDATION</span>'
        f'<p style="margin:6px 0 0 0;font-size:1.02rem;">{_esc(result["recommendation"])}</p></div>',
        unsafe_allow_html=True,
    )


def render_low_confidence_notice(result: dict[str, Any]) -> None:
    if result["risk"] != "LOW":
        return
    st.markdown(
        '<div class="aq-err"><span class="aq-num" style="color:#ffd0d8">ANALYSIS UNCERTAIN</span>'
        "<p style='margin:6px 0 0 0;'><b>The model is not sufficiently confident about this image.</b> "
        "Try a clearer image with better lighting, the fish visible at a useful scale, and less blur or obstruction.</p></div>",
        unsafe_allow_html=True,
    )


def render_quality_notice(result: dict[str, Any]) -> None:
    ok, text = quality_summary(result)
    if ok:
        st.markdown(
            f'<div class="aq-note"><span class="aq-num">IMAGE QUALITY</span><br>✓ {_esc(text)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="aq-warn"><span class="aq-num" style="color:#ffe1b0">IMAGE QUALITY</span><br>⚠ {_esc(text).upper()} — '
            "the prediction was still computed, but a sharper, brighter photo is more reliable.</div>",
            unsafe_allow_html=True,
        )


def render_risk_panel(result: dict[str, Any]) -> None:
    badge = risk_badge(result)
    st.markdown(
        f"""
<div class="aq-frame {GLOW[badge]}" style="display:flex;gap:26px;align-items:center;flex-wrap:wrap;">
  <div>{gauge_svg(float(result["confidence"]), BADGE_VAR[badge])}</div>
  <div style="flex:1 1 320px;">
  <span class="aq-num">SCREENING RISK</span>
  <div style="margin:10px 0;"><span class="aq-badge {badge}">{badge}</span></div>
  <p style="margin:0;">{RISK_MEANING[badge]}</p>
  <p class="aq-disclaimer" style="margin:10px 0 0 0;">Application-defined screening thresholds (&lt;50 % LOW · 50–80 % MODERATE · &gt;80 % HIGH). AI screening aid — not a veterinary diagnosis.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_probabilities(result: dict[str, Any]) -> None:
    ranked = result.get("ranked_predictions") or []
    rows = []
    for i, item in enumerate(ranked):
        p = float(item["probability"])
        bar = (
            "healthy"
            if (i == 0 and item["class_name"] == "Healthy Fish")
            else ("top" if i == 0 else "")
        )
        rows.append(
            f'<div class="aq-row {"top" if i == 0 else ""}"><span>{_esc(item["class_name"])}</span>'
            f'<div class="aq-bar {bar}"><span style="width:{p:.1%}"></span></div>'
            f"<span style='text-align:right;font-variant-numeric:tabular-nums'>{p:.2%}</span></div>"
        )
    st.markdown(
        f'<div class="aq-frame"><span class="aq-num">MODEL CONFIDENCE DISTRIBUTION</span>'
        f'<div class="aq-label" style="margin:6px 0 10px 0;">Softmax probabilities returned by the model for all eight classes (sum = 100 %)</div>'
        f"{''.join(rows)}</div>",
        unsafe_allow_html=True,
    )


def render_visual_explanation(result: dict[str, Any], explanation: dict[str, Any] | None) -> None:
    """Grad-CAM section: original vs attention overlay. Wording never claims biology."""
    st.markdown(
        '<div class="aq-section"><span class="aq-num">WHY DID THE MODEL PREDICT THIS?</span>'
        '<div class="aq-h2">AI activation visualization</div></div>',
        unsafe_allow_html=True,
    )
    if not explanation or "overlay_input" not in explanation:
        message = (explanation or {}).get("error", "AI visual explanation unavailable.")
        st.markdown(f'<div class="aq-warn">{_esc(message)}</div>', unsafe_allow_html=True)
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="aq-label">Original (model input · 224 × 224 after CLAHE · resize · crop)</div>',
            unsafe_allow_html=True,
        )
        st.image(explanation["model_input"], use_container_width=True)
    with c2:
        st.markdown(
            f'<div class="aq-label accent">Grad-CAM overlay · target class "{_esc(explanation["target_class"])}"</div>',
            unsafe_allow_html=True,
        )
        st.image(explanation["overlay_input"], use_container_width=True)
    px, py = explanation.get("peak", (0.5, 0.5))
    st.markdown(
        f'<div class="aq-note"><p style="margin:0 0 6px 0;"><b>The highlighted regions represent areas with stronger model activation for the predicted class.</b> '
        f"Strongest activation at {px:.0%} across, {py:.0%} down of the analysed frame; computed with Grad-CAM on "
        f"<code>{_esc(explanation.get('target_layer', 'features[-1]'))}</code> of the frozen EfficientNet-B0.</p>"
        f"<p class='aq-disclaimer' style='margin:0;'>{_esc(explanation.get('note', ''))} This is a post-hoc model explanation and not guaranteed lesion localization.</p></div>",
        unsafe_allow_html=True,
    )


def render_classes_grid(class_names: list[str]) -> None:
    cards = "".join(
        f'<div class="aq-feature"><span class="aq-num">{i:02d}</span><b>{_esc(name)}</b>'
        f"<small>{'healthy appearance' if name == 'Healthy Fish' else 'disease condition'}</small></div>"
        for i, name in enumerate(class_names)
    )
    st.markdown(
        '<div class="aq-section"><span class="aq-num">SUPPORTED CLASSES</span><div class="aq-h2">Eight conditions the model screens for</div></div>'
        f'<div class="aq-grid4">{cards}</div>',
        unsafe_allow_html=True,
    )


def render_explanation(result: dict[str, Any], explanation: dict[str, Any] | None = None) -> None:
    has_cam = bool(explanation and "overlay_input" in explanation)
    cam_line = (
        "The highlighted regions in the visual explanation show areas with stronger model activation for the predicted class — evidence of model activation, not proof of a biological symptom."
        if has_cam
        else "No visual explanation could be generated for this image."
    )
    st.markdown(
        f"""
<div class="aq-frame">
<span class="aq-num">HOW THIS RESULT WAS PRODUCED</span>
<p style="margin:8px 0 0 0;"><b>The model classified this image as {_esc(result["predicted_class"])} with {confidence_label(result["confidence"])} confidence.</b></p>
<ol style="margin:8px 0 0 0;padding-left:20px;">
<li>Your image was preprocessed exactly as during validation (CLAHE contrast enhancement → resize 256 → centre-crop 224 → ImageNet normalisation).</li>
<li>EfficientNet-B0 produced eight class probabilities from the visual patterns in the image; the highest-probability class was selected.</li>
<li>The confidence is the model's own probability for that class — not an accuracy guarantee for this photo.</li>
<li>The risk level is an application-defined screening category (below 50 % LOW / UNCERTAIN, 50–80 % MODERATE, above 80 % HIGH); a Healthy Fish prediction is shown as HEALTHY, not as a disease risk.</li>
<li>{cam_line}</li>
<li>The recommendation is generic husbandry guidance for the predicted class. AquaHealth AI is an AI screening aid, not a certified veterinary diagnosis.</li>
</ol>
<p class="aq-disclaimer" style="margin:10px 0 0 0;">Model {_esc(result.get("model_version", "n/a"))} · preprocessing {_esc(result.get("preprocessing_version", "n/a"))} · device {_esc(result.get("device", "n/a"))}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    for warning in result.get("warnings") or []:
        st.caption(f"Note: {warning}")
