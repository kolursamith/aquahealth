# ruff: noqa: E501  (HTML/CSS string literals; black does not reflow strings)
"""'How it works' page: the real pipeline as a glowing timeline, model facts from artifacts, limits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

STEPS = [
    ("01", "Upload", "Fish image — JPG, JPEG, PNG, WEBP or BMP, one fish per photo"),
    (
        "02",
        "Quality check",
        "Image validation: decode → RGB → dark/blur screening (advisory, never blocks)",
    ),
    (
        "03",
        "Preprocessing",
        "CLAHE · Resize 256 · CenterCrop 224 · ImageNet normalization — identical to validation",
    ),
    (
        "04",
        "AI vision",
        "EfficientNet-B0, ImageNet-pretrained, fine-tuned on the AquaHealth training split (frozen final model)",
    ),
    ("05", "Classification", "8 classes: seven disease conditions + Healthy Fish (softmax)"),
    ("06", "Confidence", "The model's probability for the selected class"),
    (
        "07",
        "Risk interpretation",
        "Application-defined: <50 % LOW / UNCERTAIN · 50–80 % MODERATE · >80 % HIGH · Healthy shown as HEALTHY",
    ),
    (
        "08",
        "Visual explanation",
        "Grad-CAM activation map on the last convolutional block of the same frozen model",
    ),
    (
        "09",
        "Recommendation",
        "Farmer-facing husbandry guidance for the predicted class — not a treatment prescription",
    ),
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def render_how_it_works(root: Path, model_info: str | None) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            '<div class="aq-timeline">'
            + "".join(
                f'<div class="aq-tl"><span class="aq-num">{n}</span><b>{t}</b><small>{d}</small></div>'
                for n, t, d in STEPS
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    with right:
        selection = load_json(root / "results" / "final_model_selection.json")
        final = load_json(root / "results" / "final_test" / "metrics.json")
        rows = ""
        if selection:
            v = selection["validation_metrics"]
            rows += (
                f'<div class="aq-row" style="grid-template-columns:1fr 1fr"><span class="aq-label">Selected experiment</span><span>{selection["selected_experiment"]}</span></div>'
                f'<div class="aq-row" style="grid-template-columns:1fr 1fr"><span class="aq-label">Validation Macro-F1</span><span>{v["f1_macro"]:.4f}</span></div>'
                f'<div class="aq-row" style="grid-template-columns:1fr 1fr"><span class="aq-label">Validation accuracy</span><span>{v["accuracy"]:.4f}</span></div>'
                f'<div class="aq-row" style="grid-template-columns:1fr 1fr"><span class="aq-label">ECE</span><span>{selection["expected_calibration_error"]:.4f}</span></div>'
            )
        if final:
            m = final.get("metrics", final)
            rows += (
                f'<div class="aq-row" style="grid-template-columns:1fr 1fr"><span class="aq-label accent">Official test Macro-F1</span><span>{m["f1_macro"]:.4f}</span></div>'
                f'<div class="aq-row" style="grid-template-columns:1fr 1fr"><span class="aq-label accent">Official test accuracy</span><span>{m["accuracy"]:.4f}</span></div>'
            )
        else:
            rows += '<p class="aq-disclaimer" style="margin:8px 0 0 0;">No official test result recorded yet — the numbers above are validation results.</p>'
        st.markdown(
            f'<div class="aq-frame"><span class="aq-num">MODEL CARD</span>'
            f'<p class="aq-disclaimer" style="margin:6px 0 10px 0;">Values read from saved artifacts.{(" Loaded: " + model_info) if model_info else ""}</p>{rows}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="aq-frame" style="margin-top:14px;"><span class="aq-num">LIMITS</span><ul style="margin:8px 0 0 0;padding-left:18px;">'
            "<li>Trained on ~2.4 k photos of eight conditions from one public dataset; other species, other diseases and unusual lighting are outside its experience.</li>"
            "<li>An AI screening aid, not a certified veterinary diagnosis; it never prescribes treatment.</li>"
            "<li>One image → one prediction. No pond, water-quality or sensor monitoring; no guaranteed lesion localisation.</li></ul></div>",
            unsafe_allow_html=True,
        )
