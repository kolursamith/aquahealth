"""Pure presentation helpers of the Streamlit app (no Streamlit runtime needed)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.components.result_card import quality_summary, risk_badge
from app.components.upload import image_facts
from app.theme import CSS, analyzing_html, gauge_svg, hero_html, scene_html


def test_risk_badge_shows_healthy_only_for_confident_healthy_predictions():
    assert risk_badge({"healthy": True, "risk": "HIGH"}) == "HEALTHY"
    assert risk_badge({"healthy": True, "risk": "MODERATE"}) == "HEALTHY"
    assert risk_badge({"healthy": True, "risk": "LOW"}) == "LOW"
    assert risk_badge({"healthy": False, "risk": "HIGH"}) == "HIGH"
    assert risk_badge({"healthy": False, "risk": "MODERATE"}) == "MODERATE"


def test_quality_summary_uses_backend_flags_only():
    assert quality_summary({"quality": {"dark": False, "blurry": False}}) == (
        True,
        "Suitable for analysis",
    )
    assert (
        quality_summary({"quality": {"dark": True, "blurry": False}})[1] == "Image may be too dark"
    )
    assert (
        quality_summary({"quality": {"dark": True, "blurry": True}})[1]
        == "Image may be too dark and blurry"
    )
    assert quality_summary({})[0] is True


def test_gauge_reflects_the_actual_confidence_and_never_rounds_to_certainty():
    assert "87.4%" in gauge_svg(0.874, "--high")
    assert ">99.9%" in gauge_svg(0.9997, "--high") and "100.0%" not in gauge_svg(0.9997, "--high")
    assert "100.0%" in gauge_svg(1.0, "--high")
    assert "0.0%" in gauge_svg(0.0, "--low")


def test_hero_and_scene_html_escape_text_and_carry_no_remote_assets():
    page = hero_html("k <x>", "See fish health<br>through AI", "text & more")
    assert "k &lt;x&gt;" in page and "text &amp; more" in page and "<br>" in page
    for fragment in (
        CSS,
        scene_html(),
        scene_html(small=True, nodes=False),
        analyzing_html(),
        page,
    ):
        # the only URL allowed is the SVG XML namespace; no remote scripts, fonts, images
        stripped = fragment.replace('xmlns="http://www.w3.org/2000/svg"', "")
        assert "http://" not in stripped and "https://" not in stripped
        assert "<script" not in stripped and "@import" not in stripped and "src=" not in stripped
    assert "%" not in analyzing_html().replace("100%", "")  # no fake progress percentages


def test_image_facts_reports_real_metadata_and_tolerates_garbage():
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(buffer, "PNG")
    facts = image_facts(buffer.getvalue())
    assert facts["Resolution"] == "40 × 30 px" and facts["Format"] == "PNG"
    assert "Resolution" not in image_facts(b"not an image")
