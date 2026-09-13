import pytest

from src.risk_engine import get_risk_level


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.499, "LOW"),
        (0.50, "MODERATE"),
        (0.80, "MODERATE"),
        (0.801, "HIGH"),
        (1.0, "HIGH"),
        (0.0, "LOW"),
    ],
)
def test_risk_thresholds(confidence, expected):
    assert get_risk_level(confidence) == expected


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        get_risk_level(1.5)


def test_config_class_names_match_the_frozen_manifest_order():
    from src.config import CLASS_NAMES
    from src.manifest import CANONICAL_CLASSES

    assert list(CLASS_NAMES) == list(CANONICAL_CLASSES)


@pytest.mark.parametrize("risk", ["LOW", "MODERATE", "HIGH"])
def test_healthy_fish_message_is_distinct_from_disease_message(risk):
    from src.risk_engine import HEALTHY_CLASS, get_risk_message, is_healthy

    disease = get_risk_message(risk, "EUS Disease")
    healthy = get_risk_message(risk, HEALTHY_CLASS)
    assert get_risk_message(risk) == disease, "no class -> disease wording (legacy call)"
    assert is_healthy(HEALTHY_CLASS) and not is_healthy("EUS Disease") and not is_healthy(None)
    if risk == "LOW":
        assert healthy == disease and "uncertain" in healthy
    else:
        assert healthy != disease
        assert "no disease detected" in healthy and "Healthy Fish" in healthy
        assert "disease indication" in disease


@pytest.mark.parametrize(
    "confidence,expected", [(0.499, "LOW"), (0.50, "MODERATE"), (0.80, "MODERATE"), (0.801, "HIGH")]
)
def test_release_threshold_cases_49_9_50_80_80_1(confidence, expected):
    """The four boundary cases named in the release checklist."""
    assert get_risk_level(confidence) == expected


def test_recommendation_is_class_specific_and_low_confidence_asks_for_a_better_photo():
    from src.manifest import CANONICAL_CLASSES
    from src.risk_engine import (
        HEALTHY_CLASS,
        LOW_CONFIDENCE_RECOMMENDATION,
        RECOMMENDATIONS,
        get_recommendation,
    )

    assert set(RECOMMENDATIONS) == set(CANONICAL_CLASSES), "every class has guidance"
    assert get_recommendation("EUS Disease", "LOW") == LOW_CONFIDENCE_RECOMMENDATION
    assert get_recommendation("EUS Disease", "HIGH") == RECOMMENDATIONS["EUS Disease"]
    assert "No disease indicated" in get_recommendation(HEALTHY_CLASS, "MODERATE")
    assert "consult" in get_recommendation("unknown class", "HIGH")
    for text in RECOMMENDATIONS.values():
        assert "diagnos" not in text.lower() or "not" in text.lower(), "no diagnosis claims"
