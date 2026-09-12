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
