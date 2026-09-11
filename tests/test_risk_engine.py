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
