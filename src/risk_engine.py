"""Converts model confidence into a risk level for the frontend.

Owner: Student 2
"""

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MODERATE

RISK_LOW = "LOW"
RISK_MODERATE = "MODERATE"
RISK_HIGH = "HIGH"

RISK_MESSAGES = {
    RISK_LOW: "Low-confidence prediction; result is uncertain.",
    RISK_MODERATE: "Moderate-confidence disease indication.",
    RISK_HIGH: "High-confidence disease indication.",
}


def get_risk_level(confidence: float) -> str:
    """Map a model confidence score in [0, 1] to a risk level.

    < 0.50            -> LOW
    0.50 - 0.80 (incl) -> MODERATE
    > 0.80            -> HIGH
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    if confidence < RISK_THRESHOLD_MODERATE:
        return RISK_LOW
    if confidence <= RISK_THRESHOLD_HIGH:
        return RISK_MODERATE
    return RISK_HIGH


def get_risk_message(risk: str) -> str:
    return RISK_MESSAGES[risk]
