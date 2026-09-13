"""Converts model confidence into a risk level and a message for the frontend.

The risk level is a function of confidence only (how sure the model is). The
message also depends on *what* was predicted: a confident "Healthy Fish" is a
reassuring finding, not a "disease indication", and must read distinctly.

Owner: Student 2
"""

from src.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MODERATE

RISK_LOW = "LOW"
RISK_MODERATE = "MODERATE"
RISK_HIGH = "HIGH"

HEALTHY_CLASS = "Healthy Fish"

RISK_MESSAGES = {
    RISK_LOW: "Low-confidence prediction; result is uncertain.",
    RISK_MODERATE: "Moderate-confidence disease indication.",
    RISK_HIGH: "High-confidence disease indication.",
}

HEALTHY_MESSAGES = {
    RISK_LOW: "Low-confidence prediction; result is uncertain.",
    RISK_MODERATE: "Moderate-confidence: no disease detected (Healthy Fish).",
    RISK_HIGH: "High-confidence: no disease detected (Healthy Fish).",
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


# Farmer-facing guidance per predicted class. Deliberately generic husbandry
# steps (isolate, observe, check water, consult a professional): the model is an
# AI indication, not a veterinary diagnosis, so no treatment is prescribed.
RECOMMENDATIONS = {
    "Bacterial Red Disease": (
        "Isolate fish showing red patches or ulcers, reduce stocking stress, check ammonia/"
        "nitrite and dissolved oxygen, and consult an aquaculture veterinarian before any "
        "antibiotic use."
    ),
    "Aeromoniasis": (
        "Separate affected fish, improve water exchange and aeration, remove uneaten feed, and "
        "seek veterinary advice; Aeromonas outbreaks are linked to poor water quality."
    ),
    "Bacterial Gill Disease": (
        "Check gill colour and breathing rate, lower stocking density, increase aeration and "
        "water exchange, and consult a fish-health professional."
    ),
    "EUS Disease": (
        "Isolate fish with deep ulcers immediately, avoid moving stock between ponds, keep "
        "water clean and stable, and report to the local fisheries/veterinary service."
    ),
    "Saprolegniasis": (
        "Remove fish with cotton-like growth, avoid handling injuries, keep water temperature "
        "stable and organic load low, and consult a professional about fungal treatment."
    ),
    "Parasitic Disease": (
        "Quarantine affected fish, inspect skin and gills for parasites, avoid introducing "
        "untreated new stock, and ask a veterinarian about an appropriate treatment."
    ),
    "White Tail Disease": (
        "Isolate fish with pale/whitish tails, stop transfers between tanks, disinfect nets and "
        "equipment, and contact a fish-health service; viral disease cannot be treated with "
        "antibiotics."
    ),
    HEALTHY_CLASS: (
        "No disease indicated. Keep monitoring behaviour, appetite and water quality, and "
        "re-check if symptoms appear."
    ),
}
DEFAULT_RECOMMENDATION = (
    "Observe the fish closely, check water quality, and consult an aquaculture professional."
)
LOW_CONFIDENCE_RECOMMENDATION = (
    "The model is not confident about this image. Upload a clearer, centred and well-lit photo "
    "of a single fish (side view, lesion visible) and analyse again."
)


def is_healthy(predicted_class: str | None) -> bool:
    return predicted_class == HEALTHY_CLASS


def get_risk_message(risk: str, predicted_class: str | None = None) -> str:
    """Message for a risk level; a Healthy Fish prediction gets its own wording."""
    if is_healthy(predicted_class):
        return HEALTHY_MESSAGES[risk]
    return RISK_MESSAGES[risk]


def get_recommendation(predicted_class: str | None, risk: str) -> str:
    """Farmer-facing next step for a prediction; LOW confidence asks for a better photo."""
    if risk == RISK_LOW:
        return LOW_CONFIDENCE_RECOMMENDATION
    return RECOMMENDATIONS.get(predicted_class or "", DEFAULT_RECOMMENDATION)
