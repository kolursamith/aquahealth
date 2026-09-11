"""AI backend <-> frontend integration interface.

predict(image) -> {
    "predicted_class": "Aeromoniasis",
    "confidence": 0.87,
    "risk": "HIGH",
    "message": "High-confidence disease indication.",
}

Owner: Student 1 + Student 2
"""

from typing import TypedDict

import numpy as np

from src.risk_engine import get_risk_level, get_risk_message


class Prediction(TypedDict):
    predicted_class: str
    confidence: float
    risk: str
    message: str


def predict(image: np.ndarray) -> Prediction:
    """Run the full inference pipeline on a single RGB image array.

    Raises NotImplementedError until the trained checkpoint and model-loading
    code are in place. Frontend development should use
    `app.mock_prediction.predict` in the meantime.
    """
    raise NotImplementedError(
        "Load checkpoint, preprocess(image), run model, softmax, "
        "then build a Prediction dict via risk_engine"
    )


def _build_prediction(predicted_class: str, confidence: float) -> Prediction:
    risk = get_risk_level(confidence)
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "risk": risk,
        "message": get_risk_message(risk),
    }
