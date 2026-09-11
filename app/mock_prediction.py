"""Fixed-response stand-in for src.predict.predict, so Students 3 and 4 are
not blocked on a trained model during development.
"""

import numpy as np

from src.predict import Prediction


def predict(image: np.ndarray) -> Prediction:
    return {
        "predicted_class": "Aeromoniasis",
        "confidence": 0.87,
        "risk": "HIGH",
        "message": "High-confidence disease indication.",
    }
