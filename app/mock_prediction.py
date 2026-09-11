"""Fixed-response stand-in for src.predict.predict, so Students 3 and 4 are
not blocked on a trained model during development.

Returns the same shape as ``src.predict.Prediction`` plus an optional
``probabilities`` map that the frontend renders when it is present. Class
names and risk levels are never redefined here -- they come from
``src.config`` and ``src.risk_engine``, which stay the single source of truth.

Swapping in the real backend is a one-line change in ``app/app.py``:
``from src.predict import predict``.
"""

from __future__ import annotations

import random
import time

import numpy as np

from src.config import CLASS_NAMES
from src.predict import Prediction
from src.risk_engine import get_risk_level, get_risk_message


class MockPrediction(Prediction, total=False):
    """A Prediction plus the optional extra keys the mock can supply.

    Declared as a TypedDict subclass so the extra key stays type-checked and
    ``src.predict.Prediction`` does not have to change.
    """

    probabilities: dict[str, float]


# TODO(student-1): this really belongs in src/config.py next to CLASS_NAMES.
# It lives here for now so the frontend does not edit backend-owned files.
HEALTHY_CLASS = "Healthy"

if HEALTHY_CLASS not in CLASS_NAMES:  # pragma: no cover - guards a config rename
    raise RuntimeError(f"{HEALTHY_CLASS!r} is missing from src.config.CLASS_NAMES")

DISEASE_CLASSES = [name for name in CLASS_NAMES if name != HEALTHY_CLASS]

# Simulated inference latency, so the UI's loading state is visible in demos.
SIMULATED_LATENCY_SECONDS = 2.0

# Confidence bands chosen to land on the intended src.risk_engine level.
_CONFIDENCE_BANDS = {
    "high": (0.82, 0.97),
    "moderate": (0.52, 0.78),
    "low_confidence": (0.28, 0.48),
    "healthy": (0.86, 0.96),
}

# Scenarios the mock picks from at random. "error" is deliberately excluded:
# tests/test_prediction.py calls predict() directly, so a random raise there
# would make CI flaky. Ask for it explicitly to exercise the error UI.
_RANDOM_SCENARIOS = ["high", "moderate", "healthy", "low_confidence"]
_SCENARIO_WEIGHTS = [0.30, 0.25, 0.30, 0.15]


class PredictionError(RuntimeError):
    """Raised when the backend cannot produce a prediction."""


def _probabilities(top_class: str, confidence: float) -> dict[str, float]:
    """Spread the remaining probability mass over the other classes."""
    others = [name for name in CLASS_NAMES if name != top_class]
    remaining = max(0.0, 1.0 - confidence)

    weights = [random.random() + 0.05 for _ in others]
    total = sum(weights)
    scores = {top_class: confidence}
    for name, weight in zip(others, weights):
        scores[name] = round(remaining * weight / total, 4)

    # Emit in CLASS_NAMES order so the UI always sees a stable ordering.
    return {name: float(scores[name]) for name in CLASS_NAMES}


def _build(top_class: str, confidence: float) -> MockPrediction:
    confidence = round(confidence, 4)
    risk = get_risk_level(confidence)
    return {
        "predicted_class": top_class,
        "confidence": confidence,
        "risk": risk,
        "message": get_risk_message(risk),
        "probabilities": _probabilities(top_class, confidence),
    }


def predict(image: np.ndarray, scenario: str | None = None) -> MockPrediction:
    """Pretend to run inference on ``image`` and return a Prediction-shaped dict.

    Parameters
    ----------
    image:
        RGB image array. Ignored by the mock; the parameter exists so the
        signature matches ``src.predict.predict``.
    scenario:
        Force an outcome: "high", "moderate", "low_confidence", "healthy" or
        "error". Defaults to a weighted random pick so every UI state can be
        demonstrated.
    """
    time.sleep(SIMULATED_LATENCY_SECONDS)

    if scenario is None:
        scenario = random.choices(_RANDOM_SCENARIOS, weights=_SCENARIO_WEIGHTS)[0]

    if scenario == "error":
        raise PredictionError("Mock backend failure (simulated).")

    if scenario not in _CONFIDENCE_BANDS:
        raise PredictionError(f"Unknown scenario: {scenario}")

    low, high = _CONFIDENCE_BANDS[scenario]
    confidence = random.uniform(low, high)
    top_class = HEALTHY_CLASS if scenario == "healthy" else random.choice(DISEASE_CLASSES)
    return _build(top_class, confidence)
