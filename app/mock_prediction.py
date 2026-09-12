"""Fixed-response stand-in for `src.predict.predict`.

Lets the frontend be developed and demonstrated without a trained
checkpoint. It builds the very same `PredictionResult` type as the real
predictor, so the two cannot drift apart in schema.
"""

from typing import Any

from src.predict import (
    API_VERSION,
    STATUS_OK,
    ImageInput,
    PredictionResult,
    RankedPrediction,
    load_input,
)
from src.risk_engine import get_risk_level, get_risk_message

MOCK_MODEL_VERSION = "mock (no checkpoint loaded)"
MOCK_RANKING = (
    ("Aeromoniasis", 0.87),
    ("Healthy", 0.08),
    ("Fin_Rot", 0.05),
)


def predict(image: ImageInput) -> dict[str, Any]:
    warnings = ["mock predictor: fixed response, not a model output"]
    try:
        _, input_warnings = load_input(image)
        warnings += input_warnings
    except Exception as exc:  # noqa: BLE001 - the mock mirrors the real API's error contract
        return PredictionResult(
            status="error",
            api_version=API_VERSION,
            model_version=MOCK_MODEL_VERSION,
            preprocessing_version=None,
            device=None,
            warnings=warnings,
            error=str(exc),
        ).to_dict()

    ranked = [
        RankedPrediction(rank=i + 1, class_name=name, probability=p)
        for i, (name, p) in enumerate(MOCK_RANKING)
    ]
    risk = get_risk_level(ranked[0].probability)
    return PredictionResult(
        status=STATUS_OK,
        api_version=API_VERSION,
        model_version=MOCK_MODEL_VERSION,
        preprocessing_version=None,
        device=None,
        predicted_class=ranked[0].class_name,
        confidence=ranked[0].probability,
        risk=risk,
        message=get_risk_message(risk),
        ranked_predictions=ranked,
        warnings=warnings,
    ).to_dict()
