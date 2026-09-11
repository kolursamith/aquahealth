import pytest

pytest.importorskip("numpy", reason="numpy is introduced by Layer 5")

import numpy as np  # noqa: E402

from app.mock_prediction import predict as mock_predict  # noqa: E402

REQUIRED_KEYS = {"predicted_class", "confidence", "risk", "message"}


def test_mock_prediction_schema():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    result = mock_predict(image)
    assert REQUIRED_KEYS <= result.keys()
    assert isinstance(result["predicted_class"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk"] in {"LOW", "MODERATE", "HIGH"}


# TODO(student-1, student-2): once src/predict.py is implemented against a
# trained checkpoint, add an equivalent test for src.predict.predict here.
