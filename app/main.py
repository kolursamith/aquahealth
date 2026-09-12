"""Streamlit entry point.

    upload → predict(image) → result card + risk card

The real predictor (`src.predict`) is used when a checkpoint exists at
`AQUAHEALTH_CHECKPOINT` (env) or `src.config.CHECKPOINT_PATH`; otherwise
the mock predictor answers with a fixed response and the page says so.

Owner: Student 3 + Student 4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import streamlit as st

# `streamlit run app/main.py` puts app/ (not the project root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.dashboard import render_dashboard  # noqa: E402
from app.components.result_card import render_result_card  # noqa: E402
from app.components.risk_card import render_risk_card  # noqa: E402
from app.components.upload import render_upload  # noqa: E402
from app.mock_prediction import predict as mock_predict  # noqa: E402
from src.config import CHECKPOINT_PATH, CLASS_NAMES  # noqa: E402
from src.predict import ModelLoadError, get_predictor  # noqa: E402

CHECKPOINT_ENV = "AQUAHEALTH_CHECKPOINT"


def checkpoint_location() -> Path:
    return Path(os.environ.get(CHECKPOINT_ENV, CHECKPOINT_PATH))


def select_predictor() -> tuple[Callable[[Any], dict[str, Any]], str, list[str]]:
    """Return (predict function, human-readable source, class names it knows)."""
    path = checkpoint_location()
    if not path.is_file():
        return mock_predict, f"mock predictor — no checkpoint at {path}", list(CLASS_NAMES)
    try:
        predictor = get_predictor(str(path))
    except ModelLoadError as exc:
        return mock_predict, f"mock predictor — checkpoint unusable: {exc}", list(CLASS_NAMES)

    def real_predict(image: Any) -> dict[str, Any]:
        return predictor.predict(image).to_dict()

    return real_predict, predictor.model_version, predictor.class_names


def main() -> None:
    st.set_page_config(page_title="AquaHealth AI", page_icon="🐟", layout="centered")
    st.title("AquaHealth AI")
    st.caption("AI-Based Smart Aquaculture Disease Detection")

    predict, source, class_names = select_predictor()
    st.caption(f"Model: {source}")

    image = render_upload()
    if image is None:
        render_dashboard(class_names, source)
        return

    result = predict(image)
    if result["status"] != "ok":
        st.error(f"Prediction failed: {result['error']}")
        return
    render_result_card(result)
    render_risk_card(result)


if __name__ == "__main__":
    main()
