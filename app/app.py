"""Streamlit entry point.

app.py -> predict(image) -> result

Uses the mock predictor until a trained checkpoint is available; switch the
import below to `from src.predict import predict` once src/train.py has
produced models/final_model.pth.

Owner: Student 3 + Student 4
"""

import streamlit as st

from app.components.dashboard import render_dashboard
from app.components.result_card import render_result_card
from app.components.risk_card import render_risk_card
from app.components.upload import render_upload
from app.mock_prediction import predict

st.set_page_config(page_title="AquaHealth AI", page_icon="🐟", layout="centered")


def main() -> None:
    st.title("AquaHealth AI")
    st.caption("AI-Based Smart Aquaculture Disease Detection")

    image = render_upload()
    if image is None:
        render_dashboard()
        return

    result = predict(image)
    render_result_card(result)
    render_risk_card(result)


if __name__ == "__main__":
    main()
