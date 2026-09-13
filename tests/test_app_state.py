"""Dashboard bookkeeping (app/state.py) without a Streamlit runtime."""

from app.state import SessionHistory

CLASSES = ["A", "B", "Healthy Fish"]


def _result(cls, conf, risk, healthy=False):
    return {"predicted_class": cls, "confidence": conf, "risk": risk, "healthy": healthy}


def test_empty_history():
    h = SessionHistory()
    assert h.count == 0 and h.latest is None and h.average_confidence is None
    assert h.distribution(CLASSES) == {"A": 0, "B": 0, "Healthy Fish": 0}
    assert h.risk_counts() == {"HEALTHY": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    assert h.recent() == []


def test_history_statistics():
    h = SessionHistory()
    h.add("x.jpg", _result("A", 0.9, "HIGH"))
    h.add("y.jpg", _result("Healthy Fish", 0.7, "MODERATE", healthy=True))
    h.add("z.jpg", _result("A", 0.4, "LOW"))
    assert h.count == 3 and h.latest.filename == "z.jpg"
    assert abs(h.average_confidence - (0.9 + 0.7 + 0.4) / 3) < 1e-9
    assert h.distribution(CLASSES) == {"A": 2, "B": 0, "Healthy Fish": 1}
    assert h.risk_counts() == {"HEALTHY": 1, "HIGH": 1, "MODERATE": 0, "LOW": 1}
    assert [r.filename for r in h.recent(2)] == ["z.jpg", "y.jpg"]
