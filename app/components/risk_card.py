"""Risk panel — kept as a thin alias so existing imports keep working.

Owner: Student 4
"""

from __future__ import annotations

from app.components.result_card import render_risk_panel

render_risk_card = render_risk_panel

__all__ = ["render_risk_card", "render_risk_panel"]
