"""Owner: Student 3

Recent-activity summary strip for the landing page.

Deliberately a separate module from components/dashboard.py, which is owned by
Student 4 and serves a different purpose (empty-state help + supported class
list). If the two are merged later, this is the side that should move.

The counts are mock data for the hackathon demo: swap get_summary() for a real
query once analyses are persisted.
"""

from __future__ import annotations

import streamlit as st

_MOCK_SUMMARY = {
    "total": 12,
    "high": 3,
    "moderate": 4,
    "healthy": 5,
}


def get_summary() -> dict[str, int]:
    """Return the recent-analysis counts. Replace with a real query later."""
    return dict(_MOCK_SUMMARY)


def render_summary_panel(summary: dict[str, int] | None = None) -> None:
    """Draw the four-tile summary strip."""
    data = summary or get_summary()
    total = int(data.get("total", 0))
    high = int(data.get("high", 0))
    moderate = int(data.get("moderate", 0))
    healthy = int(data.get("healthy", 0))

    def share(count: int) -> str:
        return f"{(count / total * 100):.0f}% of scans" if total else "-"

    st.markdown(
        f"""
        <div class="ah-stats">
          <div class="ah-stat">
            <div class="ah-stat-value">{total}</div>
            <div class="ah-stat-label">Total analyses</div>
            <div class="ah-stat-note">Last 7 days</div>
          </div>
          <div class="ah-stat is-bad">
            <div class="ah-stat-value">{high}</div>
            <div class="ah-stat-label">High risk</div>
            <div class="ah-stat-note">{share(high)}</div>
          </div>
          <div class="ah-stat is-warn">
            <div class="ah-stat-value">{moderate}</div>
            <div class="ah-stat-label">Moderate risk</div>
            <div class="ah-stat-note">{share(moderate)}</div>
          </div>
          <div class="ah-stat is-ok">
            <div class="ah-stat-value">{healthy}</div>
            <div class="ah-stat-label">Healthy</div>
            <div class="ah-stat-note">{share(healthy)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
