# ruff: noqa: E501  (HTML/CSS string literals; black does not reflow strings)
"""Session intelligence dashboard: only what this browser session has analysed.

Owner: Student 4
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Callable

import streamlit as st

from app.state import SessionHistory
from app.theme import confidence_label


def _metric(label: str, value: str, accent: str = "") -> str:
    size = "1.3rem" if len(value) > 12 else "2.2rem"
    return (
        f'<div class="aq-frame tight center"><div class="aq-label">{label}</div>'
        f'<div style="font-size:{size};font-weight:800;line-height:1.15;margin-top:6px;overflow-wrap:anywhere;{accent}">{html.escape(value)}</div></div>'
    )


def render_dashboard(
    history: SessionHistory, class_names: list[str], on_start: Callable[[], None]
) -> None:
    st.markdown(
        '<div class="aq-note">Session-only data: this dashboard summarises the images analysed since this page was opened. '
        "It is not connected to ponds, sensors, cameras or farm records.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    if history.count == 0:
        st.markdown(
            '<div class="aq-frame center" style="padding:48px 24px;"><span class="aq-num">SESSION INTELLIGENCE</span>'
            '<div class="aq-h2" style="margin-top:8px;">No analysis data yet</div>'
            '<p class="aq-sub" style="margin:0 auto;">Analyse a fish image and every result will appear here.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        c = st.columns([1, 1, 1])[1]
        c.button(
            "Start first analysis", type="primary", use_container_width=True, on_click=on_start
        )
        return
    latest = history.latest
    healthy = sum(1 for r in history.records if r.healthy)
    latest_conf = confidence_label(latest.confidence) if latest is not None else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_metric("Total analyses", str(history.count)), unsafe_allow_html=True)
    c2.markdown(
        _metric("Disease detected", str(history.count - healthy), "color:var(--high);"),
        unsafe_allow_html=True,
    )
    c3.markdown(_metric("Healthy", str(healthy), "color:var(--healthy);"), unsafe_allow_html=True)
    c4.markdown(
        _metric("Latest confidence", latest_conf, "color:var(--cyan);"), unsafe_allow_html=True
    )
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<span class="aq-num">RECENT ANALYSES</span>', unsafe_allow_html=True)
        cards = "".join(
            f'<div class="aq-feature" style="margin-bottom:10px;"><div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;">'
            f"<div><b style='display:inline'>{html.escape(r.predicted_class)}</b><br><small>{html.escape(r.filename)} · {r.when}</small></div>"
            f'<div style="text-align:right"><span class="aq-badge {"HEALTHY" if r.healthy else r.risk}" style="font-size:.7rem;padding:5px 10px">{"HEALTHY" if r.healthy else r.risk}</span>'
            f"<div style='font-weight:800;margin-top:4px'>{confidence_label(r.confidence)}</div></div></div></div>"
            for r in history.recent(8)
        )
        st.markdown(cards, unsafe_allow_html=True)
    with right:
        st.markdown(
            '<span class="aq-num">CLASS FREQUENCY (THIS SESSION)</span>', unsafe_allow_html=True
        )
        counts = Counter(r.predicted_class for r in history.records)
        rows = "".join(
            f'<div class="aq-row"><span>{html.escape(name)}</span><div class="aq-bar"><span style="width:{(counts.get(name, 0) / history.count):.0%}"></span></div><span>{counts.get(name, 0)}</span></div>'
            for name in class_names
        )
        st.markdown(f'<div class="aq-frame tight">{rows}</div>', unsafe_allow_html=True)
        if history.count >= 2:
            st.markdown('<span class="aq-num">CONFIDENCE HISTORY</span>', unsafe_allow_html=True)
            st.line_chart(
                {"confidence": [r.confidence for r in history.records]}, color="#22d3ee", height=180
            )
