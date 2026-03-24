"""
Telemachus / RS3 — Shared Streamlit Theme (Light).
"""
from __future__ import annotations

import streamlit as st

# ── Palette (light) ──────────────────────────────────────────────────────────

COLORS = {
    "primary": "#FFFFFF",
    "secondary": "#F7F8FA",
    "accent": "#0066CC",
    "accent2": "#0088EE",
    "success": "#2DC653",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "text": "#1A1A2E",
    "text_muted": "#6B7280",
    "border": "#E5E7EB",
}

_CSS = """
<style>
    .stDeployButton { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }
    footer { visibility: hidden !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    div[data-testid="stMetricValue"] {
        color: %(accent)s;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: %(accent)s !important;
        border-bottom-color: %(accent)s !important;
    }

    .stButton > button[kind="primary"] {
        background-color: %(accent)s;
        color: white;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: %(accent2)s;
    }

    .stage-ok { color: %(success)s; font-weight: bold; }
    .stage-err { color: %(error)s; font-weight: bold; }
    .stage-time { color: %(text_muted)s; font-size: 0.85em; }
</style>
""" % COLORS


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = "📡") -> None:
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:8px'>"
        f"<span style='font-size:2.5em'>{icon}</span>"
        f"<div>"
        f"<h1 style='margin:0; color:{COLORS['text']}; font-size:1.8em'>{title}</h1>"
        f"<p style='margin:0; color:{COLORS['text_muted']}; font-size:0.95em'>{subtitle}</p>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def pipeline_status_row(label: str, ok: bool, msg: str, elapsed_ms: float) -> None:
    icon = "✅" if ok else "❌"
    css_class = "stage-ok" if ok else "stage-err"
    st.markdown(
        f"{icon} <span class='{css_class}'>{label}</span>"
        f" — {msg} <span class='stage-time'>{elapsed_ms:.0f}ms</span>",
        unsafe_allow_html=True,
    )


def footer(product: str = "Telemachus Data Platform") -> None:
    st.divider()
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['text_muted']}; font-size:0.8em; padding:8px 0'>"
        f"{product} — Signal → Intelligence"
        f"</div>",
        unsafe_allow_html=True,
    )
