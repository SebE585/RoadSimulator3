"""
Shared bi-histogram Streamlit component.
Displays a distance-weighted Gx/Gy severity heatmap from acceleration data.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

G = 9.80665


def _incremental_distance(speed_mps: np.ndarray, dt: float) -> np.ndarray:
    return np.abs(speed_mps) * dt


def _filter_non_physical(gx: np.ndarray, gy: np.ndarray, max_g: float = 1.5) -> np.ndarray:
    return np.sqrt(gx**2 + gy**2) <= max_g


def _build_bihistogram(gx, gy, weights, range_g=0.5, class_size_g=0.02):
    n_bins = int(round(2 * range_g / class_size_g)) + 1
    edges = np.linspace(-range_g, range_g, n_bins + 1)
    mask = _filter_non_physical(gx, gy, max_g=range_g * 1.5)
    gx_f, gy_f, w_f = gx[mask], gy[mask], weights[mask]
    hist, xedges, yedges = np.histogram2d(gy_f, gx_f, bins=[edges, edges], weights=w_f)
    total = w_f.sum()
    if total > 0:
        hist = hist / total
    return hist, xedges, yedges


def _spatial_std(values, weights):
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return 0.0
    w, v = w[mask], v[mask]
    total_w = w.sum()
    if total_w <= 0:
        return 0.0
    mean = np.sum(w * v) / total_w
    var = np.sum(w * (v - mean) ** 2) / total_w
    return float(np.sqrt(var))


def render_bihistogram(
    df,
    speed_col: str = "speed",
    ax_col: str = "acc_x",
    ay_col: str = "acc_y",
    hz: float = 10.0,
    range_g: float = 0.5,
    title: str = "Driving Severity — Bi-histogram (Gx/Gy)",
):
    """Render a bi-histogram heatmap in Streamlit from a DataFrame.

    Supports both RS3 column names (speed, acc_x, acc_y) and
    Telemachus/Nostos names (speed_mps, ax_mps2, ay_mps2).
    """
    # Auto-detect column names
    for candidate in [speed_col, "speed_mps", "speed"]:
        if candidate in df.columns:
            speed = df[candidate].to_numpy(dtype=float)
            break
    else:
        st.warning("Bi-histogram: no speed column found.")
        return

    for candidate in [ax_col, "ax_mps2", "acc_x"]:
        if candidate in df.columns:
            ax_raw = df[candidate].to_numpy(dtype=float)
            break
    else:
        st.warning("Bi-histogram: no longitudinal acceleration column found.")
        return

    for candidate in [ay_col, "ay_mps2", "acc_y"]:
        if candidate in df.columns:
            ay_raw = df[candidate].to_numpy(dtype=float)
            break
    else:
        st.warning("Bi-histogram: no lateral acceleration column found.")
        return

    dt = 1.0 / hz
    gx = ax_raw / G
    gy = ay_raw / G
    dist = _incremental_distance(speed, dt)

    # Filter: moving + physical + finite
    moving = speed > 0.5
    phys = _filter_non_physical(gx, gy)
    valid = moving & phys & np.isfinite(gx) & np.isfinite(gy)

    if valid.sum() < 10:
        st.info("Not enough valid data points for bi-histogram.")
        return

    gx_v, gy_v, dist_v = gx[valid], gy[valid], dist[valid]

    # Build histogram
    hist, xedges, yedges = _build_bihistogram(gx_v, gy_v, dist_v, range_g=range_g)

    # Severity metrics
    std_gx = _spatial_std(gx_v, dist_v)
    std_gy = _spatial_std(gy_v, dist_v)

    # Plot
    fig = go.Figure(data=go.Heatmap(
        z=hist.T,
        x=np.round((xedges[:-1] + xedges[1:]) / 2, 3),
        y=np.round((yedges[:-1] + yedges[1:]) / 2, 3),
        colorscale="Hot_r",
        colorbar=dict(title="% distance"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Lateral Gy (g)",
        yaxis_title="Longitudinal Gx (g)",
        width=500, height=500,
        xaxis=dict(scaleanchor="y", constrain="domain"),
    )
    # Add crosshairs at origin
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=False)

    # Metrics
    mc = st.columns(3)
    mc[0].metric("Std Gx (spatial)", f"{std_gx:.4f} g")
    mc[1].metric("Std Gy (spatial)", f"{std_gy:.4f} g")
    mc[2].metric("Distance", f"{dist_v.sum() / 1000:.1f} km")
