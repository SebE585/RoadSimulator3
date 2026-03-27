"""
Shared bi-histogram Streamlit component.
Distance-weighted Gx/Gy severity heatmap — log scale, mG axes.
Matches the reference profile visualization.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

G = 9.80665


def _incremental_distance(speed_mps: np.ndarray, dt: float) -> np.ndarray:
    return np.abs(speed_mps) * dt


def render_bihistogram(
    df,
    speed_col: str = "speed",
    ax_col: str = "acc_x",
    ay_col: str = "acc_y",
    hz: float = 10.0,
    range_mg: float = 1000.0,
    bin_size_mg: float = 20.0,
    title: str = "Heatmap acc_x vs acc_y (distance-weighted)",
):
    """Render a bi-histogram heatmap matching the reference profile format.

    Axes in mG, log color scale, distance-weighted.
    """
    # Auto-detect columns
    speed = None
    for c in [speed_col, "speed_mps", "speed"]:
        if c in df.columns:
            speed = df[c].to_numpy(dtype=float)
            break
    ax_raw = None
    for c in [ax_col, "ax_mps2", "acc_x"]:
        if c in df.columns:
            ax_raw = df[c].to_numpy(dtype=float)
            break
    ay_raw = None
    for c in [ay_col, "ay_mps2", "acc_y"]:
        if c in df.columns:
            ay_raw = df[c].to_numpy(dtype=float)
            break

    if speed is None or ax_raw is None or ay_raw is None:
        st.warning("Bi-histogram: missing columns (speed, acc_x/ax_mps2, acc_y/ay_mps2)")
        return

    dt = 1.0 / hz

    # Convert to mG
    ax_mg = (ax_raw / G) * 1000.0
    ay_mg = (ay_raw / G) * 1000.0
    dist = _incremental_distance(speed, dt)

    # Filter: moving + finite + within range
    moving = speed > 0.5
    finite = np.isfinite(ax_mg) & np.isfinite(ay_mg) & np.isfinite(dist)
    in_range = (np.abs(ax_mg) <= range_mg * 1.5) & (np.abs(ay_mg) <= range_mg * 1.5)
    valid = moving & finite & in_range

    if valid.sum() < 10:
        st.info("Not enough data for bi-histogram.")
        return

    ax_v = ax_mg[valid]
    ay_v = ay_mg[valid]
    dist_v = dist[valid]

    # Build 2D histogram: X = acc_y (lateral), Y = acc_x (longitudinal)
    # Convention from engine/accel_stats.py: X=lateral, Y=longitudinal
    n_bins = int(2 * range_mg / bin_size_mg) + 1
    edges = np.linspace(-range_mg, range_mg, n_bins + 1)

    hist, xedges, yedges = np.histogram2d(
        ay_v, ax_v,
        bins=[edges, edges],
        weights=dist_v,
    )

    # Log scale (replace 0 with NaN for display)
    hist_log = hist.copy()
    hist_log[hist_log == 0] = np.nan
    hist_log = np.log10(hist_log)

    # Centers for axis labels
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    fig = go.Figure(data=go.Heatmap(
        z=hist_log.T,
        x=xcenters,
        y=ycenters,
        colorscale="Jet",
        colorbar=dict(title="Distance (m, log10)"),
        hoverongaps=False,
        zmin=0,
        zmax=np.nanmax(hist_log) if np.any(np.isfinite(hist_log)) else 4,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="acc_y (mG)",
        yaxis_title="acc_x (mG)",
        width=550, height=550,
        xaxis=dict(scaleanchor="y", constrain="domain", range=[-range_mg, range_mg]),
        yaxis=dict(range=[-range_mg, range_mg]),
        margin=dict(l=60, r=20, t=40, b=60),
    )
    # Crosshairs
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.3)

    st.plotly_chart(fig, use_container_width=False)

    # Severity metrics
    ax_g = ax_v / 1000.0
    ay_g = ay_v / 1000.0
    w = dist_v
    total_w = w.sum()
    if total_w > 0:
        mean_gx = np.sum(w * ax_g) / total_w
        mean_gy = np.sum(w * ay_g) / total_w
        std_gx = float(np.sqrt(np.sum(w * (ax_g - mean_gx)**2) / total_w))
        std_gy = float(np.sqrt(np.sum(w * (ay_g - mean_gy)**2) / total_w))
    else:
        std_gx = std_gy = 0.0

    mc = st.columns(3)
    mc[0].metric("Std Gx (spatial)", f"{std_gx:.4f} g")
    mc[1].metric("Std Gy (spatial)", f"{std_gy:.4f} g")
    mc[2].metric("Distance", f"{dist_v.sum() / 1000:.1f} km")
