# core2/accel_stats.py
"""
Utilitaires d'analyse de sévérité de conduite (driving severity).

Implémente :
  - Bi-histogramme pondéré par la distance (V·dt)
  - Std spatiale (speed-weighted) de Gx et Gy
  - Percentile 0.1% des accélérations extrêmes
  - Filtrage des valeurs non-physiques
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.80665  # m/s²


def _incremental_distance(speed_mps: np.ndarray, dt: float) -> np.ndarray:
    """Distance parcourue à chaque pas de temps: V(t)·dt."""
    return np.abs(speed_mps) * dt


def filter_non_physical(
    gx: np.ndarray,
    gy: np.ndarray,
    max_acc_g: float = 1.5,
) -> np.ndarray:
    """
    Retourne un masque booléen des points physiquement valides.
    Critère : sqrt(Gx² + Gy²) <= max_acc_g.
    """
    mag = np.sqrt(gx**2 + gy**2)
    return mag <= max_acc_g


def spatial_std(
    values_g: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Écart-type spatial (pondéré par la distance).

    Formule :
      mean = Σ(V·dt · Gx) / Σ(V·dt)
      std  = sqrt( Σ(V·dt · (Gx - mean)²) / Σ(V·dt) )
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values_g, dtype=float)
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


def spatial_mean(
    values_g: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Moyenne spatiale (pondérée par la distance)."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values_g, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() < 1:
        return 0.0
    w, v = w[mask], v[mask]
    total_w = w.sum()
    if total_w <= 0:
        return 0.0
    return float(np.sum(w * v) / total_w)


def percentile_01(
    values_g: np.ndarray,
    weights: np.ndarray,
    side: str = "high",
) -> float:
    """
    Percentile 0.1% pondéré par la distance.

    side="high" : 0.1% des plus hautes accélérations
    side="low"  : 0.1% des plus basses (freinage)
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values_g, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() < 10:
        return 0.0
    w, v = w[mask], v[mask]

    if side == "high":
        order = np.argsort(-v)  # décroissant
    else:
        order = np.argsort(v)   # croissant

    w_sorted = w[order]
    v_sorted = v[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    idx = np.searchsorted(cum, 0.001)
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


def build_bihistogram(
    gx: np.ndarray,
    gy: np.ndarray,
    weights: np.ndarray,
    range_g: float = 1.0,
    class_size_g: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit le bi-histogramme pondéré par la distance.

    Retourne (histogram_2d, xedges, yedges) normalisé par la distance totale.
    Axes : xedges = Gy (latéral), yedges = Gx (longitudinal).
    """
    n_bins = int(round(2 * range_g / class_size_g)) + 1
    edges = np.linspace(-range_g, range_g, n_bins + 1)

    mask = filter_non_physical(gx, gy, max_acc_g=range_g * 1.5)
    gx_f, gy_f, w_f = gx[mask], gy[mask], weights[mask]

    hist, xedges, yedges = np.histogram2d(
        gy_f, gx_f,  # X=latéral, Y=longitudinal
        bins=[edges, edges],
        weights=w_f,
    )
    total_dist = w_f.sum()
    if total_dist > 0:
        hist = hist / total_dist

    return hist, xedges, yedges


def compute_severity_criteria(
    df: pd.DataFrame,
    hz: float = 10.0,
    max_acc_g: float = 1.5,
) -> dict:
    """
    Calcule les critères de sévérité de conduite depuis un DataFrame RS3.

    Attend les colonnes: speed (m/s), acc_x (m/s²), acc_y (m/s²).
    Retourne un dict avec les métriques clés.
    """
    dt = 1.0 / hz
    speed = df["speed"].to_numpy(dtype=float)
    ax = df["acc_x"].to_numpy(dtype=float) / G
    ay = df["acc_y"].to_numpy(dtype=float) / G

    # Distance incrémentale
    dist = _incremental_distance(speed, dt)

    # Filtre mouvement + physique
    moving = speed > 0.5
    phys = filter_non_physical(ax, ay, max_acc_g=max_acc_g)
    valid = moving & phys & np.isfinite(ax) & np.isfinite(ay)

    ax_v, ay_v, dist_v = ax[valid], ay[valid], dist[valid]

    total_distance_m = dist_v.sum()

    # Std spatiales
    std_gx = spatial_std(ax_v, dist_v)
    std_gy = spatial_std(ay_v, dist_v)

    # Moyennes spatiales
    mean_gx = spatial_mean(ax_v, dist_v)
    mean_gy = spatial_mean(ay_v, dist_v)

    # Percentiles 0.1%
    p01_gx_high = percentile_01(ax_v, dist_v, side="high")
    p01_gx_low = percentile_01(ax_v, dist_v, side="low")
    p01_gy_high = percentile_01(ay_v, dist_v, side="high")
    p01_gy_low = percentile_01(ay_v, dist_v, side="low")

    # Std temporelles (pour comparaison)
    std_gx_temporal = float(np.std(ax_v)) if len(ax_v) > 1 else 0.0
    std_gy_temporal = float(np.std(ay_v)) if len(ay_v) > 1 else 0.0

    return {
        "total_distance_m": float(total_distance_m),
        "n_valid_points": int(valid.sum()),
        "n_filtered_non_physical": int((~phys & moving).sum()),
        # Critères spatiaux (pondérés distance)
        "std_gx_spatial": std_gx,
        "std_gy_spatial": std_gy,
        "mean_gx_spatial": mean_gx,
        "mean_gy_spatial": mean_gy,
        # Percentiles 0.1%
        "p01_gx_high": p01_gx_high,
        "p01_gx_low": p01_gx_low,
        "p01_gy_high": p01_gy_high,
        "p01_gy_low": p01_gy_low,
        # Temporels (pour comparaison)
        "std_gx_temporal": std_gx_temporal,
        "std_gy_temporal": std_gy_temporal,
    }
