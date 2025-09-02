# core/plot_utils.py
# -*- coding: utf-8 -*-
"""
Outils de visualisation RS3 (matplotlib pur) :
- plot_map_with_delivery : trajectoire lat/lon + segments in_delivery.
- plot_altitude : altitude_m (et slope_percent si dispo) vs temps.
- plot_event_bands : bandes verticales pour event_infra/behavior/context.
- plot_quicklook : figure 3 rangées combinant le tout, option d'export.

Conçu pour être tolérant aux colonnes manquantes.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------- Helpers ----------

def _has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Assure l'existence d'une colonne timestamp en datetime64[ns]."""
    if "timestamp" not in df.columns:
        return df
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    return df

def _iter_binary_spans(series: pd.Series) -> Iterable[Tuple[int, int]]:
    """
    Itère des plages contiguës où series == 1.
    Retourne des couples (start_idx, end_idx_inclusif).
    """
    arr = series.fillna(0).astype(int).to_numpy()
    n = len(arr)
    if n == 0:
        return
    i = 0
    while i < n:
        if arr[i] == 1:
            j = i
            while j + 1 < n and arr[j + 1] == 1:
                j += 1
            yield (i, j)
            i = j + 1
        else:
            i += 1


# ---------- Plots ----------

def plot_map_with_delivery(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Trace la trajectoire lat/lon. Si 'in_delivery' est présent, surligne
    les segments correspondants. Marqueur sur premier/dernier point.
    """
    assert _has_cols(df, ["lat", "lon"]), "Colonnes 'lat' et 'lon' requises."
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5))

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()

    # Trajectoire globale (ligne fine)
    ax.plot(lon, lat, linewidth=1.0, alpha=0.7)

    # Surlignage segments in_delivery
    if "in_delivery" in df.columns:
        in_del = df["in_delivery"].fillna(0).astype(int)
        for i0, i1 in _iter_binary_spans(in_del):
            ax.plot(lon[i0:i1 + 1], lat[i0:i1 + 1], linewidth=2.2, alpha=0.95)

    # Marqueurs début / fin
    if len(lat) > 0:
        ax.scatter([lon[0]], [lat[0]], s=30, marker="o", zorder=3)
        ax.scatter([lon[-1]], [lat[-1]], s=30, marker="X", zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trajectoire (segments livraison surlignés)" if "in_delivery" in df.columns
                 else "Trajectoire")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


def plot_altitude(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Altitude vs temps. Utilise 'altitude_m' si présent, sinon 'altitude'.
    Trace aussi 'slope_percent' en axe secondaire si disponible.
    """
    df = _ensure_ts(df)
    assert "timestamp" in df.columns, "Colonne 'timestamp' requise pour ce graphe."

    alt_col = "altitude_m" if "altitude_m" in df.columns else (
        "altitude" if "altitude" in df.columns else None
    )
    if alt_col is None:
        raise AssertionError("Aucune colonne d'altitude trouvée ('altitude_m' ou 'altitude').")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))

    # Altitude
    ax.plot(df["timestamp"], df[alt_col], linewidth=1.5)
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude / temps")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Axe secondaire : pente si dispo
    if "slope_percent" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["timestamp"], df["slope_percent"], linewidth=1.0, alpha=0.6)
        ax2.set_ylabel("Pente (%)")

    # Format des dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    return ax


def plot_event_bands(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    categories: Optional[Iterable[str]] = None,
    alpha: float = 0.12,
) -> plt.Axes:
    """
    Ajoute des bandes verticales sur l'axe temporel pour les colonnes
    binaires de catégories d'événements (event_<cat>).
    """
    df = _ensure_ts(df)
    assert "timestamp" in df.columns, "Colonne 'timestamp' requise."

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.6))

    if categories is None:
        # Par défaut : celles attendues v1.0 si présentes
        candidates = ["infra", "behavior", "context"]
        categories = [c for c in candidates if f"event_{c}" in df.columns]
        # À défaut, détection automatique
        if not categories:
            categories = sorted(
                {c.replace("event_", "") for c in df.columns if c.startswith("event_")}
            )

    # On trace une base (ligne à zéro) pour garantir un y-limits correct
    ax.plot(df["timestamp"], np.zeros(len(df)), linewidth=0.5, alpha=0.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Bandes d'événements par catégorie")
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)

    # Bandes
    for cat in categories:
        col = f"event_{cat}"
        if col not in df.columns:
            continue
        for i0, i1 in _iter_binary_spans(df[col]):
            t0 = df["timestamp"].iloc[i0]
            t1 = df["timestamp"].iloc[i1]
            ax.axvspan(t0, t1, alpha=alpha)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    return ax


def plot_quicklook(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Compose une figure 3 rangées :
      1) Carte lat/lon avec segments in_delivery,
      2) Altitude vs temps,
      3) Bandes d'événements par catégorie (event_*).

    Paramètres
    ----------
    df : DataFrame
    save_path : chemin de sauvegarde (PNG/PDF) si fourni
    show : affiche la figure si True

    Retour
    ------
    (fig, (ax1, ax2, ax3))
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9), constrained_layout=True
    )

    # 1) Carte
    try:
        plot_map_with_delivery(df, ax=ax1)
    except AssertionError as e:
        ax1.text(0.5, 0.5, f"plot_map_with_delivery: {e}", ha="center", va="center")
        ax1.axis("off")

    # 2) Altitude
    try:
        plot_altitude(df, ax=ax2)
    except AssertionError as e:
        ax2.text(0.5, 0.5, f"plot_altitude: {e}", ha="center", va="center")
        ax2.axis("off")

    # 3) Bandes d'événements
    try:
        plot_event_bands(df, ax=ax3)
    except AssertionError as e:
        ax3.text(0.5, 0.5, f"plot_event_bands: {e}", ha="center", va="center")
        ax3.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, (ax1, ax2, ax3)