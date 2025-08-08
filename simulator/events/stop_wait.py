"""Stop/Wait helpers for RoadSimulator3.
This module provides utilities to (1) add a progressive acceleration ramp
right after stop/wait events, and (2) (deprecated) inject stop/wait labels
near provided lat/lon positions.

Goals of the refactor:
- Clean import order and avoid circular imports.
- Add type hints and docstrings.
- Remove pandas warnings (e.g., SettingWithCopyWarning) by using .loc/.iloc
  on a dedicated copy.
- Keep behavior stable, with a minor fix on the unit of acc_x (m/s²).
"""

from __future__ import annotations

# Standard library
import logging
from typing import Iterable, List, Optional

# Third-party
import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Local
from core.decorators import deprecated

logger = logging.getLogger(__name__)

__all__ = [
    "apply_progressive_acceleration_after_stop_wait",
    "apply_stop_wait_at_positions",
]


def apply_progressive_acceleration_after_stop_wait(
    df: pd.DataFrame,
    hz: int = 10,
    target_speed_kmh: float = 30.0,
    duration_s: float = 5.0,
) -> pd.DataFrame:
    """
    Ajoute une accélération progressive après chaque séquence 'stop' ou 'wait'
    pour atteindre `target_speed_kmh` en `duration_s` secondes.

    Paramètres
    -----------
    df : DataFrame
        Doit contenir au minimum les colonnes: 'event', 'speed', 'acc_x'.
    hz : int
        Fréquence d'échantillonnage (Hz). Défaut : 10.
    target_speed_kmh : float
        Vitesse cible à atteindre après le stop/wait (km/h). Défaut : 30.
    duration_s : float
        Durée de la rampe (secondes). Défaut : 5.

    Retour
    ------
    DataFrame : une **copie** de `df` avec la vitesse (km/h) ajustée sur la rampe
    et une accélération longitudinale `acc_x` cohérente en m/s².
    """
    required_cols = {"event", "speed", "acc_x"}
    missing = required_cols.difference(df.columns)
    if missing:
        logger.warning("[stop_wait] Colonnes manquantes: %s", sorted(missing))
        return df

    df_out = df.copy()

    # Bornes et conversions
    n_points = max(1, int(duration_s * hz))
    target_speed_m_s = float(target_speed_kmh) / 3.6

    # Profil vitesse (m/s) de 0 -> vitesse cible, ensuite converti en km/h
    accel_profile_m_s = np.linspace(0.0, target_speed_m_s, n_points, dtype=float)
    accel_const_m_s2 = target_speed_m_s / max(duration_s, 1e-9)  # m/s² (pas de *3.6 ici)

    # Début de séquence: True si event ∈ {stop, wait} et différent de l'event précédent
    ev = df_out["event"].astype(object)
    is_stop_wait = ev.isin(["stop", "wait"])
    is_sequence_start = is_stop_wait & ev.ne(ev.shift(1))
    start_indices: List[int] = df_out.index[is_sequence_start].tolist()

    if not start_indices:
        logger.info("[stop_wait] Aucun début de séquence 'stop' ou 'wait' détecté.")
        return df_out

    for idx in start_indices:
        start_idx = int(idx) + 1  # on commence juste après l'échantillon labellisé
        end_idx = min(start_idx + n_points, len(df_out))
        if end_idx <= start_idx:
            continue

        # Fenêtre d'application
        win_len = end_idx - start_idx
        sl = slice(start_idx, end_idx)

        # Utiliser un indexing positionnel robuste pour éviter les erreurs de longueur
        speed_profile_kmh = accel_profile_m_s[:win_len] * 3.6
        rows = np.arange(start_idx, end_idx)
        col_speed = df_out.columns.get_loc("speed")
        col_accx = df_out.columns.get_loc("acc_x")
        df_out.iloc[rows, col_speed] = speed_profile_kmh
        # Accélération longitudinale (m/s²) constante durant la rampe
        df_out.iloc[rows, col_accx] = accel_const_m_s2

    return df_out


@deprecated
def apply_stop_wait_at_positions(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_m: float = 20.0,
) -> pd.DataFrame:
    """
    [DÉPRÉCIÉ] Injecte des étiquettes stop/wait proches de positions fournies.

    Notes
    -----
    - Cette fonction est maintenue pour compatibilité mais son usage est déconseillé.
    - L'import de `generate_opening_door` est fait localement pour éviter les
      import cycles et les warnings si la fonction n'est pas utilisée.
    """
    df_out = df.copy()
    # S'assurer que la colonne 'event' est bien en dtype=object pour éviter les FutureWarnings
    if "event" in df_out.columns and df_out["event"].dtype != "object":
        df_out["event"] = df_out["event"].astype("object")

    matched_indices: List[int] = []
    # ~80 s à 10 Hz
    min_spacing_pts = 800

    # Import tardif pour éviter les boucles d'import
    try:
        from simulator.events.generation import generate_opening_door  # type: ignore
    except Exception as e:  # pragma: no cover - garde-fou
        logger.debug("[stop_wait] Import tardif de generate_opening_door impossible: %s", e)
        generate_opening_door = None  # type: ignore

    for _, row in events_df.iterrows():
        lat0 = float(row["lat"])  # type: ignore[index]
        lon0 = float(row["lon"])  # type: ignore[index]
        label = str(row["event"])  # type: ignore[index]

        # Distances en mètres jusqu'à chaque point de df
        distances = df_out.apply(
            lambda r: geodesic((lat0, lon0), (r["lat"], r["lon"])).meters, axis=1
        )
        idx = int(distances.idxmin())
        if distances.iloc[idx] <= float(window_m):
            if matched_indices and abs(idx - matched_indices[-1]) < min_spacing_pts:
                continue  # trop proche du précédent
            if idx in matched_indices:
                continue

            matched_indices.append(idx)
            duration_pts = 1200 if label == "stop" else 300
            end_idx = min(idx + duration_pts, len(df_out) - 1)
            sl = slice(idx, end_idx)

            # Étiquettes et signatures très simples (placeholders)
            df_out.loc[sl, "event"] = label
            df_out.loc[sl, "speed"] = 0.0
            df_out.loc[sl, "acc_x"] = 0.0 if label == "stop" else -0.5
            df_out.loc[sl, "acc_z"] = 9.81
            df_out.loc[sl, "gyro_x"] = 0.0
            df_out.loc[sl, "gyro_y"] = 0.0
            df_out.loc[sl, "gyro_z"] = 0.0

            # Injection auto d'une ouverture de porte autour du stop/wait
            if generate_opening_door and label in {"stop", "wait"}:
                config_local = {
                    "events": {
                        "ouverture_porte": {
                            "enabled": True,
                            "around_event": label,
                            "window_before_s": 1.0,
                            "window_after_s": 2.0,
                            "intensity": 1.0,
                            "probability": 1.0,
                            "duration_pts": 30,
                            "max_events": 1,
                            "max_attempts": 3,
                        }
                    }
                }
                df_out = generate_opening_door(df_out, config=config_local)

    logger.info(
        "[stop_wait] %d événements uniques 'stop/wait' injectés à partir de positions fournies.",
        len(matched_indices),
    )
    return df_out