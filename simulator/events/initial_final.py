"""Injection des phases de démarrage et d'arrêt progressifs dans une trajectoire simulée.

Ce module fournit deux fonctions :
- `inject_initial_acceleration(df, v_max_kmh=None, duration=None)` : injecte un profil de montée en vitesse inertielle jusqu’à une cible (ex : 43 km/h) sur une durée donnée.
- `inject_final_deceleration(df, v_start_kmh=None, duration=None)` : injecte une décélération douce à la fin du trajet jusqu’à l’arrêt.

Les profils inertiels incluent :
- Une évolution progressive de la vitesse (linéaire),
- Une variation réaliste d’accélérations (`acc_x`, `acc_y`, `acc_z`) avec bruit inertiel,
- Un marquage des événements dans la colonne `event`.

Tous les paramètres (durée, amplitude, vitesses) sont extraits depuis `config/events.yaml`,
sauf si précisés manuellement.
"""

import numpy as np
import pandas as pd
import logging
from simulator.events.utils import ensure_event_column_object

logger = logging.getLogger(__name__)

G = 9.81
HZ = 10
DT = 1.0 / HZ

# Injection d'une montée en vitesse réaliste (accélération initiale)
def inject_initial_acceleration(df, v_max_kmh=None, duration=None):
    from simulator.events.config import get_event_config
    cfg = get_event_config("initial", default={
        "v_max_kmh": 43.0,
        "duration_s": 5.0,
        "acc_x": 1.5
    })

    v_max_kmh = cfg["v_max_kmh"] if v_max_kmh is None else v_max_kmh
    duration = cfg.get("duration_s", 7.0) if duration is None else duration
    acc_x = cfg.get("acc_x", 1.5)

    df = ensure_event_column_object(df)
    n = int(duration * HZ)

    for i in range(min(n, len(df))):
        ratio = i / n
        speed = v_max_kmh / 3.6 * (1 / (1 + np.exp(-2 * (ratio - 0.5))))  # sigmoïde plus douce
        df.at[i, "speed"] = speed * 3.6
        acc = np.gradient([0] + list(np.linspace(0, speed, i + 1)), DT)[-1]
        df.at[i, "acc_x"] = min(acc, v_max_kmh / 3.6 / duration) + np.random.normal(0, 0.05)
        df.at[i, "acc_y"] = np.random.normal(0, 0.05)
        df.at[i, "acc_z"] = G + np.random.normal(0, 0.1)
        df.at[i, "event"] = "acceleration_initiale"

    df.at[df.index[0], "speed"] = 0.0
    logger.info("[INITIAL] Accélération initiale injectée.")
    return df

# Injection d'une descente en vitesse réaliste (décélération finale)
def inject_final_deceleration(df, v_start_kmh=None, duration=None):
    from simulator.events.config import get_event_config
    cfg = get_event_config("final", default={
        "duration_s": 5.0,
        "v_final_kmh": 0.0
    })
    duration = cfg.get("duration_s", 7.0) if duration is None else duration

    df = ensure_event_column_object(df)
    n = int(duration * HZ)
    start_idx = max(len(df) - n, 0)
    if v_start_kmh is not None:
        v0 = v_start_kmh / 3.6
    else:
        v0 = df.at[start_idx, "speed"] / 3.6

    v_final_kmh = cfg.get("v_final_kmh", 0.0)

    for i in range(n):
        idx = start_idx + i
        if idx >= len(df): break
        ratio = i / n
        speed = v0 * (1 / (1 + np.exp(2 * (ratio - 0.5))))  # sigmoïde plus douce
        df.at[idx, "speed"] = speed * 3.6
        acc = -np.abs(np.gradient([v0] + list(np.linspace(v0, speed, i + 1)), DT)[-1])
        df.at[idx, "acc_x"] = max(acc, -v0 / duration) + np.random.normal(0, 0.05)
        df.at[idx, "acc_y"] = np.random.normal(0, 0.05)
        df.at[idx, "acc_z"] = G + np.random.normal(0, 0.1)
        df.at[idx, "event"] = "deceleration_finale"

    df.at[df.index[-1], "speed"] = 0.0
    logger.info("[FINAL] Décélération finale injectée.")
    return df
