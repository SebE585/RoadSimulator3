"""Module de génération des événements prolongés : `stop` et `wait`.

Ce module fournit les fonctions permettant :
- l’injection d’arrêts moteurs complets (`stop`) et temporaires (`wait`),
- l’expansion temporelle de ces événements selon les durées spécifiées,
- l’application de profils inertiels de décélération/accélération réalistes autour de ces arrêts,
- l’ajout de bruit inertiel aléatoire sur les phases de `wait`.

Les paramètres (durée, fréquence, nombre maximum, bruit, etc.) sont extraits depuis `config/events.yaml`.

Fonctions principales :
- `generate_stops(df)`
- `generate_waits(df)`
- `expand_stop_and_wait(df)`
- `apply_stop_or_wait_profile(df)`
- `apply_inertial_noise_on_wait(df)`
- `inject_stops_and_waits(df)` (pipeline complet)
"""

import numpy as np
import pandas as pd
import logging
from simulator.events.utils import ensure_event_column_object

from simulator.events.config import CONFIG

logger = logging.getLogger(__name__)

HZ = 10

def generate_stops(df):
    cfg = CONFIG["stop"]
    max_events = cfg["max_events"]
    min_duration = cfg["min_duration_s"]
    hz = cfg.get("hz", 10)
    max_attempts = cfg.get("max_attempts_per_event", 10)

    df = ensure_event_column_object(df)
    count = 0
    total_attempts = 0
    duration_points = int(min_duration * hz)

    while count < max_events and total_attempts < max_events * max_attempts:
        total_attempts += 1
        idx = np.random.randint(0, len(df) - duration_points)
        if df['event'].iloc[idx:idx + duration_points].isna().all():
            df.loc[idx:idx + duration_points - 1, ['speed', 'acc_x', 'acc_y', 'acc_z']] = 0
            df.loc[idx:idx + duration_points - 1, 'event'] = 'stop'
            logger.debug(f"[STOP] Injecté de l'index {idx} à {idx + duration_points - 1}")
            count += 1

    if count == 0:
        logger.warning("[STOP] Aucun stop injecté malgré les tentatives.")
    return df


def inject_stops_and_waits(df, max_events_per_type=5, hz=10, min_stop_duration=120, min_wait_duration=30):
    logger.info("[INJECTION] Génération initiale des stops...")
    df = generate_stops(df, max_events=max_events_per_type, min_duration=min_stop_duration)

    logger.info("[INJECTION] Génération initiale des waits...")
    df = generate_waits(df, max_events=max_events_per_type, min_duration=min_wait_duration)

    logger.info("[INJECTION] Expansion des stops et waits à leur durée réelle avec profils inertiels...")
    df = expand_stop_and_wait(df, hz=hz)

    logger.info("[INJECTION] Application des profils inertiels autour des stops/waits...")
    df = apply_stop_or_wait_profile(df, hz=hz)

    logger.info("[INJECTION] Application du bruit inertiel spécifique sur les waits...")
    df = apply_inertial_noise_on_wait(df, hz=hz)

    return df


def generate_waits(df):
    cfg = CONFIG["wait"]
    max_events = cfg["max_events"]
    min_duration = cfg["min_duration_s"]
    hz = cfg.get("hz", 10)
    max_attempts = cfg.get("max_attempts_per_event", 10)

    df = ensure_event_column_object(df)
    count = 0
    total_attempts = 0
    duration_points = int(min_duration * hz)

    while count < max_events and total_attempts < max_events * max_attempts:
        total_attempts += 1
        idx = np.random.randint(0, len(df) - duration_points)
        if df['event'].iloc[idx:idx + duration_points].isna().all():
            df.loc[idx:idx + duration_points - 1, ['speed', 'acc_x', 'acc_y', 'acc_z']] = 0
            df.loc[idx:idx + duration_points - 1, 'event'] = 'wait'
            logger.debug(f"[WAIT] Injecté de l'index {idx} à {idx + duration_points - 1}")
            count += 1

    if count == 0:
        logger.warning("[WAIT] Aucun wait injecté malgré les tentatives.")
    return df


def apply_stop_or_wait_profile(df, v_target_kmh=40, decel_amplitude=-3.0, accel_amplitude=3.0, hz=10):
    df = ensure_event_column_object(df)
    stop_indices = df.index[df['event'].isin(['stop', 'wait'])].tolist()

    if not stop_indices:
        logger.info("[INFO] Aucun événement stop ou wait détecté, aucun profil inertiel appliqué.")
        return df

    for idx_stop in stop_indices:
        decel_duration_s = 3
        decel_window = int(decel_duration_s * hz)
        decel_start = max(0, idx_stop - decel_window)

        v0 = df.at[decel_start, 'speed'] / 3.6
        for i in range(decel_window):
            idx = decel_start + i
            if idx >= idx_stop:
                break
            v = max(v0 + decel_amplitude * (i / hz), 0)
            df.at[idx, 'speed'] = v * 3.6
            df.at[idx, 'acc_x'] = decel_amplitude

        df.at[idx_stop, 'speed'] = 0
        df.at[idx_stop, 'acc_x'] = 0

        accel_duration_s = 3
        accel_window = int(accel_duration_s * hz)
        accel_start = idx_stop + 1
        v0 = 0

        for i in range(accel_window):
            idx = accel_start + i
            if idx >= len(df):
                break
            v = v0 + accel_amplitude * (i / hz)
            if v * 3.6 >= v_target_kmh:
                break
            df.at[idx, 'speed'] = v * 3.6
            df.at[idx, 'acc_x'] = accel_amplitude

    return df


def apply_inertial_noise_on_wait(df, hz=10, noise_std=0.05, verbose=False):
    mask_wait = df['event'] == 'wait'
    n_points = mask_wait.sum()

    if n_points == 0:
        if verbose:
            logger.info("[INFO] Aucun événement 'wait' trouvé pour bruit inertiel.")
        return df

    if verbose:
        logger.info(f"[INFO] Application du bruit inertiel sur {n_points} points 'wait'...")

    df.loc[mask_wait, 'acc_x'] += np.random.normal(0, noise_std, n_points)
    df.loc[mask_wait, 'acc_y'] += np.random.normal(0, noise_std, n_points)
    df.loc[mask_wait, 'acc_z'] += np.random.normal(0, noise_std, n_points)

    return df


def expand_stop_and_wait(df, hz=10, stop_duration_s=120, wait_duration_s=30, verbose=False):
    df = df.copy()
    stop_duration_pts = int(stop_duration_s * hz)
    wait_duration_pts = int(wait_duration_s * hz)

    total_stops = df['event'].value_counts().get('stop', 0)
    total_waits = df['event'].value_counts().get('wait', 0)

    if verbose:
        logger.info(f"[INFO] Expansion des {total_stops} stops et {total_waits} waits.")

    indices_to_expand_stop = []
    indices_to_expand_wait = []

    stop_indices = df.index[df['event'] == 'stop'].tolist()
    wait_indices = df.index[df['event'] == 'wait'].tolist()

    for idx in stop_indices:
        indices_to_expand_stop.extend(range(idx, min(idx + stop_duration_pts, len(df))))

    for idx in wait_indices:
        indices_to_expand_wait.extend(range(idx, min(idx + wait_duration_pts, len(df))))

    if indices_to_expand_stop:
        df.loc[indices_to_expand_stop, 'speed'] = 0
        df.loc[indices_to_expand_stop, 'acc_x'] = 0
        df.loc[indices_to_expand_stop, 'acc_y'] = 0
        df.loc[indices_to_expand_stop, 'acc_z'] = 0
        df.loc[indices_to_expand_stop, 'event'] = 'stop'

    if indices_to_expand_wait:
        df.loc[indices_to_expand_wait, 'speed'] = 0
        df.loc[indices_to_expand_wait, 'acc_x'] = 0
        df.loc[indices_to_expand_wait, 'acc_y'] = 0
        df.loc[indices_to_expand_wait, 'acc_z'] = 0
        df.loc[indices_to_expand_wait, 'event'] = 'wait'

    if verbose:
        logger.info(f"[INFO] Expansion complète terminée : {len(indices_to_expand_stop)} points stop, {len(indices_to_expand_wait)} points wait.")

    return df
