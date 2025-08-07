# --- Nouvelle version stable et corrigée du fichier generation.py ---
import numpy as np
import pandas as pd
import logging
from core.utils import ensure_event_column_object
from simulator.events.config import get_event_config

logger = logging.getLogger(__name__)

def has_recent_event(df, idx, spacing):
    start = max(0, idx - spacing)
    return df.loc[start:idx, 'event'].notna().any()

def mark_delivery_points(df):
    """
    Marque les points de livraison comme 'stop' ou 'wait' dans la colonne 'event'.
    """
    df = ensure_event_column_object(df)
    if "delivery" in df.columns:
        df.loc[df["delivery"] == "stop", "event"] = "stop"
        df.loc[df["delivery"] == "wait", "event"] = "wait"
    elif "type_livraison" in df.columns:
        df.loc[df["type_livraison"] == "stop", "event"] = "stop"
        df.loc[df["type_livraison"] == "wait", "event"] = "wait"
    return df

def generate_acceleration(df, config):
    logger.info("🔄 Début injection : acceleration")
    df = ensure_event_column_object(df)
    cfg = get_event_config("acceleration", default={"dummy": True})
    hz = cfg.get("hz", 10)
    min_duration_pts = 1
    max_duration_pts = 3
    min_separation_s = 3.0
    min_separation_pts = int(min_separation_s * hz)
    count, total_attempts = 0, 0
    injected_indices = []
    while count < cfg["max_events"] and total_attempts < cfg["max_events"] * cfg["max_attempts"]:
        total_attempts += 1
        duration_pts = np.random.choice([min_duration_pts, max_duration_pts])
        start_idx = np.random.randint(0, len(df) - duration_pts)
        if injected_indices and min([abs(start_idx - idx) for idx in injected_indices]) < min_separation_pts:
            continue
        global_spacing_pts = cfg.get("global_spacing_pts", 5000)
        if has_recent_event(df, start_idx, global_spacing_pts):
            continue
        # Vérification de duplication géographique
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "acceleration") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        indices = range(start_idx, start_idx + duration_pts)
        if all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            for j in indices:
                df.at[j, "acc_x"] = cfg["acc_x_mean"] + cfg["acc_x_std"] * np.random.randn()
                for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                    mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                    std = cfg.get(f"gyro_{axis}_std", 0.5)
                    df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
            df.at[start_idx, "event"] = "acceleration"
            injected_indices.append(start_idx)
            logger.debug(f"Accélération injectée à l'index {start_idx}")
            count += 1
    if count == 0:
        logger.info("⚠️ Aucun événement acceleration injecté.")
    else:
        logger.info(f"✅ Accélérations injectées : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_freinage(df, config):
    logger.info("🔄 Début injection : freinage")
    df = ensure_event_column_object(df)
    cfg = get_event_config("freinage", default={"dummy": True})
    hz = cfg.get("hz", 10)
    min_duration_pts = 1
    max_duration_pts = 3
    min_separation_s = 3.0
    min_separation_pts = int(min_separation_s * hz)
    count, total_attempts = 0, 0
    injected_indices = []
    while count < cfg["max_events"] and total_attempts < cfg["max_events"] * cfg["max_attempts"]:
        total_attempts += 1
        duration_pts = np.random.choice([min_duration_pts, max_duration_pts])
        start_idx = np.random.randint(0, len(df) - duration_pts)
        if injected_indices and min([abs(start_idx - idx) for idx in injected_indices]) < min_separation_pts:
            continue
        global_spacing_pts = cfg.get("global_spacing_pts", 5000)
        if has_recent_event(df, start_idx, global_spacing_pts):
            continue
        # Vérification de duplication géographique
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "freinage") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        indices = range(start_idx, start_idx + duration_pts)
        if all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            for j in indices:
                df.at[j, "acc_x"] = cfg["acc_x_start"] + cfg.get("acc_x_std", 0.3) * np.random.randn()
                for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                    mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                    std = cfg.get(f"gyro_{axis}_std", 0.5)
                    df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
            df.at[start_idx, "event"] = "freinage"
            injected_indices.append(start_idx)
            logger.debug(f"Freinage injecté à l'index {start_idx}")
            count += 1
    if count == 0:
        logger.info("⚠️ Aucun événement freinage injecté.")
    else:
        logger.info(f"✅ Freinages injectés : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_dos_dane(df, config):
    logger.info("🔄 Début injection : dos_dane")
    df = ensure_event_column_object(df)
    cfg = get_event_config("dos_dane", default={
        "dummy": True,
        "max_events": 5,
        "max_attempts": 10,
        "phase_length": 3,
        "amplitude_step": 3.0,
        "min_spacing_pts": 400,
        "global_spacing_pts": 5000,
        "gyro_axes_used": ["x", "y", "z"],
        "gyro_x_mean": 1.0,
        "gyro_x_std": 0.5,
        "gyro_y_mean": 1.0,
        "gyro_y_std": 0.5,
        "gyro_z_mean": 1.0,
        "gyro_z_std": 0.5,
    })
    hz = cfg.get("hz", 10)
    min_separation_s = 3.0
    min_separation_pts = int(min_separation_s * hz)
    count, total_attempts = 0, 0
    injected_indices = []
    while count < cfg["max_events"] and total_attempts < cfg.get("max_attempts", 10) * cfg["max_events"]:
        total_attempts += 1
        start_idx = np.random.randint(0, len(df))
        if injected_indices and min([abs(start_idx - idx) for idx in injected_indices]) < min_separation_pts:
            continue
        global_spacing_pts = cfg.get("global_spacing_pts", 5000)
        if has_recent_event(df, start_idx, global_spacing_pts):
            continue
        if df.at[start_idx, 'road_type'] not in ["residential", "service", "tertiary"]:
            continue
        # Vérification de duplication géographique
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "dos_dane") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        if pd.isna(df.at[start_idx, 'event']):
            amplitude = cfg.get("amplitude_step", 3.0)
            indices = [start_idx, start_idx + 1] if start_idx + 1 < len(df) else [start_idx]
            for i, idx in enumerate(indices):
                if idx >= len(df):
                    continue
                df.at[idx, "acc_z"] = amplitude if i == 0 else -amplitude
                for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                    mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                    std = cfg.get(f"gyro_{axis}_std", 0.5)
                    df.at[idx, f"gyro_{axis}"] = mean + std * np.random.randn()
            df.at[start_idx, "event"] = "dos_dane"
            injected_indices.append(start_idx)
            logger.debug(f"Événement dos_dane injecté à l'index {start_idx}")
            count += 1
    df = stronger_deduplication(df, window=3)
    logger.debug(f"[DOS_DANE] {count} événements injectés sur {total_attempts} tentatives")
    if count == 0:
        logger.info("⚠️ Aucun événement dos_dane injecté.")
    else:
        logger.info(f"✅ Dos d'âne injectés : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_nid_de_poule(df, config):
    logger.info("🔄 Début injection : nid_de_poule")
    df = ensure_event_column_object(df)
    cfg = get_event_config("nid_de_poule", default={"dummy": True})
    hz = cfg.get("hz", 10)
    min_separation_s = 3.0
    min_separation_pts = int(min_separation_s * hz)
    pattern = cfg.get("pattern", [cfg.get("pattern_value1", 8.0), cfg.get("pattern_value2", -10.0), cfg.get("pattern_value3", 7.0)])
    count, total_attempts = 0, 0
    injected_indices = []
    while count < cfg["max_events"] and total_attempts < cfg["max_events"] * cfg["max_attempts"]:
        total_attempts += 1
        start_idx = np.random.randint(0, len(df))
        if injected_indices and min([abs(start_idx - idx) for idx in injected_indices]) < min_separation_pts:
            continue
        global_spacing_pts = cfg.get("global_spacing_pts", 5000)
        if has_recent_event(df, start_idx, global_spacing_pts):
            continue
        if df.at[start_idx, 'road_type'] not in ["residential", "service"]:
            continue
        # Vérification de duplication géographique
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "nid_de_poule") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        if pd.isna(df.at[start_idx, "event"]):
            df.at[start_idx, "acc_z"] = pattern[0]
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                std = cfg.get(f"gyro_{axis}_std", 0.5)
                df.at[start_idx, f"gyro_{axis}"] = mean + std * np.random.randn()
            df.at[start_idx, "event"] = "nid_de_poule"
            injected_indices.append(start_idx)
            logger.debug(f"Événement nid_de_poule injecté à l'index {start_idx}")
            count += 1
    df = stronger_deduplication(df, window=3)
    if count == 0:
        logger.info("⚠️ Aucun événement nid_de_poule injecté.")
    else:
        logger.info(f"✅ Nids de poule injectés : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

MIN_STOP_SPACING = 300
def is_far_enough(new_index, selected_indices, min_spacing=MIN_STOP_SPACING):
    return all(abs(new_index - idx) >= min_spacing for idx in selected_indices)

def generate_stop(df, config):
    logger.info("🔄 Début injection : stop")
    df = ensure_event_column_object(df)
    cfg = get_event_config("stop", default={"dummy": True})
    global_spacing_pts = cfg.get("global_spacing_pts", 5000)
    hz = cfg.get("hz", 10)
    duration_pts = int(hz * 5)
    min_spacing_pts = int(hz * 60)
    last_stop_index = -np.inf
    candidate_indices = []
    for i in range(0, len(df) - duration_pts):
        if has_recent_event(df, i, global_spacing_pts):
            continue
        if i - last_stop_index < min_spacing_pts:
            continue
        indices = range(i, i + duration_pts)
        if all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            candidate_indices.append(i)
    selected_indices = []
    for idx in candidate_indices:
        if is_far_enough(idx, selected_indices, min_spacing=min_spacing_pts):
            if not has_recent_event(df, idx, spacing=min_spacing_pts):
                selected_indices.append(idx)
        if len(selected_indices) >= cfg["max_events"]:
            break
    logger.debug(f"Indices des stops retenus (espacés) : {selected_indices}")
    for i in selected_indices:
        indices = range(i, i + duration_pts)
        # Vérification de duplication géographique
        lat = df.at[i, "lat"]
        lon = df.at[i, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "stop_start") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        df.at[i, "event"] = "stop_start"
        for j in indices:
            df.at[j, "acc_x"] = cfg["acc_x_mean"] + cfg["acc_x_std"] * np.random.randn()
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                std = cfg.get(f"gyro_{axis}_std", 0.5)
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
    count = len(selected_indices)
    if count == 0:
        logger.info("⚠️ Aucun événement stop injecté.")
    else:
        logger.info(f"✅ Stops injectés : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_wait(df, config):
    logger.info("🔄 Début injection : wait")
    df = ensure_event_column_object(df)
    cfg = get_event_config("wait", default={"dummy": True})
    global_spacing_pts = cfg.get("global_spacing_pts", 5000)
    hz = cfg.get("hz", 10)
    duration_pts = int(hz * 3)
    min_spacing_pts = int(hz * 60)
    candidate_indices = []
    for i in range(0, len(df) - duration_pts):
        if has_recent_event(df, i, global_spacing_pts):
            continue
        indices = range(i, i + duration_pts)
        if all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            candidate_indices.append(i)
    selected_indices = []
    for idx in candidate_indices:
        if all(abs(idx - prev) >= min_spacing_pts for prev in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) >= cfg["max_events"]:
            break
    logger.debug(f"Indices des waits retenus (espacés) : {selected_indices}")
    for i in selected_indices:
        indices = range(i, i + duration_pts)
        # Vérification de duplication géographique
        lat = df.at[i, "lat"]
        lon = df.at[i, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "wait_start") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        df.at[i, "event"] = "wait_start"
        for j in indices:
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                std = cfg.get(f"gyro_{axis}_std", 0.5)
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
    count = len(selected_indices)
    if count == 0:
        logger.info("⚠️ Aucun événement wait injecté.")
    else:
        logger.info(f"✅ Attentes injectées : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_opening_door(df, config):
    logger.info("🔄 Début injection : ouverture_porte")
    df = ensure_event_column_object(df)
    cfg = get_event_config("ouverture_porte", default={
        "duration_pts": 5,
        "max_events": 5,
        "max_attempts": 10,
        "min_spacing_pts": 400,
        "global_spacing_pts": 5000,
        "gyro_axes_used": ["x", "y", "z"],
        "gyro_x_mean": 1.0,
        "gyro_x_std": 0.5,
        "gyro_y_mean": 0.0,
        "gyro_y_std": 0.5,
        "gyro_z_mean": 0.0,
        "gyro_z_std": 0.5
    })
    count, total_attempts = 0, 0
    duration_pts = cfg["duration_pts"]
    last_injected_idx = -np.inf
    max_events = config.get("events", {}).get("ouverture_porte", {}).get("max_events", 5)
    max_attempts = config.get("events", {}).get("ouverture_porte", {}).get("max_attempts", 10)
    while count < max_events and total_attempts < max_events * max_attempts:
        total_attempts += 1
        start_idx = np.random.randint(0, len(df) - duration_pts)
        global_spacing_pts = cfg.get("global_spacing_pts", 5000)
        if has_recent_event(df, start_idx, global_spacing_pts):
            continue
        if count > 0 and abs(start_idx - last_injected_idx) < cfg.get("min_spacing_pts", 400):
            continue
        if df.at[start_idx, "event"] not in ["stop_start", "wait_start", "stop", "wait"]:
            continue
        # Vérification de duplication géographique
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "ouverture_porte") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        indices = range(start_idx, start_idx + duration_pts)
        for j in indices:
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                std = cfg.get(f"gyro_{axis}_std", 0.5)
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "ouverture_porte"
        last_injected_idx = start_idx
        logger.debug(f"Ouverture de porte injectée à l'index {start_idx}")
        count += 1
    if count == 0:
        logger.info("⚠️ Aucun événement ouverture_porte injecté.")
    else:
        logger.info(f"✅ Ouvertures de porte injectées : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)

def stronger_deduplication(df, window=5):
    """
    Supprime les paquets d'événements dupliqués à quelques points près.
    Si un même label 'event' apparaît plusieurs fois à faible intervalle, on ne garde que le premier.
    """
    df = df.copy()
    previous_event_idx = None
    for idx, row in df.iterrows():
        if pd.isna(row["event"]):
            continue
        if previous_event_idx is not None and (idx - previous_event_idx) <= window and row["event"] == df.at[previous_event_idx, "event"]:
            df.at[idx, "event"] = np.nan
        else:
            previous_event_idx = idx
    return df


# --- Ajout de la fonction generate_trottoir ---
def generate_trottoir(df, config):
    logger.info("🔄 Début injection : trottoir")
    df = ensure_event_column_object(df)
    cfg = get_event_config("trottoir", default={"dummy": True})
    hz = cfg.get("hz", 10)
    min_separation_s = 3.0
    min_separation_pts = int(min_separation_s * hz)
    count, total_attempts = 0, 0
    injected_indices = []
    while count < cfg["max_events"] and total_attempts < cfg["max_events"] * cfg["max_attempts"]:
        total_attempts += 1
        start_idx = np.random.randint(0, len(df))
        if injected_indices and min([abs(start_idx - idx) for idx in injected_indices]) < min_separation_pts:
            continue
        global_spacing_pts = cfg.get("global_spacing_pts", 5000)
        if has_recent_event(df, start_idx, global_spacing_pts):
            continue
        if df.at[start_idx, 'road_type'] not in ["residential", "service", "tertiary"]:
            continue
        # Vérification de duplication géographique
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        epsilon = 1e-5  # ~1 mètre
        nearby_duplicates = df[
            (df["event"] == "trottoir") &
            (df["lat"].between(lat - epsilon, lat + epsilon)) &
            (df["lon"].between(lon - epsilon, lon + epsilon))
        ]
        if not nearby_duplicates.empty:
            continue
        if pd.isna(df.at[start_idx, "event"]):
            df.at[start_idx, "acc_z"] = cfg.get("acc_z", 7.0)
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                mean = cfg.get(f"gyro_{axis}_mean", 1.0)
                std = cfg.get(f"gyro_{axis}_std", 0.5)
                df.at[start_idx, f"gyro_{axis}"] = mean + std * np.random.randn()
            df.at[start_idx, "event"] = "trottoir"
            injected_indices.append(start_idx)
            logger.debug(f"Trottoir injecté à l'index {start_idx}")
            count += 1
    df = stronger_deduplication(df, window=3)
    if count == 0:
        logger.info("⚠️ Aucun événement trottoir injecté.")
    else:
        logger.info(f"✅ Trottoirs injectés : {count}")
    return df.sort_values("timestamp").reset_index(drop=True)