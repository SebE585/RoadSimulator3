def _inject_events_loop(df, cfg, event_name, propose_fn, apply_fn, is_valid_location_fn=None):
    """
    Boucle utilitaire pour injecter des événements dans un DataFrame.

    Cette version *sépare* la proposition d'emplacement (propose_fn) de
    l'application de l'injection (apply_fn) pour éviter toute mutation du
    DataFrame avant la validation finale. Cela corrige les doublons
    liés aux écritures précoces dans df par l'ancien `inject_fn`.

    Args:
        df (pd.DataFrame): DataFrame à modifier.
        cfg (dict): Configuration de l'événement.
        event_name (str): Nom de l'événement.
        propose_fn (callable): Fonction qui reçoit `injected_indices` et
            retourne l'une des formes suivantes (pas de mutation ici !) :
              - (start_idx, duration_pts) pour une proposition OK
              - ("skip", reason) pour instrumenter une raison de rejet
              - None pour "aucune proposition"
        apply_fn (callable): Fonction appelée *après validation*. Signature :
            apply_fn(start_idx, duration_pts) -> None. C'est elle qui
            applique réellement les écritures dans `df`.
        is_valid_location_fn (callable, optional): Validation externe
            d'emplacement (start_idx) -> bool.
    Returns:
        pd.DataFrame: DataFrame modifié.
    """
    hz = cfg.get("hz", 10)
    min_separation_s = 3.0
    min_separation_pts = int(min_separation_s * hz)
    count, total_attempts = 0, 0
    injected_indices = []
    max_events = cfg.get("max_events", 1)
    max_attempts = cfg.get("max_attempts", 10)
    global_spacing_pts = cfg.get("global_spacing_pts", 5000)

    # Diagnostics counters
    diag = {
        "proposed": 0,
        "proposed_none": 0,
        "skip_reason": {},  # from propose_fn ('skip', reason)
        "too_close_self": 0,
        "global_spacing": 0,
        "invalid_location": 0,
        "accepted": 0,
    }

    # Keep a short sample of attempts for debug (not all to avoid huge memory)
    attempts_sample = []

    while count < max_events and total_attempts < max_events * max_attempts:
        total_attempts += 1
        # propose_fn doit proposer : (start_idx, duration_pts) ou ('skip', reason) ou None
        proposal = propose_fn(injected_indices)
        if proposal is None:
            diag["proposed_none"] += 1
            continue

        # Normalize outputs (aucune mutation de df ne doit avoir eu lieu à ce stade)
        if isinstance(proposal, tuple) and len(proposal) == 2 and all(isinstance(x, (int, np.integer)) for x in proposal):
            start_idx, duration_pts = proposal
        elif isinstance(proposal, tuple) and len(proposal) >= 2 and proposal[0] == "skip":
            reason = str(proposal[1])
            diag["skip_reason"][reason] = diag["skip_reason"].get(reason, 0) + 1
            continue
        else:
            # Unexpected shape → treat as None
            diag["proposed_none"] += 1
            continue

        diag["proposed"] += 1

        # Enforce local spacing between same-event injections in this pass
        if injected_indices and min([abs(start_idx - idx) for idx in injected_indices]) < min_separation_pts:
            diag["too_close_self"] += 1
            attempts_sample.append((event_name, start_idx, "too_close_self"))
            continue

        # Enforce global spacing against ANY prior event already present in df
        if global_spacing_pts and _has_recent_event(df, start_idx, global_spacing_pts):
            diag["global_spacing"] += 1
            attempts_sample.append((event_name, start_idx, "global_spacing"))
            continue

        # External location validator (e.g., road_type filter)
        if is_valid_location_fn is not None and not is_valid_location_fn(start_idx):
            diag["invalid_location"] += 1
            attempts_sample.append((event_name, start_idx, "invalid_location"))
            continue

        # ✅ Validation passée — on applique réellement l'injection maintenant
        apply_fn(start_idx, duration_pts)
        injected_indices.append(start_idx)
        count += 1
        diag["accepted"] += 1

    # Final diagnostics log (concise)
    if diag["accepted"] < max_events:
        logger.debug(
            "[INJECT_DIAG][%s] accepted=%d / wanted=%d | proposed=%d (none=%d) | too_close_self=%d | global_spacing=%d | invalid_location=%d | skip_reason=%s",
            event_name,
            diag["accepted"], max_events,
            diag["proposed"], diag["proposed_none"],
            diag["too_close_self"], diag["global_spacing"], diag["invalid_location"],
            diag["skip_reason"],
        )
        if attempts_sample:
            sample_str = ", ".join([f"(idx={i},why={w})" for _, i, w in attempts_sample[:10]])
            logger.debug("[INJECT_DIAG][%s] sample: %s", event_name, sample_str)
    else:
        logger.debug("[INJECT_DIAG][%s] All OK: accepted=%d", event_name, diag["accepted"]) 

    return df
import numpy as np
import pandas as pd
import logging
from core.utils import ensure_event_column_object
from simulator.events.config import get_event_config
from deprecated import deprecated

"""
Generation des événements inertiels réalistes pour RoadSimulator3.
Chaque fonction injecte un type d'événement spécifique dans la colonne 'event' 
ainsi que des signaux accéléromètre et gyroscope réalistes.
"""


logger = logging.getLogger(__name__)

def _apply_overrides(cfg: dict, config: dict, event_name: str) -> dict:
    """
    Merge YAML overrides coming from `config['events'][event_name]` into the per-event cfg.
    Values in `config` take precedence over `cfg`.
    """
    try:
        overrides = (config or {}).get("events", {}).get(event_name, {})
        if overrides:
            cfg = {**cfg, **overrides}
    except Exception:
        # Keep silent: fall back to cfg if structure is unexpected
        pass
    return cfg

def _has_recent_event(df, idx, spacing):
    start = max(0, idx - spacing)
    return df.loc[start:idx, 'event'].notna().any()


def generate_acceleration(df, config):
    """Inject acceleration events into the DataFrame.

    This function injects 'acceleration' events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_x, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected acceleration events and modified IMU signals.
    """
    logger.info("🔄 Début injection : acceleration")
    df = ensure_event_column_object(df)
    cfg = get_event_config("acceleration")
    cfg = _apply_overrides(cfg, config, "acceleration")
    logger.debug("[CFG][acceleration] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    min_duration_pts = 1
    max_duration_pts = 3
    injected_count = [0]
    def propose_fn(injected_indices):
        duration_pts = np.random.choice([min_duration_pts, max_duration_pts])
        start_idx = np.random.randint(0, len(df) - duration_pts)
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "acceleration", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        indices = range(start_idx, start_idx + duration_pts)
        if not all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            return ("skip", "occupied_indices")
        return (start_idx, duration_pts)

    def apply_fn(start_idx, duration_pts):
        indices = range(start_idx, start_idx + duration_pts)
        for j in indices:
            df.at[j, "acc_x"] = cfg["acc_x_mean"] + cfg["acc_x_std"] * np.random.randn()
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                gyro_key_mean = f"gyro_{axis}_mean"
                gyro_key_std = f"gyro_{axis}_std"
                mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 1.0
                std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.5
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "acceleration"
        logger.debug(f"Accélération injectée à l'index {start_idx}")
        injected_count[0] += 1

    df = _inject_events_loop(df, cfg, "acceleration", propose_fn, apply_fn)
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement acceleration injecté.")
    else:
        logger.info(f"✅ Accélérations injectées : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_freinage(df, config):
    """Inject braking events into the DataFrame.

    This function injects 'freinage' (braking) events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_x, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected braking events and modified IMU signals.
    """
    logger.info("🔄 Début injection : freinage")
    df = ensure_event_column_object(df)
    cfg = get_event_config("freinage")
    cfg = _apply_overrides(cfg, config, "freinage")
    logger.debug("[CFG][freinage] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    min_duration_pts = 1
    max_duration_pts = 3
    injected_count = [0]
    def propose_fn(injected_indices):
        duration_pts = np.random.choice([min_duration_pts, max_duration_pts])
        start_idx = np.random.randint(0, len(df) - duration_pts)
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "freinage", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        indices = range(start_idx, start_idx + duration_pts)
        if not all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            return ("skip", "occupied_indices")
        return (start_idx, duration_pts)

    def apply_fn(start_idx, duration_pts):
        indices = range(start_idx, start_idx + duration_pts)
        for j in indices:
            df.at[j, "acc_x"] = cfg["acc_x_start"] + cfg.get("acc_x_std", 0.3) * np.random.randn()
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                gyro_key_mean = f"gyro_{axis}_mean"
                gyro_key_std = f"gyro_{axis}_std"
                mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 1.0
                std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.5
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "freinage"
        logger.debug(f"Freinage injecté à l'index {start_idx}")
        injected_count[0] += 1

    df = _inject_events_loop(df, cfg, "freinage", propose_fn, apply_fn)
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement freinage injecté.")
    else:
        logger.info(f"✅ Freinages injectés : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_dos_dane(df, config):
    """Inject speed bump (dos d'âne) events into the DataFrame.

    This function injects 'dos_dane' (speed bump) events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_z, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected speed bump events and modified IMU signals.
    """
    logger.info("🔄 Début injection : dos_dane")
    df = ensure_event_column_object(df)
    cfg = get_event_config("dos_dane")
    cfg = _apply_overrides(cfg, config, "dos_dane")
    logger.debug("[CFG][dos_dane] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    injected_count = [0]
    def propose_fn(injected_indices):
        start_idx = np.random.randint(0, len(df))
        if df.at[start_idx, 'road_type'] not in ["residential", "service", "tertiary"]:
            return ("skip", "road_type")
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "dos_dane", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        if not pd.isna(df.at[start_idx, 'event']):
            return ("skip", "occupied_indices")
        return (start_idx, 1)

    def apply_fn(start_idx, duration_pts):
        amplitude = cfg.get("amplitude_step", 3.0)
        indices = [start_idx, start_idx + 1] if start_idx + 1 < len(df) else [start_idx]
        for i, idx in enumerate(indices):
            if idx >= len(df):
                continue
            df.at[idx, "acc_z"] = amplitude if i == 0 else -amplitude
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                gyro_key_mean = f"gyro_{axis}_mean"
                gyro_key_std = f"gyro_{axis}_std"
                mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 1.0
                std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.5
                df.at[idx, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "dos_dane"
        logger.debug(f"Événement dos_dane injecté à l'index {start_idx}")
        injected_count[0] += 1

    def is_valid_location_fn(start_idx):
        return df.at[start_idx, 'road_type'] in ["residential", "service", "tertiary"]

    df = _inject_events_loop(df, cfg, "dos_dane", propose_fn, apply_fn, is_valid_location_fn)
    df = _stronger_deduplication(df, window=3)
    logger.debug(f"[DOS_DANE] {injected_count[0]} événements injectés")
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement dos_dane injecté.")
    else:
        logger.info(f"✅ Dos d'âne injectés : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)

def generate_nid_de_poule(df, config):
    """Inject pothole (nid de poule) events into the DataFrame.

    This function injects 'nid_de_poule' (pothole) events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_z, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected pothole events and modified IMU signals.
    """
    logger.info("🔄 Début injection : nid_de_poule")
    df = ensure_event_column_object(df)
    cfg = get_event_config("nid_de_poule")
    cfg = _apply_overrides(cfg, config, "nid_de_poule")
    logger.debug("[CFG][nid_de_poule] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    pattern = cfg.get("pattern", [cfg.get("pattern_value1", 8.0), cfg.get("pattern_value2", -10.0), cfg.get("pattern_value3", 7.0)])
    injected_count = [0]
    def propose_fn(injected_indices):
        start_idx = np.random.randint(0, len(df))
        if df.at[start_idx, 'road_type'] not in ["residential", "service"]:
            return ("skip", "road_type")
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "nid_de_poule", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        if not pd.isna(df.at[start_idx, "event"]):
            return ("skip", "occupied_indices")
        return (start_idx, 1)

    def apply_fn(start_idx, duration_pts):
        df.at[start_idx, "acc_z"] = pattern[0]
        for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
            gyro_key_mean = f"gyro_{axis}_mean"
            gyro_key_std = f"gyro_{axis}_std"
            mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 1.0
            std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.5
            df.at[start_idx, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "nid_de_poule"
        logger.debug(f"Événement nid_de_poule injecté à l'index {start_idx}")
        injected_count[0] += 1

    def is_valid_location_fn(start_idx):
        return df.at[start_idx, 'road_type'] in ["residential", "service"]

    df = _inject_events_loop(df, cfg, "nid_de_poule", propose_fn, apply_fn, is_valid_location_fn)
    df = _stronger_deduplication(df, window=3)
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement nid_de_poule injecté.")
    else:
        logger.info(f"✅ Nids de poule injectés : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)


@deprecated(reason="Cette fonction n'est plus utilisée dans le code courant.")
def generate_opening_door(df, config):
    """Inject door opening events into the DataFrame.

    This function injects 'ouverture_porte' (door opening) events into the 'event' column of the input DataFrame,
    modifying the IMU (gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected door opening events and modified IMU signals.
    """
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
        if _has_recent_event(df, start_idx, global_spacing_pts):
            continue
        if count > 0 and abs(start_idx - last_injected_idx) < cfg.get("min_spacing_pts", 400):
            continue
        if df.at[start_idx, "event"] not in ["stop_start", "wait_start", "stop", "wait"]:
            continue
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "ouverture_porte", lat, lon, epsilon=1e-5):
            continue
        indices = range(start_idx, start_idx + duration_pts)
        for j in indices:
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                gyro_key_mean = f"gyro_{axis}_mean"
                gyro_key_std = f"gyro_{axis}_std"
                mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 1.0
                std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.5
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

def _stronger_deduplication(df, window=5):
    """Remove clusters of duplicated events within a window.

    If the same event label appears multiple times within a short interval, only the first is kept.

    Args:
        df (pd.DataFrame): Input DataFrame containing event data.
        window (int, optional): Number of points within which to deduplicate events. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with deduplicated events.
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


def generate_trottoir(df, config):
    """Inject curb (trottoir) events into the DataFrame.

    This function injects 'trottoir' (curb) events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_z, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected curb events and modified IMU signals.
    """
    logger.info("🔄 Début injection : trottoir")
    df = ensure_event_column_object(df)
    cfg = get_event_config("trottoir")
    cfg = _apply_overrides(cfg, config, "trottoir")
    logger.debug("[CFG][trottoir] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    injected_count = [0]
    def propose_fn(injected_indices):
        start_idx = np.random.randint(0, len(df))
        if df.at[start_idx, 'road_type'] not in ["residential", "service", "tertiary"]:
            return ("skip", "road_type")
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "trottoir", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        if not pd.isna(df.at[start_idx, "event"]):
            return ("skip", "occupied_indices")
        return (start_idx, 1)

    def apply_fn(start_idx, duration_pts):
        df.at[start_idx, "acc_z"] = cfg.get("acc_z", 7.0)
        for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
            gyro_key_mean = f"gyro_{axis}_mean"
            gyro_key_std = f"gyro_{axis}_std"
            mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 1.0
            std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.5
            df.at[start_idx, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "trottoir"
        logger.debug(f"Trottoir injecté à l'index {start_idx}")
        injected_count[0] += 1

    def is_valid_location_fn(start_idx):
        return df.at[start_idx, 'road_type'] in ["residential", "service", "tertiary"]

    df = _inject_events_loop(df, cfg, "trottoir", propose_fn, apply_fn, is_valid_location_fn)
    df = _stronger_deduplication(df, window=3)
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement trottoir injecté.")
    else:
        logger.info(f"✅ Trottoirs injectés : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)

def _is_nearby_duplicate(df, event, lat, lon, epsilon=1e-5):
    """Check if a similar event exists nearby in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing at least columns 'event', 'lat', 'lon'.
        event (str): Event name to check for.
        lat (float): Latitude of the candidate event.
        lon (float): Longitude of the candidate event.
        epsilon (float, optional): Proximity threshold for latitude and longitude. Defaults to 1e-5 (~1 meter).

    Returns:
        bool: True if a nearby duplicate event exists, False otherwise.
    """
    nearby_duplicates = df[
        (df["event"] == event) &
        (df["lat"].between(lat - epsilon, lat + epsilon)) &
        (df["lon"].between(lon - epsilon, lon + epsilon))
    ]
    return not nearby_duplicates.empty
def generate_stop(df, config):
    """Inject stop events into the DataFrame.

    This function injects 'stop' events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_x, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected stop events and modified IMU signals.
    """
    logger.info("🔄 Début injection : stop")
    df = ensure_event_column_object(df)
    cfg = get_event_config("stop")
    cfg = _apply_overrides(cfg, config, "stop")
    logger.debug("[CFG][stop] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    injected_count = [0]
    def propose_fn(injected_indices):
        min_duration_pts = cfg.get("min_duration_pts", 1)
        max_duration_pts = cfg.get("max_duration_pts", 3)
        duration_pts = np.random.choice([min_duration_pts, max_duration_pts])
        start_idx = np.random.randint(0, len(df) - duration_pts)
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "stop", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        indices = range(start_idx, start_idx + duration_pts)
        if not all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            return ("skip", "occupied_indices")
        return (start_idx, duration_pts)

    def apply_fn(start_idx, duration_pts):
        indices = range(start_idx, start_idx + duration_pts)
        for j in indices:
            df.at[j, "acc_x"] = cfg.get("acc_x", 0.0) + cfg.get("acc_x_std", 0.3) * np.random.randn()
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                gyro_key_mean = f"gyro_{axis}_mean"
                gyro_key_std = f"gyro_{axis}_std"
                mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 0.0
                std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.2
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "stop"
        logger.debug(f"Stop injecté à l'index {start_idx}")
        injected_count[0] += 1

    df = _inject_events_loop(df, cfg, "stop", propose_fn, apply_fn)
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement stop injecté.")
    else:
        logger.info(f"✅ Stops injectés : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)


def generate_wait(df, config):
    """Inject wait events into the DataFrame.

    This function injects 'wait' events into the 'event' column of the input DataFrame,
    modifying the IMU (acc_x, gyro_x/y/z) signal columns at the selected indices.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns 'event', 'lat', 'lon'.
        config (dict): Global configuration dictionary (unused here but required for API compatibility).

    Returns:
        pd.DataFrame: Updated DataFrame with injected wait events and modified IMU signals.
    """
    logger.info("🔄 Début injection : wait")
    df = ensure_event_column_object(df)
    cfg = get_event_config("wait")
    cfg = _apply_overrides(cfg, config, "wait")
    logger.debug("[CFG][wait] max_events=%s max_attempts=%s global_spacing_pts=%s hz=%s",
                 cfg.get("max_events"), cfg.get("max_attempts"), cfg.get("global_spacing_pts"), cfg.get("hz"))
    injected_count = [0]
    def propose_fn(injected_indices):
        min_duration_pts = cfg.get("min_duration_pts", 1)
        max_duration_pts = cfg.get("max_duration_pts", 3)
        duration_pts = np.random.choice([min_duration_pts, max_duration_pts])
        start_idx = np.random.randint(0, len(df) - duration_pts)
        lat = df.at[start_idx, "lat"]
        lon = df.at[start_idx, "lon"]
        if _is_nearby_duplicate(df, "wait", lat, lon, epsilon=1e-5):
            return ("skip", "nearby_duplicate")
        indices = range(start_idx, start_idx + duration_pts)
        if not all(pd.isna(df.at[j, 'event']) for j in indices if j < len(df)):
            return ("skip", "occupied_indices")
        return (start_idx, duration_pts)

    def apply_fn(start_idx, duration_pts):
        indices = range(start_idx, start_idx + duration_pts)
        for j in indices:
            df.at[j, "acc_x"] = cfg.get("acc_x", 0.0) + cfg.get("acc_x_std", 0.3) * np.random.randn()
            for axis in cfg.get("gyro_axes_used", ["x", "y", "z"]):
                gyro_key_mean = f"gyro_{axis}_mean"
                gyro_key_std = f"gyro_{axis}_std"
                mean = cfg[gyro_key_mean] if gyro_key_mean in cfg else 0.0
                std = cfg[gyro_key_std] if gyro_key_std in cfg else 0.2
                df.at[j, f"gyro_{axis}"] = mean + std * np.random.randn()
        df.at[start_idx, "event"] = "wait"
        logger.debug(f"Wait injecté à l'index {start_idx}")
        injected_count[0] += 1

    df = _inject_events_loop(df, cfg, "wait", propose_fn, apply_fn)
    if injected_count[0] == 0:
        logger.info("⚠️ Aucun événement wait injecté.")
    else:
        logger.info(f"✅ Waits injectés : {injected_count[0]}")
    return df.sort_values("timestamp").reset_index(drop=True)