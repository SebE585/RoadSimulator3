import logging
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from simulator.events.gyro import recompute_inertial_acceleration

from core.config_loader import load_config

def interpolate_target_speed_progressively(df, alpha=0.1, force=False, config=None):
    if config is not None:
        force = config.get("simulation", {}).get("force_target_speed", False)
    df = df.copy()
    if not force:
        return df
    if "target_speed" not in df.columns:
        return df

    target = df["target_speed"].ffill().bfill().to_numpy()
    smoothed = target.copy()
    logging.debug(f"Valeurs initiales target_speed : min={np.nanmin(target):.2f}, max={np.nanmax(target):.2f}")

    for i in range(1, len(smoothed)):
        smoothed[i] = (1 - alpha) * smoothed[i-1] + alpha * target[i]
        if np.isnan(smoothed[i]):
            logging.warning(f"NaN détecté à l’index {i} dans smoothed target_speed")

    df["target_speed"] = smoothed
    df["target_speed"] = df["target_speed"].clip(lower=0, upper=150)

    return df

def apply_target_speed_by_road_type(df, speed_by_type=None):
    if speed_by_type is None:
        from core.config_loader import load_config as get_config
        config = get_config()
        speed_by_type = config.get("simulation", {}).get("target_speed_by_road_type", {})
    else:
        logging.debug("speed_by_type reçu : %s", speed_by_type)
        for k, v in speed_by_type.items():
            try:
                speed_by_type[k] = float(v)
            except Exception as e:
                logging.warning(f"Clé {k} avec valeur invalide : {v} ({e}) → remplacée par 50")
                speed_by_type[k] = 50.0

    logging.debug("Dictionnaire utilisé pour speed_by_type : %s", speed_by_type)
    df = df.copy()
    df["target_speed"] = df["road_type"].map(speed_by_type)
    df["target_speed"] = df["target_speed"].ffill().bfill()
    if df["target_speed"].isna().any():
        logging.warning("Certaines target_speed sont toujours NaN après application.")
    logging.debug("Statistiques target_speed par road_type :\n%s", df.groupby("road_type")["target_speed"].describe())
    return df

def smooth_target_speed(df, window=5, config=None):
    if config is not None:
        df = apply_target_speed_by_road_type(df, speed_by_type=config["simulation"]["target_speed_by_road_type"])
    df = df.copy()
    if "target_speed" in df.columns:
        df["target_speed"] = df["target_speed"].rolling(window=window, min_periods=1, center=True).mean()
    return df

def recompute_speed(df, iterations=5, alpha=0.3, config=None):
    df = df.copy()
    if "heading" in df.columns:
        df["heading"] = df["heading"].ffill().bfill()
    if df["timestamp"].isna().any():
        logging.warning("Timestamps NaN détectés")
    if not df["timestamp"].is_monotonic_increasing:
        logging.warning("Timestamps non croissants détectés — cela peut causer des vitesses négatives.")
    distances = [0]
    time_deltas = [0]
    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        dist = geodesic(p1, p2).meters
        if np.isnan(dist) or dist < 0:
            logging.warning(f"Distance invalide à l’index {i} → dist remplacée par 0.0m")
            dist = 0.0
        dt = (df.loc[i, 'timestamp'] - df.loc[i-1, 'timestamp']).total_seconds()
        if dt <= 0 or np.isnan(dt):
            logging.warning(f"∆t non valide à l’index {i} → valeur remplacée par 0.1s")
            dt = 0.1
        distances.append(dist)
        time_deltas.append(dt)
    df["distance_m"] = distances
    df["delta_t"] = time_deltas
    speeds_m_s = [0 if dt <= 0 else max(0, d / dt) for d, dt in zip(distances, time_deltas)]
    df["speed"] = pd.Series(speeds_m_s) * 3.6
    df["speed"] = df["speed"].clip(lower=0, upper=110)

    if "target_speed" in df.columns:
        for _ in range(iterations):
            modulated_target_speed = df["target_speed"].fillna(df["speed"])
            if "sinuosity" in df.columns:
                sinuosity_modulation = (1 - 0.3 * df["sinuosity"].clip(0, 1))
                modulated_target_speed = modulated_target_speed * sinuosity_modulation
            df["speed"] = (1 - alpha) * df["speed"] + alpha * modulated_target_speed

    df["speed"] = df["speed"].rolling(window=5, min_periods=1, center=True).mean()
    df["speed"] = df["speed"].clip(lower=0, upper=150)

    df = adjust_speed_progressively(df, config=config)

    return df

def adjust_speed_progressively(df, max_acc_kmh_per_s=10, hz=10, config=None):
    df = df.copy()
    if config is None:
        config = load_config()
    target_speeds = config.get("simulation", {}).get("target_speed_by_road_type", {})
    if config is not None:
        simulation_config = config.get("simulation") if isinstance(config, dict) else None
        if simulation_config is not None:
            max_acc_raw = simulation_config.get("max_acc_kmh_per_s", max_acc_kmh_per_s)
            hz_raw = simulation_config.get("hz", hz)
        else:
            max_acc_raw = max_acc_kmh_per_s
            hz_raw = hz

        logging.debug("max_acc_raw = %s", max_acc_raw)

        # Si max_acc_raw est un dict, on récupère la valeur float dans la clé 'value'
        if isinstance(max_acc_raw, dict):
            max_acc_raw = max_acc_raw.get('value', 10)
            logging.debug("max_acc_raw extrait du dict : %s", max_acc_raw)

        try:
            max_acc_kmh_per_s = float(max_acc_raw)
        except Exception as e:
            logging.warning(f"max_acc_kmh_per_s invalide : {max_acc_raw} ({e}) → valeur par défaut 10.0")
            max_acc_kmh_per_s = 10.0

        try:
            hz = float(hz_raw)
        except Exception as e:
            logging.warning(f"hz invalide : {hz_raw} ({e}) → valeur par défaut 10")
            hz = 10

    if "target_speed" not in df.columns or "speed" not in df.columns:
        return df

    df["target_speed"] = df["target_speed"].ffill().bfill()
    if isinstance(max_acc_kmh_per_s, dict):
        max_acc_kmh_per_s = max_acc_kmh_per_s.get('value', 10.0)
    max_delta_per_step = max_acc_kmh_per_s / hz
    speeds = df["speed"].to_numpy()
    targets = df["target_speed"].to_numpy()

    for i in range(1, len(speeds)):
        delta = targets[i] - speeds[i - 1]
        if abs(delta) <= max_delta_per_step:
            speeds[i] = targets[i]
        else:
            speeds[i] = speeds[i - 1] + np.sign(delta) * max_delta_per_step

    speeds = np.clip(speeds, a_min=0, a_max=None)
    df["speed"] = speeds

    # Ajout de la logique complémentaire pour corrections ascendantes
    road_types = df.get("road_type", pd.Series(["unknown"] * len(df))).ffill().bfill().to_numpy()
    i = 0
    while i < len(df) - 1:
        current_speed = speeds[i]
        target_speed = targets[i]
        road_type = road_types[i]
        next_road_type = road_types[i+1]
        if current_speed > target_speed:
            start = i
            while i < len(df) - 1 and speeds[i] > targets[i]:
                i += 1
            end = i
            df.loc[start:end, "speed"] = np.linspace(speeds[start], targets[end], end - start + 1)
            print(f"[ADJUST] Correction descendante : {road_type} ({current_speed:.2f}) >> {next_road_type} ({target_speed:.2f}) → -{current_speed - target_speed:.2f} km/h")
        elif current_speed < target_speed:
            start = i
            while i < len(df) - 1 and speeds[i] < targets[i]:
                i += 1
            end = i
            df.loc[start:end, "speed"] = np.linspace(current_speed, target_speed, end - start + 1)
            print(f"[ADJUST] Correction ascendante : {road_type} ({current_speed:.2f}) << {next_road_type} ({target_speed:.2f}) → +{target_speed - current_speed:.2f} km/h")
        else:
            i += 1

    # Import ici pour éviter les problèmes d'import circulaire
    from core.kinematics import recompute_inertial_acceleration
    df = recompute_inertial_acceleration(df, hz=hz)
    return df

def cap_speed_to_target(df, alpha=0.2):
    df = df.copy()
    if "speed" not in df.columns or "target_speed" not in df.columns:
        return df

    speed = df["speed"].to_numpy()
    target = df["target_speed"].ffill().bfill().to_numpy()

    for i in range(1, len(speed)):
        delta = target[i] - speed[i]
        if delta < 0:
            speed[i] = speed[i-1] + alpha * delta
        else:
            speed[i] = speed[i]

    df["speed"] = speed
    return df

def cap_global_speed_delta(df, max_delta_kmh=15.0):
    df = df.copy()
    speed = df["speed"].values
    capped_speed = speed.copy()
    delta = np.diff(speed, prepend=speed[0])
    for i in range(1, len(speed)):
        if abs(delta[i]) > max_delta_kmh:
            capped_speed[i] = capped_speed[i-1] + np.sign(delta[i]) * max_delta_kmh
    df["speed"] = capped_speed
    return df

def simulate_variable_speed(df: pd.DataFrame, full_config: dict) -> pd.DataFrame:
    if 'road_type' not in df.columns:
        df['speed'] = full_config.get('simulation', {}).get('initial_speed_kmh', 30)
        return df

    speed_by_type = full_config.get('simulation', {}).get('target_speed_by_road_type', {
        'motorway': 110, 'primary': 70, 'secondary': 50, 'tertiary': 40,
        'residential': 35, 'service': 25, 'unknown': 50
    })

    df['target_speed'] = df['road_type'].map(speed_by_type).fillna(50)
    np.random.seed(42)
    noise_std_ratio = 0.02
    noise = np.random.normal(0, df['target_speed'] * noise_std_ratio)
    df['target_speed'] = (df['target_speed'] + noise).clip(lower=10, upper=130)

    alpha = 0.15
    iterations = 15
    speed = df['target_speed'].copy()
    for _ in range(iterations):
        speed = speed.shift(1).fillna(speed.iloc[0]) * (1 - alpha) + speed * alpha
    speed = speed.rolling(window=15, min_periods=1, center=True).mean()

    max_acc_per_step = 2.5
    speed_diff = speed.diff().fillna(0)
    speed_diff_clipped = speed_diff.clip(-max_acc_per_step, max_acc_per_step)
    speed = speed.shift(1).fillna(speed.iloc[0]) + speed_diff_clipped
    speed = speed.clip(lower=0)

    df['speed'] = speed
    df['ramp_applied'] = True
    return df