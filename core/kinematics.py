import pandas as pd
import numpy as np
import warnings

# Deprecation decorator
def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated and may be removed in future versions.", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper
from geopy.distance import geodesic
from core.config_loader import load_full_config
from core.geo_utils import haversine_distance

def interpolate_target_speed_progressively(df, iterations=10, alpha=0.5, window=10, force=False):
    df = df.copy()
    if "target_speed" not in df.columns:
        raise ValueError("Colonne 'target_speed' manquante dans le DataFrame.")
    df["target_speed"] = df["target_speed"].clip(lower=0, upper=150)
    print(f"[DEBUG] target_speed (initiale) : min={df['target_speed'].min():.2f}, max={df['target_speed'].max():.2f}")
    for i in range(iterations):
        diff = df["target_speed"] - df["speed"]
        df["speed"] = df["speed"] + alpha * diff
        df["speed"] = df["speed"].clip(lower=0)
        if df["speed"].isna().any():
            print(f"[WARN] NaN détecté à l’itération {i+1}")
        if force:
            print(f"[DEBUG] Itération {i+1}/{iterations} : sans lissage")
        else:
            smoothed = smooth_signal(df["speed"], window=window).clip(lower=0, upper=150)
            df["speed"] = smoothed
            print(f"[DEBUG] Itération {i+1}/{iterations} : min={df['speed'].min():.2f}, max={df['speed'].max():.2f}")
    return df

def compute_distance(df):
    df = df.reset_index(drop=True)
    distances = [0.0]
    for i in range(1, len(df)):
        pt1 = (df.at[i-1, 'lat'], df.at[i-1, 'lon'])
        pt2 = (df.at[i, 'lat'], df.at[i, 'lon'])
        dist = geodesic(pt1, pt2).meters
        distances.append(dist)
    df = df.copy()
    df['distance_m'] = distances
    return df

def compute_heading(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lon_rad = np.radians(lon2 - lon1)

    x = np.sin(delta_lon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon_rad)

    heading_rad = np.arctan2(x, y)
    heading_deg = (np.degrees(heading_rad) + 360) % 360
    return heading_deg

def calculate_heading(df):
    """
    Calcule le heading (cap) entre chaque paire de points GPS consécutifs dans un DataFrame.
    Ajoute la colonne 'heading' dans le DataFrame.
    """
    lat1 = df['lat'].iloc[:-1].values
    lon1 = df['lon'].iloc[:-1].values
    lat2 = df['lat'].iloc[1:].values
    lon2 = df['lon'].iloc[1:].values

    headings = compute_heading(lat1, lon1, lat2, lon2)
    # Pour le dernier point, on peut répéter le dernier heading calculé
    headings = np.append(headings, headings[-1])

    df = df.copy()
    df['heading'] = pd.Series(headings).ffill().bfill()
    return df

def compute_curvature(heading):
    heading_rad = np.radians(heading)
    d_heading = np.gradient(heading_rad)
    curvature = np.gradient(d_heading)
    return curvature

def compute_sinuosity(heading, window):
    heading_rad = np.radians(heading)
    sinuosity = np.full(len(heading), np.nan)
    half_window = window // 2
    for i in range(half_window, len(heading) - half_window):
        segment = heading_rad[i - half_window:i + half_window + 1]
        delta = np.max(segment) - np.min(segment)
        sinuosity[i] = delta
    return sinuosity

def smooth_signal(signal, window=5):
    return pd.Series(signal).rolling(window=window, min_periods=1, center=True).mean()

def recompute_speed(df, iterations=10, alpha=0.5):
    distances = [0]
    time_deltas = [0]

    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        dist = geodesic(p1, p2).meters
        dt = (df.loc[i, 'timestamp'] - df.loc[i-1, 'timestamp']).total_seconds()
        distances.append(dist)
        time_deltas.append(dt)

    df['distance_m'] = distances
    df['delta_t'] = time_deltas

    speeds_m_s = [0 if dt == 0 else d / dt for d, dt in zip(distances, time_deltas)]
    df['speed'] = pd.Series(speeds_m_s) * 3.6  # km/h

    df['speed'] = df['speed'].ffill().fillna(0).clip(lower=0)
    print(f"[DEBUG] Vitesse recalculée : min={df['speed'].min():.2f} km/h, max={df['speed'].max():.2f} km/h, moyenne={df['speed'].mean():.2f} km/h")

    if "target_speed" in df.columns:
        print(f"[DEBUG] Présence de target_speed : min={df['target_speed'].min():.2f} km/h, max={df['target_speed'].max():.2f} km/h")
        config = load_full_config()
        # Correction des incohérences de hiérarchie des vitesses par type de route
        if "road_type" in df.columns and "target_speed" in df.columns:
            priority = {
                "motorway": 6,
                "primary": 5,
                "secondary": 4,
                "tertiary": 3,
                "residential": 2,
                "service": 1
            }
            grouped = df.groupby("road_type")["target_speed"].mean()
            sorted_types = sorted(priority.keys(), key=lambda k: priority[k])

            for i in range(len(sorted_types)-1):
                rt_lower = sorted_types[i]
                rt_higher = sorted_types[i+1]
                if rt_lower in grouped and rt_higher in grouped:
                    v_lower = grouped[rt_lower]
                    v_higher = grouped[rt_higher]
                    if v_lower > v_higher:
                        delta = v_lower - v_higher + 1.0
                        print(f"[ADJUST] Correction descendante : {rt_lower} ({v_lower:.2f}) > {rt_higher} ({v_higher:.2f}) → -{delta:.2f} km/h")
                        df.loc[df["road_type"] == rt_lower, "target_speed"] -= delta
                        grouped[rt_lower] -= delta
                    elif v_lower < v_higher - 20:  # tolérance excessive
                        delta = v_higher - v_lower - 10.0
                        print(f"[ADJUST] Correction ascendante : {rt_lower} ({v_lower:.2f}) << {rt_higher} ({v_higher:.2f}) → +{delta:.2f} km/h")
                        df.loc[df["road_type"] == rt_lower, "target_speed"] += delta
                        grouped[rt_lower] += delta
            # Deuxième passe stricte : réordonner les moyennes de vitesse selon la hiérarchie
            previous_v = None
            for rt in reversed(sorted_types):
                if rt in grouped:
                    if previous_v is not None and grouped[rt] > previous_v:
                        delta = grouped[rt] - previous_v + 1.0
                        print(f"[FORCE] Correction stricte : {rt} ({grouped[rt]:.2f}) > précédent ({previous_v:.2f}) → -{delta:.2f} km/h")
                        df.loc[df["road_type"] == rt, "target_speed"] -= delta
                        grouped[rt] -= delta
                    previous_v = grouped[rt]
            # Troisième passe : forcer la hiérarchie descendante finale
            final_grouped = df.groupby("road_type")["target_speed"].mean()
            for i in range(len(sorted_types)-1):
                rt_higher = sorted_types[i]
                rt_lower = sorted_types[i+1]
                if rt_higher in final_grouped and rt_lower in final_grouped:
                    if final_grouped[rt_lower] > final_grouped[rt_higher]:
                        delta = final_grouped[rt_lower] - final_grouped[rt_higher] + 1.0
                        print(f"[FORCE-FINAL] Correction : {rt_lower} ({final_grouped[rt_lower]:.2f}) > {rt_higher} ({final_grouped[rt_higher]:.2f}) → -{delta:.2f} km/h")
                        df.loc[df["road_type"] == rt_lower, "target_speed"] -= delta
                        final_grouped[rt_lower] -= delta
            # Quatrième passe stricte : forcer ordre décroissant
            final_grouped = df.groupby("road_type")["target_speed"].mean()
            last_value = None
            for rt in sorted_types:
                if rt in final_grouped:
                    v = final_grouped[rt]
                    if last_value is not None and v > last_value:
                        delta = v - last_value + 1.0
                        print(f"[FORCE-STRICT] {rt} > précédent ({last_value:.2f}) → -{delta:.2f} km/h")
                        df.loc[df["road_type"] == rt, "target_speed"] -= delta
                        final_grouped[rt] -= delta
                    last_value = final_grouped[rt]

            # Affichage des statistiques de vitesse par type de route en tranches de 10%
            print("[DEBUG] Vitesse par portion (10%) par type de route :")
            nb_parts = 10
            for i in range(nb_parts):
                start = int(i * len(df) / nb_parts)
                end = int((i + 1) * len(df) / nb_parts)
                sub_df = df.iloc[start:end]
                grouped = sub_df.groupby("road_type")["target_speed"].mean().sort_values(ascending=False)
                print(f" Portion {i+1}/{nb_parts} :")
                for rt, v in grouped.items():
                    print(f"   - {rt:12s} : {v:.2f} km/h")

        force = config.get("simulation", {}).get("force_target_speed", False)
        if force:
            from core.kinematics_speed import adjust_speed_progressively
            df = adjust_speed_progressively(df, config)
            return df
        df = interpolate_target_speed_progressively(df, iterations=iterations, alpha=alpha, window=10, force=force)

    df["speed"] = df["speed"].clip(upper=150)

    if "acc_x" in df.columns:
        df["acc_x"] = df["acc_x"].clip(lower=-20, upper=20)
    if "acc_y" in df.columns:
        df["acc_y"] = df["acc_y"].clip(lower=-20, upper=20)
    if "acc_z" in df.columns:
        df["acc_z"] = df["acc_z"].clip(lower=-30, upper=30)

    return df

def resample_trajectory_to_10hz(df, timestamp_col='timestamp', hz=10):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    # Ensure non-object dtypes before interpolate (avoids FutureWarning)
    df = df.infer_objects(copy=False)
    # Optionally coerce numeric columns to numbers (safe)
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    freq = f"{int(1000/hz)}ms"
    df_resampled = df.resample(freq).interpolate(method='linear')
    df_resampled = df_resampled.reset_index()
    return df_resampled


# Nouvelle fonction : reproject_trajectory_from_target_speed
def reproject_trajectory_from_target_speed(df, hz=10):
    """
    Reprojete les positions GPS pour que la vitesse suive la 'target_speed' fournie (en km/h),
    en maintenant un échantillonnage temporel constant à hz Hz.

    Args:
        df (pd.DataFrame): Doit contenir 'lat', 'lon', 'timestamp', 'target_speed'.
        hz (int): fréquence temporelle (ex : 10 Hz)

    Returns:
        pd.DataFrame: même structure avec nouvelles lat/lon suivant la vitesse cible.
    """
    from geopy.distance import distance as geopy_distance
    from geopy import Point
    import numpy as np

    dt = 1.0 / hz
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    # Lissage fort sur target_speed (~5 s)
    df["target_speed"] = smooth_signal(df["target_speed"], window=hz*5).clip(lower=0, upper=150)

    if "heading" not in df.columns:
        df = calculate_heading(df)

    lat, lon = df.loc[0, "lat"], df.loc[0, "lon"]
    new_lats = [lat]
    new_lons = [lon]

    for i in range(1, len(df)):
        if not np.isfinite(df.loc[i - 1, "heading"]) or not np.isfinite(df.loc[i, "target_speed"]):
            print(f"[ERROR] NaN détecté à i={i}: heading={df.loc[i - 1, 'heading']}, target_speed={df.loc[i, 'target_speed']}")
            # Reprendre la dernière position connue pour éviter de perdre un point
            new_lats.append(new_lats[-1])
            new_lons.append(new_lons[-1])
            continue
        heading = df.loc[i-1, "heading"]
        v_kmh = df.loc[i, "target_speed"]
        v_ms = v_kmh / 3.6
        dist_m = v_ms * dt

        origin = Point(lat, lon)
        destination = geopy_distance(meters=dist_m).destination(origin, heading)
        lat, lon = destination.latitude, destination.longitude

        new_lats.append(lat)
        new_lons.append(lon)

    df["lat"] = new_lats
    df["lon"] = new_lons
    return df


def compute_kinematic_metrics(df, hz=10, resample=True):
    """
    Calcule et enrichit la trajectoire avec les métriques cinématiques clés :
    - heading (cap en degrés)
    - curvature (courbure locale)
    - sinuosity (sinuosité sur fenêtre glissante)
    - recalcul de la vitesse
    - (optionnel) rééchantillonnage à fréquence fixe (ex : 10 Hz)

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes lat, lon, timestamp.
        hz (int): Fréquence cible pour le rééchantillonnage.
        resample (bool): Si True, rééchantillonne la trajectoire à hz Hz.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes kinematic_metrics.
    """
    if resample:
        df = resample_trajectory_to_10hz(df, hz=hz)
        # Si target_speed présent, reprojeter la trajectoire selon la vitesse cible
        if "target_speed" in df.columns:
            df = reproject_trajectory_from_target_speed(df, hz=hz)

    # Calcul du heading
    lat1 = df['lat'].values[:-1]
    lon1 = df['lon'].values[:-1]
    lat2 = df['lat'].values[1:]
    lon2 = df['lon'].values[1:]
    headings = compute_heading(lat1, lon1, lat2, lon2)
    headings = np.append(headings, np.nan)
    df['heading'] = pd.Series(headings).ffill().bfill()

    # Calcul de la courbure et sinuosité
    df['curvature'] = compute_curvature(df['heading'].values)
    df['sinuosity'] = compute_sinuosity(df['heading'].values, window=hz*5)  # 5 secondes glissantes

    # Recalcul vitesse (basé sur la distance et timestamps)
    # df = recompute_speed(df)  # Désactivé pour préserver les vitesses simulées (target_speed) après reprojection

    return df

def compute_cumulative_distance(df):
    distances = [0]
    for i in range(1, len(df)):
        dist = haversine_distance(df.loc[i-1, 'lat'], df.loc[i-1, 'lon'], df.loc[i, 'lat'], df.loc[i, 'lon'])
        distances.append(dist)
    cumdist = np.cumsum(distances)
    return pd.Series(cumdist)

def compute_total_distance(df):
    cumdist = compute_cumulative_distance(df)
    return cumdist.iloc[-1]


# Ajout : Fonction de visualisation des vitesses cibles par type de route pour chaque portion (10%)
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np


# If recompute_inertial_acceleration is needed here, import it from imu_utils:

# Import check_inertial_stats if needed by other modules
from core.imu_utils import check_inertial_stats


# Ajout : calcul de l'accélération linéaire à partir de la vitesse
def calculate_linear_acceleration(df, freq_hz=10):
    """
    Calcule l'accélération linéaire à partir des variations de la vitesse.

    Args:
        df (pd.DataFrame): Doit contenir une colonne 'speed' (en km/h).
        freq_hz (int): Fréquence d’échantillonnage (ex: 10 Hz).

    Returns:
        pd.DataFrame: Le DataFrame avec une nouvelle colonne 'acc_x' en m/s².
    """
    df = df.copy()
    speed_m_s = df["speed"] / 3.6  # conversion km/h -> m/s
    acc_x = np.gradient(speed_m_s) * freq_hz  # dérivée temporelle à freq_hz
    df["acc_x"] = acc_x.clip(-20, 20)
    return df


# Ajout : calcul de la vitesse angulaire (gyro_z) à partir du heading
def calculate_angular_velocity(df, freq_hz=10):
    """
    Calcule la vitesse angulaire (gyro_z) à partir des variations du heading (cap).

    Args:
        df (pd.DataFrame): Doit contenir une colonne 'heading' (en degrés).
        freq_hz (int): Fréquence d’échantillonnage (ex: 10 Hz).

    Returns:
        pd.DataFrame: Le DataFrame avec une nouvelle colonne 'gyro_z' en deg/s.
    """
    df = df.copy()
    heading_rad = np.radians(df["heading"])
    gyro_z = np.gradient(heading_rad) * freq_hz  # dérivée temporelle
    gyro_z_deg = np.degrees(gyro_z).clip(-180, 180)
    df["gyro_z"] = gyro_z_deg
    return df


# Nouvelle fonction : recompute_inertial_acceleration
def recompute_inertial_acceleration(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Recalcule acc_x et acc_y à partir de la vitesse (km/h) et du heading.
    - heading accepté en degrés ou radians (détection auto).
    - acc_z conservé si présent, sinon 9.81 par défaut.
    """
    out = df.copy()

    if "speed" not in out.columns or "heading" not in out.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'speed' et 'heading'.")

    # speed en m/s
    v = pd.to_numeric(out["speed"], errors="coerce").fillna(0.0).to_numpy() / 3.6

    # heading en radians (auto-détection si donné en degrés)
    heading_vals = pd.to_numeric(out["heading"], errors="coerce").fillna(0.0).to_numpy()
    if np.nanmax(np.abs(heading_vals)) > 2 * np.pi + 1e-6:
        theta = np.radians(heading_vals)
    else:
        theta = heading_vals

    dv = np.gradient(v) * float(hz)
    dtheta = np.gradient(theta) * float(hz)

    acc_x = dv * np.cos(theta) - v * np.sin(theta) * dtheta
    acc_y = dv * np.sin(theta) + v * np.cos(theta) * dtheta

    out["acc_x"] = acc_x
    out["acc_y"] = acc_y
    out["acc_z"] = pd.to_numeric(out.get("acc_z", 9.81)).fillna(9.81)

    return out


# Helper: enrich_inertial_coupling
def enrich_inertial_coupling(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Couche d'enrichissement inertiel « souple » pour garantir la cohérence minimale
    entre vitesse/heading et signaux IMU. Ne remplace rien d'existant si déjà présent.

    - (Re)calcule acc_x à partir de la dérivée de la vitesse si absente ou entièrement NaN.
    - (Re)calcule gyro_z à partir de la dérivée du heading si absent ou entièrement NaN.
    - Crée des colonnes manquantes avec des valeurs par défaut sûres.
      (acc_y=0, acc_z=9.81, gyro_x=0, gyro_y=0)
    """
    out = df.copy()

    # Recalcule acc_x si nécessaire (dérivée de la vitesse en m/s²)
    need_acc_x = ("acc_x" not in out.columns) or out["acc_x"].isna().all()
    if need_acc_x and "speed" in out.columns:
        try:
            out = calculate_linear_acceleration(out, freq_hz=hz)
        except Exception:
            # Valeur par défaut neutre si échec
            out["acc_x"] = 0.0

    # Recalcule gyro_z si nécessaire (dérivée du heading en deg/s)
    need_gyro_z = ("gyro_z" not in out.columns) or out["gyro_z"].isna().all()
    if need_gyro_z and "heading" in out.columns:
        try:
            out = calculate_angular_velocity(out, freq_hz=hz)
        except Exception:
            out["gyro_z"] = 0.0

    # Colonnes IMU manquantes → valeurs par défaut sûres
    if "acc_y" not in out.columns:
        out["acc_y"] = 0.0
    if "acc_z" not in out.columns:
        out["acc_z"] = 9.81
    if "gyro_x" not in out.columns:
        out["gyro_x"] = 0.0
    if "gyro_y" not in out.columns:
        out["gyro_y"] = 0.0

    # Remplir les NaN restants avec des valeurs neutres
    for col, default in [("acc_x", 0.0), ("acc_y", 0.0), ("acc_z", 9.81), ("gyro_x", 0.0), ("gyro_y", 0.0), ("gyro_z", 0.0)]:
        if col in out.columns:
            out[col] = out[col].fillna(default)

    return out