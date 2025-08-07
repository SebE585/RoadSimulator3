import numpy as np
import pandas as pd
from core.kinematics import recompute_speed, smooth_signal
from simulator.events.noise import inject_inertial_noise
from core.geo_utils import compute_heading, compute_curvature, compute_sinuosity
from simulator.events.gyro import recompute_inertial_acceleration
from core.osmnx.mapping import HIGHWAY_TO_TYPE

def fill_heading(df):
    lat1 = df['lat'].values[:-1]
    lon1 = df['lon'].values[:-1]
    lat2 = df['lat'].values[1:]
    lon2 = df['lon'].values[1:]

    headings = compute_heading(lat1, lon1, lat2, lon2)
    headings = np.append(headings, np.nan)

    df['heading'] = headings
    df['heading'] = df['heading'].ffill()

    return df

def fill_curvature(df):
    heading_filled = df['heading'].ffill().values
    curvature = compute_curvature(heading_filled)
    df['curvature'] = curvature
    return df

def fill_sinuosity(df, window):
    heading_filled = df['heading'].ffill().values
    sinuosity = compute_sinuosity(heading_filled, window)
    df['sinuosity'] = sinuosity
    return df

def smooth_signal(series, window=5):
    """
    Applique une moyenne glissante pour lisser un signal.
    """
    return series.rolling(window=window, center=True, min_periods=1).mean()

def finalize_trajectory(df, hz=10, smoothing_window=3, config=None):
    """
    Pipeline complet de post-traitement :
    - recalcul de la vitesse (si nécessaire)
    - recalcul accélérations inertielles
    - injection d'un bruit inertiel réaliste
    - lissage des accélérations
    - calcul et remplissage des colonnes géométriques : heading, curvature, sinuosity
    - reconstruction du type de route (road_type) si nécessaire
    - vérification des NaN

    Args:
        df (pd.DataFrame): DataFrame contenant les données GPS/IMU.
        hz (int): Fréquence d'échantillonnage (Hz).
        smoothing_window (int): Taille de la fenêtre pour le lissage des signaux d'accélération.
        config (dict, optional): Configuration globale pour les paramètres de simulation.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les calculs et traitements.
    """

    if 'speed' not in df.columns:
        raise ValueError("[PIPELINE] La colonne 'speed' est absente. Veuillez appliquer 'apply_target_speed_by_road_type' ou 'recompute_speed' avant.")

    print("[PIPELINE] Recalcul des accélérations inertielles...")
    df = recompute_inertial_acceleration(df, hz=hz)

    # Interpolation progressive de la target_speed si demandé dans la config
    if 'target_speed' in df.columns and config and config.get("simulation", {}).get("force_target_speed", False):
        from core.kinematics_speed import interpolate_target_speed_progressively
        print("[PIPELINE] Interpolation progressive de la target_speed...")
        df = interpolate_target_speed_progressively(df, alpha=0.1, force=True)

    if config is not None and "simulation" in config and "inertial_noise_std" in config["simulation"]:
        std = config["simulation"]["inertial_noise_std"]
    else:
        std = smoothing_window / 100  # valeur par défaut

    print(f"[PIPELINE] Injection du bruit inertiel (std={std})...")
    noise_params = {
        "acc_std": std,
        "gyro_std": 0.15,
        "acc_bias": 0.0,
        "gyro_bias": 0.0,
    }
    df = inject_inertial_noise(df, noise_params)

    print(f"[PIPELINE] Lissage des accélérations (fenêtre={smoothing_window})...")
    df['acc_x'] = smooth_signal(df['acc_x'], window=smoothing_window)
    df['acc_y'] = smooth_signal(df['acc_y'], window=smoothing_window)
    df['acc_z'] = smooth_signal(df['acc_z'], window=smoothing_window)

    print("[PIPELINE] Calcul des grandeurs géométriques (heading, curvature, sinuosity)...")
    df = fill_heading(df)
    df = fill_curvature(df)
    df = fill_sinuosity(df, window=hz * 5)  # fenêtre de 5 secondes

    print("[PIPELINE] Vérification finale des NaN...")
    nan_summary = df.isna().sum()
    if nan_summary.sum() == 0:
        print("[PIPELINE] ✅ Pas de NaN détectés.")
    else:
        print(f"[PIPELINE] ⚠️ NaN détectés : {nan_summary.to_dict()}")

    return df