# core/geo_utils.py
import numpy as np
import pandas as pd

def compute_heading(lat1, lon1, lat2, lon2):
    """
    Calcule le cap (heading) entre deux points GPS en degrés.
    """
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    heading = np.degrees(np.arctan2(y, x))
    return (heading + 360) % 360

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en mètres entre deux points GPS (lat/lon) avec la formule de Haversine.
    """
    R = 6371000  # Rayon de la Terre en mètres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_sinuosity(headings, window):
    """
    Calcule la sinuosity sur un vecteur numpy 'headings' avec une fenêtre entière.
    """
    n = len(headings)
    sinuosity_values = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)

        delta_heading = np.abs(headings[end-1] - headings[start])
        sinuosity_values[i] = delta_heading  # ou calcul plus complexe selon définition

    return sinuosity_values


def compute_curvature(headings):
    """
    Compute curvature from a 1D numpy array of headings (radians or degrees),
    sans essayer d'accéder à un DataFrame.
    """

    import numpy as np

    if isinstance(headings, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(headings):
            raise TypeError("La série 'headings' contient des timestamps au lieu d’angles.")
        headings = headings.astype(float)
    elif isinstance(headings, np.ndarray):
        if np.issubdtype(headings.dtype, np.datetime64):
            raise TypeError("Le tableau 'headings' contient des timestamps au lieu d’angles.")
        headings = headings.astype(float)
    else:
        raise TypeError("Entrée non prise en charge : headings doit être une série Pandas ou un array NumPy.")

    d_heading = np.diff(headings, n=2)  # dérivée seconde approximée
    d_heading = np.insert(d_heading, 0, [0, 0])  # pad pour garder la taille

    return d_heading

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000  # rayon de la Terre en mètres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def recompute_speed(df, config=None):
    """
    Recalcule la vitesse (km/h) à partir des positions GPS et timestamps, méthode vectorisée.
    """
    df = df.copy()
    lat1, lon1 = df["lat"].shift(), df["lon"].shift()
    lat2, lon2 = df["lat"], df["lon"]

    df["distance_m"] = haversine_np(lat1, lon1, lat2, lon2)
    df["delta_t"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    speed_m_s = df["distance_m"] / df["delta_t"].replace(0, np.nan)
    df["speed"] = speed_m_s.fillna(0) * 3.6  # m/s → km/h
    df["speed"] = df["speed"].clip(lower=0)

    # Si target_speed est présent, on peut ajouter un champ 'target_speed_diff'
    if "target_speed" in df.columns:
        if config and config.get("simulation", {}).get("force_target_speed", False):
            from core.kinematics_speed import interpolate_target_speed_progressively
            df = interpolate_target_speed_progressively(df)
        df["target_speed_diff"] = df["speed"] - df["target_speed"]
    return df