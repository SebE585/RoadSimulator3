"""
interpolation.py
Fonctions d’interpolation géodésique
"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import timedelta
import warnings

def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"Fonction obsolète : {func.__name__}", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper


@deprecated
@deprecated
def interpolate_route(route, frequency_hz=10):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Interpole un itinéraire sous forme de tuples (lat, lon) pour obtenir un point toutes les X ms.
    Ajoute la vitesse entre chaque point (en km/h) et un timestamp uniforme.
    """
    distance_per_step_m = 1.0  # interpolation à 1 mètre
    interpolated = []

    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]

        lat1, lon1 = start
        lat2, lon2 = end

        dist = geodesic((lat1, lon1), (lat2, lon2)).meters
        steps = max(1, int(dist // distance_per_step_m))

        lats = np.linspace(lat1, lat2, steps)
        lons = np.linspace(lon1, lon2, steps)

        for j in range(steps):
            interpolated.append((lats[j], lons[j]))

    # Création du DataFrame
    df = pd.DataFrame(interpolated, columns=["lat", "lon"])

    # Calcul de la vitesse entre les points (en km/h)
    speeds = [0]
    for i in range(1, len(df)):
        p1 = (df.at[i - 1, "lat"], df.at[i - 1, "lon"])
        p2 = (df.at[i, "lat"], df.at[i, "lon"])
        d_m = geodesic(p1, p2).meters
        dt_s = 1.0 / frequency_hz
        v_kmh = (d_m / dt_s) * 3.6
        speeds.append(v_kmh)
    df["speed"] = speeds

    # Ajout du timestamp (pas constant, basé sur fréquence)
    start_time = pd.Timestamp.now().normalize()
    dt_ms = int(1000 / frequency_hz)
    df["timestamp"] = pd.date_range(start=start_time, periods=len(df), freq=f"{dt_ms}ms")

    return df
