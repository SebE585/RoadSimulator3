# core/gps_utils.py
import numpy as np

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