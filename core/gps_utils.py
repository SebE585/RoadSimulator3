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

def compute_speed_from_gps(lat, lon, hz=10):
    """
    Calcule la vitesse en m/s à partir de lat/lon (en degrés) à une fréquence donnée.
    """
    R = 6371000  # Rayon de la Terre en mètres
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Décalage pour calcul entre points successifs
    lat1 = lat_rad[:-1]
    lat2 = lat_rad[1:]
    lon1 = lon_rad[:-1]
    lon2 = lon_rad[1:]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c  # distance entre points successifs

    speed = np.zeros_like(lat)
    speed[1:] = dist * hz
    speed[0] = speed[1]  # pour éviter artefact au début

    return speed