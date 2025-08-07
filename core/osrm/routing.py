"""
routing.py – Génération et validation de trajectoires.

Gère la génération de trajets pseudo-aléatoires, la détection de boucles,
le calcul de heading, et la détection de virages (delta heading).
"""

import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .client import get_route_from_coords


def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calcule la distance en kilomètres entre deux points géographiques.

    Args:
        coord1 (Tuple[float, float]): (lat, lon) du premier point.
        coord2 (Tuple[float, float]): (lat, lon) du second point.

    Returns:
        float: Distance en kilomètres.
    """
    R = 6371.0  # Rayon terrestre en km
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


@deprecated
def _route_has_loop(route: List[Tuple[float, float]], threshold_m: float = 30.0) -> bool:
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Détecte une boucle dans une liste de points GPS si deux points sont trop proches.

    Args:
        route (List[Tuple[float, float]]): Liste de coordonnées (lat, lon).
        threshold_m (float): Seuil en mètres sous lequel deux points sont considérés comme un retour.

    Returns:
        bool: True si boucle détectée, sinon False.
    """
    coords = np.array(route)
    for i in range(len(coords)):
        for j in range(i + 10, len(coords)):  # Sauter les proches voisins
            if haversine(tuple(coords[i]), tuple(coords[j])) * 1000 < threshold_m:
                return True
    return False


@deprecated
def generate_random_route_around(
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    base_location: Tuple[float, float],
    total_km: float,
    max_attempts: int = 20,
    max_step_km: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Génère un itinéraire aléatoire sans boucle autour d'un point de départ.

    Args:
        base_location (Tuple[float, float]): Coordonnée (lat, lon) de départ.
        total_km (float): Distance cible totale de la route.
        max_attempts (int): Nombre maximal d'essais.
        max_step_km (float): Distance maximale d'un segment.

    Returns:
        List[Tuple[float, float]]: Liste de coordonnées (lat, lon).

    Raises:
        ValueError: Si la base est invalide.
        RuntimeError: Si aucune route simple n’a pu être générée.
    """
    if base_location is None or len(base_location) != 2:
        raise ValueError("Base location invalide.")
    lat, lon = base_location
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Coordonnée de départ invalide : {base_location}")

    for attempt in range(1, max_attempts + 1):
        route_points = [base_location]
        current_distance = 0.0

        while current_distance < total_km:
            last = route_points[-1]
            angle = random.uniform(0, 360)
            step_distance = random.uniform(0.5, max_step_km)

            delta_lat = step_distance / 111.0 * math.cos(math.radians(angle))
            delta_lon = step_distance / (111.0 * math.cos(math.radians(last[0]))) * math.sin(math.radians(angle))

            next_lat = last[0] + delta_lat
            next_lon = last[1] + delta_lon
            next_point = (round(next_lat, 6), round(next_lon, 6))

            if next_point == last:
                continue

            segment_distance = haversine(last, next_point)
            current_distance += segment_distance
            route_points.append(next_point)

        line = LineString([(lon, lat) for lat, lon in route_points])
        if line.is_simple:
            return route_points
        else:
            print(f"[Tentative {attempt}] ❌ Route avec boucle détectée. Recommence...")

    raise RuntimeError(f"Échec de génération sans boucle après {max_attempts} tentatives.")


def compute_heading(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcule l'azimut (heading) entre deux points GPS.

    Args:
        p1, p2 (Tuple[float, float]): Coordonnées (lat, lon).

    Returns:
        float: Angle en degrés [0, 360).
    """
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def delta_heading(h1: float, h2: float) -> float:
    """
    Calcule la différence d’angle minimale entre deux orientations.

    Args:
        h1 (float): Premier angle en degrés.
        h2 (float): Deuxième angle en degrés.

    Returns:
        float: Différence minimale entre les deux angles.
    """
    diff = abs(h2 - h1)
    return min(diff, 360 - diff)


def get_osrm_turns(coords: List[Tuple[float, float]], angle_threshold: float = 30.0) -> List[dict]:
    """
    Détecte les virages dans une trajectoire GPS via variation de heading.

    Args:
        coords (List[Tuple[float, float]]): Coordonnées interpolées (lat, lon).
        angle_threshold (float): Seuil en degrés pour considérer un virage.

    Returns:
        List[dict]: Liste des virages détectés (index, delta_heading, location).
    """
    turns = []
    for i in range(1, len(coords) - 1):
        h1 = compute_heading(coords[i - 1], coords[i])
        h2 = compute_heading(coords[i], coords[i + 1])
        delta = delta_heading(h1, h2)
        if delta > angle_threshold:
            turns.append({
                "index": i,
                "delta_heading": delta,
                "location": coords[i]
            })
    return turns


def validate_turns(df: pd.DataFrame, indices: List[int], threshold: float = 0.5) -> bool:
    """
    Vérifie si les virages détectés ont une accélération latérale (acc_y) significative.

    Args:
        df (pd.DataFrame): DataFrame avec colonne acc_y.
        indices (List[int]): Index des virages à vérifier.
        threshold (float): Seuil minimal en m/s² pour considérer comme valide.

    Returns:
        bool: True si au moins un virage présente une acc_y > threshold.
    """
    if "acc_y" not in df.columns:
        return False
    return any(abs(df.loc[i, "acc_y"]) > threshold for i in indices if i in df.index)
