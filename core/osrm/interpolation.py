"""
interpolation.py – Interpolation géodésique entre points GPS.

Fournit une interpolation à pas fixe (e.g. 0.83 m) entre deux coordonnées
en utilisant la distance géodésique (via geopy).
"""

from typing import List, Tuple
from geopy.distance import geodesic


def interpolate_route_at_fixed_step(coords: List[Tuple[float, float]], step_m: float) -> List[Tuple[float, float]]:
    """
    Interpole une série de coordonnées GPS à intervalle fixe en mètres.

    Cette fonction divise chaque segment entre deux points successifs de la route
    en sous-segments équidistants d’environ `step_m` mètres, en utilisant une interpolation linéaire
    sur les coordonnées latitude/longitude.

    Args:
        coords (List[Tuple[float, float]]): Liste de tuples (lat, lon) représentant le trajet initial.
        step_m (float): Distance fixe entre deux points interpolés, en mètres.

    Returns:
        List[Tuple[float, float]]: Liste étendue de coordonnées interpolées.

    Raises:
        ValueError: Si la liste est vide, trop courte ou si `step_m` est invalide.
    """
    if not coords or len(coords) < 2:
        raise ValueError("Liste de coordonnées vide ou trop courte pour interpolation.")
    if step_m <= 0:
        raise ValueError("Le pas d’interpolation doit être strictement positif.")
    for lat, lon in coords:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Coordonnée invalide détectée : ({lat}, {lon})")

    new_coords = []
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        dist = geodesic(start, end).meters
        steps = max(int(dist / step_m), 1)
        for j in range(steps):
            lat = start[0] + (end[0] - start[0]) * j / steps
            lon = start[1] + (end[1] - start[1]) * j / steps
            new_coords.append((lat, lon))

    new_coords.append(coords[-1])
    return new_coords
