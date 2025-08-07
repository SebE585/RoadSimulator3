"""
client.py – Appels directs à l'API OSRM locale.

Contient les fonctions de requêtes HTTP à l'API OSRM `/route/v1/driving`,
avec gestion d’erreurs, tentatives multiples, et décodage de géométrie.
"""

import requests
import time
from typing import List, Tuple

# URL de l’instance OSRM locale (adapter si nécessaire)
OSRM_URL = "http://localhost:5003"

def get_route_from_coords(coords: List[Tuple[float, float]]) -> Tuple[dict, List[Tuple[float, float]]]:
    """
    Appelle l'API OSRM pour calculer un itinéraire entre les coordonnées données.

    Args:
        coords (List[Tuple[float, float]]): Liste de points GPS (lat, lon) décrivant le trajet.

    Returns:
        Tuple[dict, List[Tuple[float, float]]]:
            - Le dictionnaire GeoJSON brut de la géométrie.
            - La liste des coordonnées décodées [(lat, lon), ...].

    Raises:
        ValueError: Si les coordonnées sont invalides ou si la réponse OSRM est vide.
        RuntimeError: Après 3 échecs de requête OSRM.
    """
    if not coords or any(c is None for c in coords):
        raise ValueError("[OSRM] Coordonnées vides ou contenant des None.")
    for lat, lon in coords:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"[OSRM] Coordonnée invalide : ({lat}, {lon})")

    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{OSRM_URL}/route/v1/driving/{coords_str}?overview=full&geometries=geojson"

    for attempt in range(3):
        try:
            response = requests.get(url)
            data = response.json()
            if "routes" in data and len(data["routes"]) > 0:
                geometry = data["routes"][0]["geometry"]
                coordinates = [(lat, lon) for lon, lat in geometry["coordinates"]]
                return geometry, coordinates
            raise ValueError("[OSRM] Aucune route retournée par le serveur.")
        except Exception as e:
            print(f"[OSRM] Erreur (tentative {attempt+1}/3): {e}")
            time.sleep(1)
    raise RuntimeError("[OSRM] Échec de la requête OSRM après 3 tentatives.")


def decode_polyline(encoded_polyline: str) -> List[Tuple[float, float]]:
    """
    Décode une chaîne de polyline encodée en une liste de points (lat, lon).

    Args:
        encoded_polyline (str): Chaîne polyline encodée.

    Returns:
        List[Tuple[float, float]]: Liste de points décodés.
    """
    import polyline
    return polyline.decode(encoded_polyline)
