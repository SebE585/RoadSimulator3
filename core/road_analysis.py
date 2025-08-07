import numpy as np
import requests

OSRM_URL = "http://localhost:5001"

@deprecated
def get_osrm_roundabouts(coords):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Interroge OSRM pour détecter les ronds-points le long du parcours.

    Args:
        coords (list of tuple): liste de (lon, lat)

    Returns:
        list of dict: ronds-points détectés avec leur position et type.
    """
    MAX_POINTS = 50
    all_roundabouts = []
    for i in range(0, len(coords), MAX_POINTS):
        chunk = coords[i:i + MAX_POINTS]
        coords_str = ";".join(f"{lon},{lat}" for lon, lat in chunk)
        url = f"{OSRM_URL}/route/v1/driving/{coords_str}?steps=true&annotations=nodes"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        for route in data.get("routes", []):
            for leg in route.get("legs", []):
                for step in leg.get("steps", []):
                    maneuver = step.get("maneuver", {})
                    if maneuver.get("type") in ("roundabout", "rotary"):
                        all_roundabouts.append({
                            "location": maneuver.get("location"),
                            "type": maneuver.get("type"),
                        })
    return all_roundabouts

