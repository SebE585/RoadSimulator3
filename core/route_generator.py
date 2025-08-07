"""
route_generator.py
Génération d’itinéraires via villes + OSRM
"""

import random
from core.osrm_utils import get_route_from_coords, interpolate_route_points

towns_around_romilly = [
    ("Romilly-sur-Andelle", 49.3653, 1.2361),
    ("Pont-Saint-Pierre", 49.3553, 1.1906),
    ("Pitres", 49.3364, 1.1733),
    ("Le Manoir", 49.3568, 1.2342),
    ("Les Damps", 49.3305, 1.1811),
    ("Val-de-Reuil", 49.2738, 1.2127),
    ("Louviers", 49.2164, 1.1711),
    ("Elbeuf", 49.2797, 1.0044),
    ("Le Neubourg", 49.1487, 0.8994),
    ("Rouen", 49.4431, 1.0993),
    ("Amfreville-la-Mi-Voie", 49.4002, 1.1228),
    ("Sotteville-lès-Rouen", 49.4083, 1.0999),
]

def generate_route_via_towns(n_points=6):
    selected = random.sample(towns_around_romilly, n_points)
    print("Villes sélectionnées :")
    for name, lat, lon in selected:
        print(f"{name}: {lat}, {lon}")
    coords = [(lat, lon) for _, lat, lon in selected]
    return coords

def simulate_route_from_towns(n_points=6):
    """
    Simule un itinéraire en sélectionnant n_points villes aléatoires parmi une liste prédéfinie.
    Retourne les points interpolés du trajet et la géométrie polyline.

    :param n_points: nombre de villes à inclure dans le trajet
    :return: tuple (interpolated_points, geometry)
    """
    coords = generate_route_via_towns(n_points)

    geometry, coordinates = get_route_from_coords(coords)
    print(f"Trajet OSRM obtenu : {len(coordinates)} points")

    interpolated_points = interpolate_route_points(coordinates, 0.83)
    print(f"Trajet interpolé : {len(interpolated_points)} points (10 Hz approx.)")

    return interpolated_points, geometry
