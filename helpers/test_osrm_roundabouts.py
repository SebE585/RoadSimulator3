import requests

# Coordonnées GPS des points demandés (lon, lat)
points = [
    (1.2361, 49.3653),   # Romilly-sur-Andelle
    (1.1906, 49.3553),   # Pont-Saint-Pierre
    (1.1733, 49.3364),   # Pitres
    (1.2342, 49.3568),   # Le Manoir
]

# Construire la chaîne de coordonnées pour OSRM (format lon,lat;lon,lat;...)
coords_str = ';'.join(f"{lon},{lat}" for lon, lat in points)

# URL OSRM locale
osrm_url = f"http://localhost:5001/route/v1/driving/{coords_str}?steps=true&annotations=nodes"

print("Requête OSRM :", osrm_url)

# Faire la requête
response = requests.get(osrm_url)
response.raise_for_status()
data = response.json()

# Afficher le code retour
print("Code retour :", data.get("code"))

# Parcourir les legs et steps pour détecter les rond-points (maneuver.type == 'roundabout' ou 'rotary')
print("Ronds-points détectés dans les étapes :")
for route_i, route in enumerate(data.get('routes', [])):
    print(f"Route {route_i}:")
    for leg_i, leg in enumerate(route.get('legs', [])):
        print(f"  Leg {leg_i}:")
        for step_i, step in enumerate(leg.get('steps', [])):
            maneuver = step.get('maneuver', {})
            mtype = maneuver.get('type', '')
            if mtype in ['roundabout', 'rotary']:
                location = maneuver.get('location', [])
                print(f"    Step {step_i}: type='{mtype}', location={location}")

