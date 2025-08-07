# 🧭 Module `simulator/osrm/` – Gestion des trajets via OSRM

Ce module regroupe l'ensemble des fonctions nécessaires pour :

- Construire un itinéraire réaliste à partir de points GPS.
- Interpoler le trajet à pas constant (~0.83 m).
- Éviter les boucles et incohérences géographiques.
- Simuler un trajet haute fréquence (10 Hz) en s’appuyant sur un serveur OSRM local.
- Ajouter des vérifications robustes sur les coordonnées fournies.
- Faciliter les tests grâce à des fonctions mockables.

---

## 📁 Structure du module

simulator/osrm/
│
├── client.py # Fonctions d'appel bas-niveau à l'API OSRM (route, etc.)
├── routing.py # Construction de trajets pseudo-aléatoires, détection de boucles, virages
├── interpolation.py # Interpolation à pas constant (distance géodésique)
└── simulate.py # Simulation globale (OSRM + interpolation + timestamp)


---

## 🧩 Description des fichiers

### `client.py`

- `get_route_from_coords(coords: List[Tuple[float, float]]) -> Tuple[dict, List[Tuple[float, float]]]` :
  - Appelle l’API `/route/v1/driving/` de OSRM.
  - Gère les erreurs de coordonnées (NaN, None, hors bornes).
  - Gère 3 tentatives avec backoff.
  - Retourne le GeoJSON et la liste [(lat, lon)].

- `decode_polyline(encoded_polyline)` :
  - Décode une polyline encodée en liste de points GPS.

---

### `routing.py`

- `generate_random_route_around(base_location, total_km, ...)` :
  - Génère une trajectoire pseudo-aléatoire autour d’un point.
  - Rejette les boucles via `LineString.is_simple`.
  - Vérifie les coordonnées.

- `_route_has_loop(route, threshold_m)` :
  - Détection personnalisée de boucles via Haversine.

- `get_osrm_turns(...)` + `validate_turns(...)` :
  - Détection de virages par changement de heading (delta_heading).
  - Validation par `acc_y`.

---

### `interpolation.py`

- `interpolate_route_at_fixed_step(coords, step_m)` :
  - Interpole chaque segment à une distance fixe (~0.83 m).
  - Utilise `geopy.distance.geodesic`.
  - Vérifie que le pas et les coordonnées sont valides.

---

### `simulate.py`

- `simulate_route_via_osrm(cities_coords, hz=10, step_m=0.83)` :
  - Enchaîne : appel OSRM + interpolation + timestamps 10 Hz.
  - Vérifie la validité des coordonnées avant de simuler.
  - Retourne un `pd.DataFrame` avec `lat`, `lon`, `timestamp`.

---

## ⚙️ Dépendances

- OSRM doit être lancé en local sur `http://localhost:5003`
- Requiert les packages :
  - `requests`, `geopy`, `shapely`, `pandas`, `numpy`, `polyline`

---

## 🧪 Tests intégrés

- ✅ **Unitaires mockés** :
  - `get_route_from_coords()` avec `unittest.mock` (succès et erreurs).
  - `simulate_route_via_osrm()` avec réponse OSRM simulée.
  - `generate_random_route_around()` : géométrie simple, pas de boucle.
  - `interpolate_route_at_fixed_step()` : segments courts avec vérification du nombre de points.

- ✅ **Tests d’intégration** :
  - Possible avec un serveur OSRM actif (via Docker).
  - Exemple : `curl http://localhost:5003/route/v1/driving/...`

---

## ✅ Vérifications ajoutées (robustesse)

- Coordonnées vérifiées (NaN, None, bornes géographiques).
- Pas d’interpolation négatif.
- Gestion des erreurs réseau (OSRM) avec message explicite.
- Docstrings sur toutes les fonctions publiques.
- Code typé et testé.

---

## 📌 Exemple d’utilisation

```python
from simulator.osrm.simulate import simulate_route_via_osrm

coords = [(49.4431, 1.0993), (49.4944, 0.1079)]
df = simulate_route_via_osrm(coords, hz=10, step_m=0.83)
print(df.head())

📝 TODOs à suivre

Ajouter des loggers dans routing.py et simulate.py
Refactorer _route_has_loop() pour une version vectorisée rapide
Ajouter un cache local pour éviter les appels OSRM redondants
Ajouter un fallback polyline + routing simple si OSRM n’est pas dispo
