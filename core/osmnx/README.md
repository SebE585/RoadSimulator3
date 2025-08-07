# 🌐 Module `core/osmnx/` – Enrichissement typologique via OSM

Ce module fournit des fonctions d’enrichissement de trajectoires GPS avec les **types de routes (`road_type`)** issus d’OpenStreetMap, via une **API SSE locale** robuste.

---

## 📁 Structure

```plaintext
core/osmnx/
├── client.py                # Enrichissement SSE (streaming via Flask)
├── mapping.py               # Mapping highway OSM vers road_type harmonisé
├── test_osmnx.py            # Tests d'intégration (avec API active)
├── test_osmnx_mocked.py     # Tests unitaires simulés (mock API)
```

---

## ⚙️ Fonctionnalités principales

- Enrichit un `DataFrame` contenant les colonnes `lat`, `lon` avec :
  - `road_type` : type routier harmonisé (`primary`, `residential`, etc.)
  - `osm_highway` : valeur brute du tag OSM `highway=*`

- Fonctionne via un **serveur SSE local** exposé sur  
  `http://localhost:5002/nearest_road_batch_stream/`.

- Robuste aux erreurs réseau, à l’incomplétude et aux formats inattendus.

- Utilise `get_edge_type_nearest()` pour garantir une classification exploitable.

---

## 🚀 Exemple d’utilisation

```python
import pandas as pd
from core.osmnx.client import enrich_road_type_stream

df = pd.DataFrame({
    "lat": [49.4431, 49.4944],
    "lon": [1.0993, 0.1079]
})

df = enrich_road_type_stream(df)
print(df[["lat", "lon", "road_type", "osm_highway"]])
```

---

## ✅ Résultat attendu

| lat     | lon     | road_type   | osm_highway |
|---------|---------|-------------|-------------|
| 49.4431 | 1.0993  | residential | residential |
| 49.4944 | 0.1079  | unknown     | unknown     |

---

## 🧪 Tests

Tests complets avec serveur SSE actif :

```bash
pytest core/osmnx/test_osmnx.py -s -v
```

Tests unitaires sans serveur (mocked) :

```bash
pytest core/osmnx/test_osmnx_mocked.py -s -v
```

---

## 🔌 Dépendances

- `requests`
- `sseclient`
- `pandas`
- `tqdm`
- `pytest`

---

## 🛠️ Serveur SSE requis

Le service Flask doit être lancé localement :

```bash
docker compose up osmnx-service
```

---


---

## 🔄 Exemple de conversion depuis un fichier .osm PBF

```bash
# Étape 1 : filtrer les routes depuis le .pbf
osmium tags-filter haute-normandie-latest.osm.pbf w/highway -o roads-only.osm

# Étape 2 : charger le XML filtré dans un graphe OSMnx
python scripts/build_graph_from_xml.py
```

✅ Ce script applique une bidouille pour contourner un bug de `osmnx.utils_graph.remove_isolated_nodes` lié à un `warn` devenu booléen.


## 🗺️ Mapping OSM → Typologie

Le mapping utilisé (voir `mapping.py`) :

```python
HIGHWAY_TO_TYPE = {
    "motorway": "motorway",
    "trunk": "primary",
    "primary": "primary",
    "secondary": "secondary",
    "tertiary": "tertiary",
    "residential": "residential",
    "service": "residential",
    "unclassified": "unclassified",
    "track": "other",
    "path": "other",
    "cycleway": "other",
    ...
}
```

---

## ⚠️ Astuce : Bug `TypeError: 'bool' object is not callable`

Un bug fréquent lors de l’utilisation de `ox.utils_graph.remove_isolated_nodes(G)` est lié à une variable nommée `warn` qui écrase la fonction `warnings.warn`.

**Solution recommandée** : ajouter en haut de vos scripts si vous surchargez `warn` :

```python
import warnings
del warn  # si elle existe
```

Ou remplacez par :

```python
from warnings import warn as _warn
_warn("message", FutureWarning)
```

Ce contournement est nécessaire si vous manipulez manuellement le code de OSMnx ou vos scripts customisés.

---

## 🧭 Roadmap

- [ ] Permettre un fallback local (sans serveur SSE)
- [ ] Ajout de logs géolocalisés des erreurs
- [ ] Configuration du mapping `highway → road_type` via YAML
- [ ] Fallback simplifié basé sur GraphML local (hors réseau)
