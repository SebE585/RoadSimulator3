# 🚗 RoadSimulator3

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sebastien_Edet-blue)](https://www.linkedin.com/in/sebastienedet/)
[![AGPL License](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CC-BY-SA License](https://img.shields.io/badge/License-CC--BY--SA-green.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![CC-BY License](https://img.shields.io/badge/License-CC--BY-green.svg)](https://creativecommons.org/licenses/by/4.0/)

RoadSimulator3 (RS3) est un **simulateur inertiel haute fréquence (10 Hz)** qui génère des trajectoires réalistes à partir d’OpenStreetMap (OSRM/OSMnx) et produit des signaux **accéléromètre + gyroscope**, des **événements** (arrêts, virages, chocs…) et des **rapports** interactifs HTML.

---

## 📦 Prérequis

- Python **3.11+**
- `make`
- **Docker** + **Docker Compose**

### Services utilisés
- **OSRM** : routage à partir d’OSM
- **OSMnx-service** : typologie de route (voie, type, classes…)
- **SRTM** *(via plugin Altitude)* : altimétrie (élévation) — requis uniquement si le plugin **rs3-plugin-altitude** est activé

> ⚠️ **Important** : Les images Docker **n’incluent pas** d’extraits OSM/SRTM. Il faut **télécharger et préparer** les données **avant** de lancer la stack.

---

## 🗺️ Préparer les données (obligatoire)

### 1) OSM / OSRM (extrait régional)

1. **Télécharger** un extrait `.osm.pbf` couvrant votre zone (ex.: depuis Geofabrik ou un miroir régional) et placez-le dans `data/osm/`.
2. **Construire** les fichiers OSRM (profil voiture, pipeline CH par défaut) avec l’image officielle `osrm/osrm-backend` :

```bash
# Créer l’arborescence locale
mkdir -p data/osm

# Exemple avec un extrait nommé france-normandie-latest.osm.pbf
export OSM_PBF=data/osm/france-normandie-latest.osm.pbf

# Extraction (profil voiture)
docker run --rm -t -v "$PWD/data/osm:/data" osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/$(basename "$OSM_PBF")

# Contraction (CH) — simple et rapide pour servir via osrm-routed
docker run --rm -t -v "$PWD/data/osm:/data" osrm/osrm-backend \
  osrm-contract /data/$(basename "$OSM_PBF" .osm.pbf).osrm

# Vous devez obtenir des fichiers /data/osm/*.osrm*
ls -lh data/osm | grep .osrm
```

> 💡 Alternative MLD (grands graphes, profils perso) : remplacer `osrm-contract` par `osrm-partition && osrm-customize` et configurez le service en conséquence.

### 2) SRTM (élévation) — *uniquement si le plugin Altitude est utilisé*

Cette étape est **optionnelle** et requise seulement si vous avez installé/activé le plugin **rs3-plugin-altitude** (AGPL). Si vous n'utilisez pas ce plugin, vous pouvez ignorer cette section.

1. **Télécharger** les tuiles **SRTM 1 Arc-Second** (ou équivalent) couvrant votre zone (ex.: NASA, ViewfinderPanoramas) au format `.hgt` ou `.hgt.zip`.
2. Placez-les dans `data/srtm/` puis **décompressez** si nécessaire :

```bash
mkdir -p data/srtm
# Copiez/telechargez vos tuiles ici, puis :
find data/srtm -name "*.zip" -exec unzip -o {} -d data/srtm \;
```

> ℹ️ Le service SRTM lira directement les fichiers `.hgt` présents dans `data/srtm`.

### 3) OSMnx-service

Aucune donnée à préparer : le service consomme l’extrait OSRM pour l’itinéraire et interroge OSMnx / cache interne pour enrichir les types de voies.

---

## ⚡ Démarrage rapide (Quickstart)

```bash
git clone https://github.com/SebE585/RoadSimulator3.git
cd RoadSimulator3

# 1) Préparer les données (obligatoire)
#   - Téléchargez un .osm.pbf (voir ci-dessus) et construisez les fichiers .osrm
#   - Téléchargez les tuiles SRTM dans data/srtm

# 2) Lancer la stack de services
docker compose up -d

# 3) Lancer une simulation
make simulate
```

---

## 🚀 Lancer une simulation

Simulation standard :

```bash
make simulate
```

Selon la configuration, le pipeline exécute :
1. Itinéraire via OSRM
2. Interpolation à **10 Hz**
3. Injection d’événements (arrêts, virages, chocs…)
4. Bruit réaliste (acc + gyro)
5. Validation du dataset (schéma & cohérence spatio-temporelle)
6. Exports dans `data/simulations/<horodatage>/` :
   - `output_osrm_trajectory.csv`
   - `report.html` (rapport interactif)
   - `map.html` (carte du trajet)

---

## 🧪 Exemples supplémentaires

- **Simulation de flotte** (si disponible) :

```bash
make fleet PROFILE=parcels VEHICLES=VL-01,VL-02
```

- **Validation dataset** :

```bash
python tools/validate_dataset.py --csv data/simulations/last_trace.csv --strict-order
```

---

## 📑 Schéma du dataset (RS3 v1.0)

| Colonne         | Type      | Détails                                 |
|-----------------|-----------|-----------------------------------------|
| timestamp       | datetime  | ISO 8601 (10 Hz)                        |
| lat             | float     | Latitude (WGS84)                        |
| lon             | float     | Longitude (WGS84)                       |
| altitude_m      | float     | Altitude (m)                            |
| speed           | float     | Vitesse (m/s)                           |
| acc_x           | float     | Accélération X (m/s²)                   |
| acc_y           | float     | Accélération Y (m/s²)                   |
| acc_z           | float     | Accélération Z (m/s²)                   |
| gyro_x          | float     | Gyroscope X (rad/s)                     |
| gyro_y          | float     | Gyroscope Y (rad/s)                     |
| gyro_z          | float     | Gyroscope Z (rad/s)                     |
| in_delivery     | bool      | Véhicule en phase de livraison          |
| delivery_state  | category  | État de livraison                       |
| event           | category  | Type d’événement                        |
| event_infra     | category  | Événement lié à l’infrastructure        |
| event_behavior  | category  | Événement lié au comportement           |
| event_context   | category  | Événement contextuel                     |

---

## 🗺️ Architecture (résumé)

- **Core** : logique de simulation et pipeline
- **Plugins** : enrichissements (capteurs, exporteurs, métriques…)
- **Services** : OSRM / OSMnx-service / SRTM (via Docker)
- **Outputs** : CSV, HTML, cartes; dossiers `data/simulations/...`

---

## 🗓️ Feuille de route

- Intégration de données temps réel
- Indicateurs avancés (énergie, pente cumulée, style de conduite)
- Export Parquet, cache Redis
- Application Web (API + UI) 

---

## 🔐 Licences

| Composant              | Licence       |
|------------------------|---------------|
| Noyau RS3 (core)       | AGPL-3.0      |
| Documentation          | CC-BY-SA 4.0  |
| Exemples & tutoriels   | CC-BY 4.0     |
| Plugins pro éventuels  | Propriétaire  |

---

## 🤝 Contribution

Les contributions sont bienvenues ! Voir `CONTRIBUTING.md`.

## 📬 Contact

**Sébastien EDET** — sebastien.edet@gmail.com
