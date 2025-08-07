# 🛣️ RoadSimulator3 – Feuille de route (mise à jour)

## ✅ Version 0.9 – Base modulaire stable (pré-1.0)

**Inclut également les apports de la version 2.2 :**

- Colonnes `gyro_x`, `gyro_y`, `gyro_z` simulées.
- Simulation de l'ouverture de porte.
- Bruit inertiel réaliste (acc + gyro).
- Couplage acc/gyro partiellement intégré.

**Statut :** 100% – version gelée

- Architecture modulaire complète (`core/`, `simulator/`, `runner/`, etc.).
- Simulation inertielle à 10 Hz : injection + détection (10 événements).
- Gestion des `stop` et `wait`.
- Visualisation HTML + PNG avec légende interactive.
- Variation de vitesse selon typologie et sinuosité.
- Heatmap acc_x vs acc_y.
- Simulation de longs trajets (>400 km avec 15+ livraisons).
- Structure YAML centralisée, config inertielle robuste.
- Interfaces OSRM, OSMnx, SRTM locales fonctionnelles.

**Anciennes versions internes absorbées :** 1.0 → 1.6, 2.2

---

## ⏳ Version 1.0 – Livraison publique stable

**Statut :** en cours (finalisation)

### Objectifs techniques :

- Nettoyage des fichiers inutiles (`__pycache__`, `.DS_Store`, etc.).
- Suppression des répertoires obsolètes (`old/`, `deprecated/`, etc.).
- Centralisation des paramètres dans `config.yaml`.
- Unification des noms de scripts (`simulate_xxx.py`, `check_xxx.py`, etc.).
- Nettoyage des logs et sorties (`out/`, `logs/`, `outputs/`, etc.).
- Mise en place d’un `Makefile` ou script de gestion (`run_simulate.sh`, etc.).
- Ajout du script `clean_outputs.sh` pour nettoyage automatique.
- Ajout de `VERSION.md`, `CHANGELOG.md`, `README.md` enrichi.
- Gèle des dépendances (`requirements.txt` figé).
- Ajout de tests unitaires critiques (détection, injection, pipeline).
- Marquage Git `v1.0`.

---

## 🚧 Version 1.1 – Contextes géographiques et météo (en cours)

**Statut :** 60%

**Objectifs :**

- Enrichissement par zone (urbain, centre-ville, etc.) via grille H3.
- Ajout de la pente et du dévers (IGN BD ALTI / SRTM).
- Intégration des données météo (température, vent, précipitation…).
- Ajout du type de surface via OSM.
- Colonnes enrichies : `zone_type`, `h3_index`, `slope_percent`, `banking_angle`, `surface_type`, `meteo_*`
- Script dédié : `enrich_context.py`

---

## 🧠 Version 1.2 – Nouveaux indicateurs

**Objectifs :**

- Calcul de la pente cumulée.
- Vitesse moyenne en virage, moyenne acc_y.
- Puissance instantanée estimée (Watts).
- Temps d’approche des stops.
- Typologie de stationnement par zone.

---

## 🗺️ Version 1.3 – Harbers Boundaries

**Objectifs :**

- Segmentation avancée avec modèle Harbers.
- Détection contextuelle par grille H3.
- Résolution H3 à optimiser.

---

## 🌐 Version 2.0 – Application Web complète

**Objectifs :**

- Frontend React ou Vue.js.
- Backend Flask (API simulation/enrichissement).
- Authentification utilisateur.
- Visualisation interactive avec filtre, zoom, requêtes.

---

## 🐳 Version 2.1 – Dockerisation complète

**Statut :** 70% – reste autonome, **non intégrée à la v1.0**

**Objectifs atteints :**

- `docker-compose.yml` fonctionnel.
- Conteneurs OSRM, OSMnx-service, SRTM API opérationnels.
- API Flask dockerisée.

**Reste à finaliser :**

- Ajout d’un réseau Docker dédié (docker-compose).
- Nettoyage des volumes et logs (`down -v`, `clean_volumes.sh`).
- Intégration d’un `Makefile` ou `start_stack.sh`.
- Ajout d’un guide dans `README.md` ou `docs/docker.md`.
- Vérifications automatiques / tests d’intégration conteneurs.

> Cette version est désormais intégrée à la livraison stable `v1.0`.

---

## 🔧 Pistes futures

- Visualisations dynamiques supplémentaires (rejouer les trajets, slider temporel).
- Analyses limites de conduite (fatigue, agressivité, non-respect limitations).
- Simulation gyroscope perfectionnée (filtrage, signature étendue, erreurs MEMS).
- Couplage inertiel multivarié adaptatif (acc + gyro + contexte route).
- Export Parquet pour big data / cache Redis pour pipeline distribuée.
- Support du mode offline/online pour enrichissement différé, partiel ou dégradé.

---

**Dernière mise à jour : 2025-07-25**