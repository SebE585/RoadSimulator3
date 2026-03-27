# RoadSimulator3 -- CHANGELOG

## v1.1.0 (2026-03-27)

### Architecture
- **Migration complete vers engine** : 26 stages modulaires, pipeline contract-based (rs3-contracts)
- Renommage du runner : `runner/simulate.py` (ancien `run_simulation2.py`)
- Suppression du runner legacy (`run_simulation.py`, `run_simulate.sh`)
- Suppression de `core/reports.py` (deprecie)
- Suppression de la doc mkdocs (doublon avec Teleforge)

### Simulation
- **DrivingDynamics** : distribution realiste d'acceleration par profil conducteur (Ornstein-Uhlenbeck)
- **DeviceRotator** : simulation d'orientation device (roll/pitch/yaw)
- **EventInjector** : 7 types d'evenements injectables
- **MultiRateSampler** : GPS 1Hz + IMU 10Hz dans un meme dataset
- **SpeedSmoother / SpeedSync** : lissage et synchronisation realistes
- **GPS Noise v2** : blackout tunnels, cold start drift, jitter correle HDOP

### Analyse
- **Bi-histogramme** (`engine/accel_stats.py`) : severity analysis distance-ponderee (Gx/Gy)
- Criteres de severite : std_gx, std_gy, percentile 0.1%, profils conducteur

### Web UI
- **Streamlit Web UI** : simulation interactive avec carte Folium
- Sections GPS / IMU / Rotation / Events configurables
- Onglets Carte / Acceleration / Gyroscope / QA
- Bouton "Analyser dans Nostos" (integration D0 partagee)
- Refonte complete v2.0 (312 lignes, etait 697)

### Export
- **Export D0 conforme RFC-0013** : d0.parquet + manifest.json + ground_truth.json
- Signal-only (pas d'altitude en D0, repoussee en D1 DEM)
- GPS noise + heading + hdop inclus

### Correctifs
- FutureWarning pandas 2.x (legs_stitch, events, noise)
- Schema nullable, validators gyro, legs_stitch duplicate index
- QA ternary expression leak, gz_variability skip sans gyro
- OSRM default port 5000

---

## v1.0.0 (2025-09-02)

### Release initiale
- Premiere release publique stable
- Architecture modulaire (`core/`, `simulator/`, `runner/`)
- Simulation inertielle 10 Hz : GPS + accelerometre + gyroscope
- 10 types d'evenements (freinage, acceleration, dos d'ane, nid de poule, virage, arret, attente, ouverture porte)
- Rapport HTML interactif (Plotly)
- Heatmap acceleration Gx/Gy
- Gestion stop/wait avec injection inertielle
- Integration OSRM, OSMnx, SRTM
- Configuration YAML centralisee
- Makefile, tests smoke, dependances figees
- Enrichissement altitude (`altitude_m`, `slope_percent`) via API SRTM
- Schema de dataset (`dataset_schema.yaml`) + validation
- Licences : Core AGPL-3.0, Docs CC-BY-SA, Exemples CC-BY
