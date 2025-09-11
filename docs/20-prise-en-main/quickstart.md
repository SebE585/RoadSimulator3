# Quickstart

Cette page décrit comment exécuter rapidement une simulation RS3.

---

## Étape 1 — Lancer les services
```bash
docker compose up -d
```
Services lancés :
- **OSRM** : moteur de routage
- **OSMnx-service** : enrichissement routier
- **SRTM-service** (optionnel, plugin altitude)

---

## Étape 2 — Simulation simple
```bash
make simulate
```

👉 Produit un dossier :  
`data/simulations/simulated_YYYYMMDD_HHMMSS/`

Contenant :
- `output_osrm_trajectory.csv`
- `output_report.html`
- `output_map.png`

---

## Étape 3 — Vérifier les résultats
- Ouvrez le fichier `output_report.html` dans votre navigateur.  
- Explorez les colonnes du CSV (`timestamp`, `lat`, `lon`, `acc_*`, `gyro_*`).

---

## Étape 4 — Options avancées
Exécuter le runner directement :
```bash
python -m runner.run_simulation       --config config/simulation.yaml       --hz 20       --count 500
```
