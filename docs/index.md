# RoadSimulator3

Bienvenue dans la documentation officielle de **RoadSimulator3 (RS3)**.

RS3 est un simulateur inertiel haute fréquence (**10 Hz**) permettant de générer des trajectoires réalistes enrichies de signaux accéléromètre et gyroscope.  
Il est conçu pour la recherche et l’expérimentation sur la logistique du **dernier kilomètre**, en intégrant :
- des modèles inertiels réalistes (acc, gyro),
- des événements contextuels (arrêts, livraisons, virages),
- une architecture modulaire extensible par **plugins**.

---

## 📦 Prérequis rapides
- Python ≥ 3.11
- Docker + Docker Compose
- GNU Make

---

## 🚀 Démarrage rapide
```bash
git clone https://github.com/SebE585/RoadSimulator3.git
cd RoadSimulator3
docker compose up -d       # démarre OSRM/OSMnx/SRTM
make simulate              # lance une simulation simple
```

👉 Résultats dans :  
`data/simulations/simulated_YYYYMMDD_HHMMSS/`

---

## 📊 Exemple de sortie
- `output_osrm_trajectory.csv` (trajectoire brute 10 Hz)
- `output_report.html` (rapport interactif)
- `output_map.png` (visualisation statique)

---
