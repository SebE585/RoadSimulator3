# 📋 CHANGELOG.md – RoadSimulator3

---

## [v1.0] – 2025-08-07

### Ajouts
- Simulation gyroscopique complète (`gyro_x/y/z`) avec bruit inertiel MEMS
- Ajout de `generate_gyroscope_signals()` centralisée
- Ajout des colonnes gyro dans les exports CSV et JSON
- Visualisation gyro + événements dans `plot_utils.py`

### Nettoyage
- Suppression des fonctions redondantes et fichiers obsolètes
- Refactor des logs (passage à `logger.debug/info`)
- Nettoyage automatique via `scripts/clean_outputs.sh`

### Documentation
- Docstrings complétés
- Roadmap et plan de version mis à jour

---

## [v0.9] – 2025-07-25

- Architecture modulaire stabilisée
- Simulation inertielle à 10 Hz avec 10 événements injectables
- Ajout du bruit inertiel réaliste
- Visualisation HTML + PNG fonctionnelle
- Structure YAML complète pour configuration inertielle
