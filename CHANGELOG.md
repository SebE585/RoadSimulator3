# RoadSimulator3 – CHANGELOG

## v1.0.0 (2025-09-02)
- Première release publique stable
- Dépendances figées, tests smoke, Makefile, clean script
- README/NOTICE/docs mis à jour
- Ajout des marqueurs début/fin de livraison (`in_delivery`, `delivery_state`)
- Projection des colonnes par catégories d’événements (`event_infra`, `event_behavior`, `event_context`)
- Enrichissement altitude (`altitude_m`, `slope_percent`, `altitude_smoothed`) via API SRTM
- Correctif gyro pour l’événement `acceleration_initiale`
- Enforce du schéma de dataset via `dataset_schema.yaml` et `tools/validate_dataset.py`
- Nouveaux outils :
  - `plot_utils.py` (visualisation quicklook)
  - `validate_dataset.py`
