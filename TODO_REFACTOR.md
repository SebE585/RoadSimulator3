# ✅ TODO – Nettoyage & Refactor final RoadSimulator3 v1.0

**Date de génération : 2025-08-07 16:05**

---

## ✅ Version v0.9-alpha validée

✅ Structure modulaire fonctionnelle  
✅ Simulation inertielle à 10 Hz + événements  
✅ Fichiers inutiles identifiés  
✅ Refactor `generation.py` / `detection.py` effectué  
✅ Injection inertielle centralisée & réaliste  
✅ Tag prêt à être posé : `v0.9-alpha`

---

## 🧹 1. Nettoyage fichiers / dossiers

- [ ] Supprimer `__pycache__/`, `.DS_Store`, `.ipynb_checkpoints/`
- [ ] Supprimer répertoires `old/`, `deprecated/`, `tmp/`, `sandbox/`
- [ ] Supprimer fichiers logs bruts, outputs orphelins
- [ ] Ajouter un script `scripts/clean_outputs.sh` pour automatiser

---

## 🪵 2. Nettoyage logs et debug

- [ ] Supprimer tous les `print()` inutiles
- [ ] Remplacer les `print()` utiles par des appels au logger (`logger.debug/info`)
- [ ] Vérifier qu’il y a un logger par module (`__name__`)

---

## 🧠 3. Suppression ou fusion de fonctions inutiles

- [x] Revoir `simulator/events/generation.py` pour identifier les fonctions redondantes
- [ ] Revoir `core/gyroscope.py` et fusionner dans `generate_gyroscope_signals()`
- [x] Supprimer ou marquer `@deprecated` les fonctions non utilisées
  - [x] Supprimer `detect_spatio_temporal_anomalies()`, `check_post_simulation()` et `get_log_path` dans `check/check_realism.py`
  - [x] Supprimer ou marquer `@deprecated` les fonctions non utilisées détectées par `vulture` :
    → Liste générée automatiquement depuis `logs/vulture_unused.txt`
    - [ ] `check/check_realism.py`: `detect_spatio_temporal_anomalies`
    - [ ] `core/config_loader.py`: `get_context_config`, `get_context_sources`, `resolve_mnt_path`
    - [ ] `core/h3_zone_mapper.py`: `enrich_with_h3_and_zones`
    - [ ] `core/interpolation.py`: `interpolate_route`
    - [ ] `core/kinematics.py`: `check_speed_plateaux`, `plot_target_speed_by_road_type_per_portion`, `detect_speed_plateaux`
    - [ ] `core/meteo_loader.py`: `load_meteo_data`, `enrich_with_meteo`, `enrich_with_meteo_dummy`
    - [ ] `core/osm_zone_loader.py`: `load_osm_zones`, `enrich_with_osm_zones`
    - [ ] `core/osmnx/enrich.py`: `enrich_road_type`
    - [ ] `core/osrm/client.py`: `decode_polyline`
    - [ ] `core/osrm/routing.py`: `_route_has_loop`, `generate_random_route_around`
    - [ ] `core/osrm/simulate.py`: `simulate_route_via_osrm_from_events`
    - [ ] `core/pbf_utils.py`: `enrich_road_type_pbf`
    - [ ] `core/reprojection.py`: `total_dist`
    - [ ] `core/road_analysis.py`: `get_osrm_roundabouts`
    - [ ] `core/terrain/enrich.py`: `enrich_terrain`
    - [ ] `core/utils.py`: `find_latest_trace`, `get_log_path`
    - [ ] `core/validation.py`: `is_regular_sampling`
    - [ ] `core/visualization.py`: `plot_speed`
    - [ ] `simulator/events/__init__.py`: `inject_inertial_events`
    - [ ] `simulator/events/generation.py`: `mark_delivery_points`, `generate_stop`, `generate_wait`
    - [ ] `simulator/events/neutral.py`: `inject_neutral_phases`, `speed_mps`
    - [ ] `simulator/events/roundabouts.py`: `inject_inertial_signature_for_turns` (x2), `generate_inertial_signature_for_osrm_roundabouts`
    - [ ] `simulator/events/stops_and_waits.py`: `inject_stops_and_waits`
    - [ ] `simulator/events/tracker.py`: `reset`, `as_dict`
    - [ ] `simulator/pipeline/pipeline.py`: `cap_global_speed_delta`, `enrich_inertial_coupling`
    - [ ] `simulator/vizualisation/map_renderer.py`: `evt_type`, `EC`
    - [ ] `tests/test_events_detection.py`: `detect_acceleration_test`
    - [ ] `tests/test_kinematics.py`: `clean_and_recompute`
    - [ ] `helpers/enhance_inertial_lat.py`: `compare_roundabout_detections`
    - [ ] `helpers/explore_turn_detection_params.py`: `circular_smooth_heading`
    - [ ] `tools/generate_graphml.py`: `way`
    - [ ] `tools/ign/merge_rgealti_tifs.py`: `dirnames`
    - [ ] `services/osmnx-service/app.py`: `start_stream`, `stream_results`
    - [ ] `services/srtm-service/app.py`: `enrich_terrain_api`
  🔗 Voir aussi : `logs/vulture_unused.txt`
  - [ ] Se baser sur l'analyse statique `logs/vulture_unused.txt` pour automatiser l'identification

  - [ ] (optionnel) Créer un script `scripts/mark_unused.py` pour insérer automatiquement `@deprecated` + `logger.warning(...)`

---

## Décorateurs et compatibilité

- [x] Ajout d’un décorateur neutre @deprecated dans `core/decorators.py` + remplacement global des appels.
- [x] Tous les fichiers ont été mis à jour avec le décorateur @deprecated

---

## 📝 4. Complétion des docstrings

- [x] Compléter les docstrings des fonctions critiques :
  - [ ] `simulate_route_via_osrm()`
  - [x] Fonctions `generate_` et `detect_`
  - [ ] Fonctions d’export (CSV, JSON)
- [ ] Respecter format Google ou NumPy pour les docstrings

---

## ✅ 5. Tests unitaires

- [ ] Créer ou compléter :
  - [ ] `tests/test_gyroscope_simulation.py`
  - [ ] `tests/test_export_csv.py`
  - [ ] `tests/test_event_injection.py`
  - [ ] `tests/test_pipeline.py`

---

## 🧱 6. Refactor fichiers & structure

- [ ] Centraliser toute la config inertielle dans `config/events.yaml`
- [x] Nettoyer les hard-codes dans les scripts (`simulate_and_check.py`)
- [x] Réorganiser :
  - [x] `core/gyroscope.py`
  - [x] `simulator/events/generation.py`
  - [x] `simulator/events/detection.py`
  - [ ] `simulator/plot_utils.py`

---

## 🚀 7. Finalisation version 1.0

- [ ] Ajouter `VERSION.md` avec historique des versions
- [ ] Ajouter `CHANGELOG.md` (blocs par version)
- [ ] Enrichir `README.md`
- [ ] Ajouter un `Makefile` ou `run_all.sh` pour automatiser le pipeline

