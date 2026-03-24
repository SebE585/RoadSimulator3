# RoadSimulator3 — TODO

> 24 mars 2026

## À déployer
- [ ] `bash deploy/deploy_webui.sh` → simulate.roadsimulator3.fr
- [ ] DNS A record + certbot SSL

## Features
- [ ] Micro-déplacements : demi-tour, créneau, stationnement trottoir/chaussée
- [ ] Roundabout injection (rond-points réalistes, géométrie circulaire)
- [ ] Module UI commun avec Telemachus (event detection, heatmap)
- [ ] Lien "Analyser dans Telemachus" fonctionnel (après déploiement)

## Qualité
- [ ] Fix `FutureWarning` legs_stitch.py fillna downcasting
- [ ] Altitude plugin (RS3_ALTITUDE_CFG) — connecter à l'Elevation API serveur

## Voir aussi
- [telemachus-platform/docs/ROADMAP.md](../telemachus-platform/docs/ROADMAP.md) — Roadmap pipeline D0→D4
