# ✅ RoadSimulator3 – TODO pour finaliser la version 1.0

## 🎯 Objectifs de la version 1.0  
Livrer une version **stable, propre, testée, documentée et prête à diffuser**.

---

## 📁 Nettoyage et structure
- [x] Suppression des répertoires obsolètes (`old/`, `deprecated/`, etc.)
- [x] Élagage des notebooks non utilisés dans `notebooks/`
- [x] Suppression des fichiers inutiles : `.DS_Store`, `.pyc`, `__pycache__/`
- [x] Nettoyage des sorties (`outputs/`, `logs/`, `out*/`, `cache/`)
- [ ] Vérification finale post-`scripts/clean_outputs.sh`

---

## ⚙️ Organisation du projet
- [x] Ajout d’un `Makefile` avec commandes :
  - `make simulate`
  - `make check`
  - `make clean`
  - `make zip`
- [x] Script `scripts/clean_outputs.sh` pour purge des outputs
- [ ] Vérifier cohérence des noms de scripts (`simulate_xxx.py`, `check_xxx.py`, etc.)

---

## 🧪 Tests unitaires
- [ ] Tests pour détection inertielle (`detectors.py`)
- [ ] Tests pour événements (`generation.py`)
- [ ] Test `simulate_and_check.py` sur petit trajet
- [ ] Test bout en bout (simulation → CSV → HTML)

---

## 📦 Dépendances et packaging
- [x] Versions figées dans `requirements.txt`
- [x] Compatibilité Python 3.12 ARM/x86 validée
- [ ] Ajouter fichier `environment.yml` (conda)

---

## 📖 Documentation
- [ ] Compléter `README.md` (arborescence, exemples, dépendances)
- [ ] Ajouter `docs/` avec guide d’utilisation
- [ ] Documenter le YAML de config inertielle (`config/events.yaml`)

---

## 🧾 Fichiers de version
- [x] `VERSION.md` → `v1.0.0`
- [x] `CHANGELOG.md` mis à jour (depuis v0.9)
- [ ] Vérifier que tous les scripts supportent `--version`

---

## ✅ Finalisation
- [ ] Marquage Git `v1.0.0`
- [ ] Génération de l’archive `.zip` propre (`RoadSimulator3_v1.0.zip`)
- [ ] Test d’installation sur machine vierge