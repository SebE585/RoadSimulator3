# ✅ RoadSimulator3 – TODO pour finaliser la version 1.0

## 🎯 Objectifs de la version 1.0

Livrer une version stable, propre, testée, documentée et prête à être diffusée.

---

## 📁 Nettoyage et structure

- [ ] Supprimer tous les répertoires obsolètes (`old/`, `deprecated/`, etc.).
- [ ] Élaguer les notebooks non utilisés dans `notebooks/`.
- [ ] Supprimer les fichiers inutiles : `.DS_Store`, `.pyc`, `__pycache__/`.
- [ ] Nettoyer les sorties dans `outputs/`, `logs/`, `out*/`, `cache/`.

---

## ⚙️ Organisation du projet

- [ ] Ajouter un `Makefile` avec les commandes standard :
  - `make simulate`
  - `make check`
  - `make clean`
  - `make zip`
- [ ] Ajouter un script `clean_outputs.sh` (logs, outputs, cache).

---

## 🧪 Tests unitaires

- [ ] Ajouter des tests pour la détection inertielle (`detectors.py`)
- [ ] Ajouter des tests pour les événements (`generation.py`)
- [ ] Tester `simulate_and_check.py` sur un petit trajet
- [ ] Ajouter test de bout en bout (simulation → CSV → HTML)

---

## 📦 Dépendances et packaging

- [ ] Figer les versions dans `requirements.txt`
- [ ] Vérifier la compatibilité sur Python 3.12 (ARM et x86)
- [ ] Ajouter un fichier `environment.yml` pour conda (optionnel)

---

## 📖 Documentation

- [ ] Compléter le `README.md` avec :
  - Arborescence du projet
  - Exemples d’usage
  - Dépendances système
- [ ] Ajouter `docs/` avec guide d’utilisation
- [ ] Documenter le fichier de configuration YAML (`config/`)

---

## 🧾 Fichiers de version

- [ ] Créer `VERSION.md` → `v1.0.0`
- [ ] Créer `CHANGELOG.md` avec récapitulatif des changements depuis la v0.9
- [ ] Vérifier que tous les scripts affichent leur version en ligne de commande (`--version`)

---

## ✅ Finalisation

- [ ] Marquer `v1.0.0` dans Git (`git tag v1.0.0`)
- [ ] Générer l’archive `.zip` propre (`RoadSimulator3_v1.0.zip`)
- [ ] Test d'installation sur machine vierge