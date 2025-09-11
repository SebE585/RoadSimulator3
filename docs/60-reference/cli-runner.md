---
title: "CLI Runner"
description: "Documentation détaillée de l’outil en ligne de commande principal pour lancer des simulations dans RoadSimulator3."
---

# CLI Runner
Le **CLI Runner** est l’outil principal pour lancer des simulations dans RoadSimulator3 (RS3). Il orchestre l’exécution du pipeline de simulation, en utilisant des configurations flexibles pour générer des données synthétiques adaptées à de nombreux cas d’usage.

Le script principal est [`runner/run_simulation.py`](../../runner/run_simulation.py). Pour les simulations de flotte multi-véhicules, un autre outil associé est [`runner/run_fleet.py`](../../runner/run_fleet.py).

---

## Usage

Signature complète :
```bash
python -m runner.run_simulation --config CONFIG [--hz N] [--count N] [--profile NAME] [--output OUTPUT_PATH]
```

Outil associé pour les simulations de flotte :
```bash
python -m runner.run_fleet --config CONFIG [options]
```
> **Note :** `runner/run_fleet.py` permet de lancer plusieurs simulations en parallèle selon la configuration d’une flotte de véhicules.

---

## Options principales

- `--config CONFIG`  
  Chemin vers le fichier de configuration YAML.  
  **Obligatoire.**  
  Exemple : `--config config/simulation.yaml`

- `--hz N`  
  Fréquence d’échantillonnage en Hertz (Hz).  
  **Défaut :** `10`  
  Valeur attendue : entier positif.  
  Exemple : `--hz 20`

- `--count N`  
  Nombre de points à générer (utile pour debug ou tests courts).  
  **Défaut :** génère tous les points selon la config/trajectoire.  
  Exemple : `--count 1000`

- `--profile NAME`  
  Profil de simulation à utiliser (définit le type de scénario ou d’objet simulé).  
  Valeurs courantes : `parcels`, `furniture`, etc.  
  Exemple : `--profile parcels`

- `--output OUTPUT_PATH`  
  Chemin du dossier ou du fichier de sortie pour les résultats de simulation.  
  **Défaut :** peut dépendre de la config, sinon dans le dossier courant.  
  Exemple : `--output results/sim1.csv`

---

## Exemples

Simulation simple :
```bash
python -m runner.run_simulation --config config/simulation.yaml
```

Simulation avec fréquence doublée :
```bash
python -m runner.run_simulation --config config/simulation.yaml --hz 20
```

Simulation avec profil personnalisé et sortie dédiée :
```bash
python -m runner.run_simulation --config config/simulation.yaml --profile furniture --output results/furniture.csv
```

---

### Exemples avancés

**Simulation multi-véhicules avec run_fleet.py :**
```bash
python -m runner.run_fleet --config config/fleet.yaml --hz 5 --output results/fleet/
```

**Utilisation d’un fichier de configuration avancé :**
```bash
python -m runner.run_simulation --config config/complex_sim.yaml --output results/complex.csv
```

**Simulation de test avec nombre de points limité :**
```bash
python -m runner.run_simulation --config config/simulation.yaml --count 100
```

---

### Bonnes pratiques

- **Toujours utiliser un fichier de configuration** pour garantir la reproductibilité et la clarté des paramètres de simulation.
- **Versionner les sorties** : inclure la date, l’heure ou un identifiant unique dans les chemins de sortie pour éviter d’écraser des résultats.
- **Valider les résultats** : vérifier la cohérence et l’intégrité des fichiers générés (format, valeurs attendues, absence d’erreurs).
- **Documenter les profils utilisés** dans vos rapports ou tickets pour faciliter le support et la traçabilité.
- **Utiliser un environnement virtuel** pour garantir la compatibilité des dépendances Python.
