
# RoadSimulator3

🚚 Simulation et Analyse de Trajets Routiers avec Données IMU & GPS

RoadSimulator3 est un simulateur avancé permettant de :

- Générer des trajets réalistes basés sur OSRM
- Injecter des événements inertiels simulés (accélérations, freinages, dos d'âne, trottoir, nid de poule)
- Valider la qualité de la simulation avec une batterie de tests de réalisme
- Produire des visualisations graphiques et des exports CSV exploitables

## 📂 Arborescence du Projet

```
RoadSimulator3/
│
├── config/            # Paramètres globaux et seuils (config.yaml)
├── docs/              # Documentation technique et scientifique
├── tools/             # Scripts d'analyse et de sauvegarde de config
├── core/              # Calculs de trajectoire, sinuosité, analyse route
├── simulator/         # Générateurs et détecteurs d'événements
├── check/             # Vérifications de réalisme
├── runner/            # Pipeline complet simulation + analyse
├── tests/             # Tests unitaires
├── static/, templates/ # Visualisations web et ressources
├── out/               # Résultats de simulation
└── archive/           # Anciennes versions
```

## 🚀 Installation

### Prérequis :

- Python 3.11+
- OSRM backend opérationnel en local (port 5001)

### Étapes :

```bash
git clone <repository_url>
cd RoadSimulator3
pip install -r requirements.txt
./run_simulate.sh
./run_tests.sh
```

## ⚙️ Configuration

- `config/config.yaml` : paramètres de simulation, seuils des événements
- Pour sauvegarder les réglages actuels :

```bash
python tools/save_config.py
```

## 📈 Résultats

Chaque simulation génère :

- Un CSV détaillé : `out/simulated_<timestamp>/trace.csv`
- Un graphe des événements : `graph_events.png`
- Un résumé de réalisme en console

## 🛠️ Outils complémentaires

- `tools/analyse_events.py <path_to_csv>` : analyser les événements dans un CSV
- `tools/save_config.py` : sauvegarder la configuration courante en YAML

## 🧪 Tests

Tests couvrant :

- La génération et détection des événements
- La cohérence inertielle
- La sinuosité et la détection de virages

Lancement : `./run_tests.sh`

## 🤩 Technologies

- Python 3
- Pandas, NumPy, Matplotlib
- OSRM pour le routage
- Geopy pour les calculs géographiques

## 📚 Documentation

Détails scientifiques et paramètres dans `docs/` :

- `parametres_evenements.md`
- `config_simulation.md`

## 📝 Licence

Licence MIT.

## 👨‍💻 Auteur

Sébastien Edet

---

# 📌 Roadmap RoadSimulator

## ✅ Version 1.1.1 – Petits correctifs
- Suppression des warnings pandas
- Ordre strict des colonnes CSV :
  - `timestamp`, `lat`, `lon`, `speed`, `acc_x`, `acc_y`, `acc_z`, `event`
- Validation automatique du CSV post-export

## ✅ Version 1.1 – Corrections trajectoire & inertie
- Correction des sinuosités vides
- Correction détection :
  - Rond-points
  - Virages (serré, moyen, large)
- Amélioration accélération latérale dans les virages
- Paramètre debug pour visualisation des virages/ronds-points détectés

## 🕐 Version 1.2 – Arrêts moteur
- Ajout :
  - `stop` : moteur éteint ≥ 2 min
  - `wait` : moteur tournant entre 30s et 2 min
- Détection et simulation des arrêts
- Marquage dans la colonne `event` et visualisation

## 🕐 Version 1.3 – Adaptation vitesse
- Variation de la vitesse en fonction :
  - Type de route (via OSRM ou OSM)
  - Niveau de sinuosité détectée
- Effet réaliste sur la vitesse de croisière

## 🕐 Version 1.4 – Sortie HTML
- Carte interactive :
  - Trajet + événements
  - Légende dynamique
- Exports :
  - HTML interactif
  - PNG statique (via Selenium)
- Tableau récapitulatif des événements et indicateurs

## 🕐 Version 1.5 – Heatmap accélérations
- Génération d'une heatmap :
  - Accélérations longitudinales vs latérales
- Export :
  - PNG statique
  - HTML interactif (optionnel)
- Filtres dynamiques pour lisibilité

## 🕐 Version 1.6 – Trajet longue distance aléatoire
- Trajets simulés >400 km
- 15+ livraisons
- Événements inertiels variés injectés
- Optimisations de performance

## 🕐 Version 1.7 – Nouveaux indicateurs
- Indicateurs supplémentaires :
  - Nombre de stops / waits
  - Dénivelé cumulé
  - Temps passé en virage par catégorie
  - Moyenne accélération latérale
  - Vitesse moyenne par type de route
- Visualisations graphiques associées

## 🕐 Version 2.0 – Application Web
- Interface web complète :
  - Création de trajets personnalisés
  - Simulation d’événements manuelle ou automatique
  - Visualisation des indicateurs et trajets
- Fonctionnalités :
  - Exports CSV, JSON, HTML, PNG
  - Authentification utilisateur (optionnel)
- Stack : Flask + React/Vue.js + MongoDB (optionnel pour l’historique)

---

# 🔧 Pistes d’amélioration & Config recommandée

## Seuils de détection virages

- **Seuil recommandé pour validation des virages (acc_y) : 0.15 m/s²**
- Ce seuil équilibre bien la sensibilité et la robustesse de la détection, validé par tests statistiques approfondis.
- À ajuster avec prudence selon le niveau de bruit et la nature des trajets simulés.

## Configuration recommandée (extrait config.yaml) :

```yaml
general:
  detection_thresholds:
    acc_y_virage: 0.5
    heading_virage: 15
    validate_turns_threshold: 0.15
```

## Utilisation dans le pipeline

- Le seuil `validate_turns_threshold` est chargé depuis la configuration et utilisé dans la fonction de validation des virages.
- Il est conseillé d’adapter ce seuil dans `config/config.yaml` et de relancer la simulation pour impact immédiat.

---

# 📁 Fichiers utiles

- `runner/simulate_and_check.py` : pipeline complet simulation + analyse intégrant ce seuil
- `config/config.yaml` : configuration globale avec seuils
- `helpers/diagnose_turn_validation.py` : script d’analyse dédié à la validation des seuils
- `helpers/explore_turn_detection_params.py` : exploration automatique des seuils

---

# 👩‍🔧 Comment contribuer ?

- Signaler les bugs et proposer des améliorations via GitHub Issues
- Soumettre des Pull Requests avec tests unitaires et documentation mise à jour
- Participer aux discussions sur la roadmap

---

# 📝 Remerciements

Merci à tous les contributeurs pour leur implication dans ce projet open-source.

---

© 2025 Sébastien Edet – Licence MIT
