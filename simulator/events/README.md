# 📦 Module `simulator.events`

Ce module gère l’injection et la détection **d’événements inertiels** sur des trajectoires simulées (IMU, GPS) à 10 Hz.

---

## 🔁 Point d’entrée principal

### `apply_all_events(df)`

Cette fonction applique automatiquement :

- Les événements **ponctuels** : `acceleration`, `freinage`, `dos_dane`, `trottoir`, `nid_de_poule`
- Les événements **prolongés** : `stop` (moteur éteint), `wait` (ralenti)
- L'**accélération initiale** et la **décélération finale**

Elle centralise l’application des fonctions du module selon les paramètres définis dans `events.yaml`.

---

## 🗂️ Structure des fichiers

| Fichier               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `__init__.py`         | Initialise le module et expose `apply_all_events(df)`                       |
| `config.py`           | Chargement centralisé de la configuration YAML (`events.yaml`)              |
| `generation.py`       | Génération des événements ponctuels (acceleration, freinage, etc.)          |
| `stop_and_wait.py`    | Injection des événements `stop` et `wait` + gestion des profils inertiels   |
| `initial_final.py`    | Accélération de démarrage et décélération de fin réalistes                  |
| `noise.py`            | Ajout de bruit inertiel aux phases `wait` et autres                         |
| `inject.py`           | Pipeline complet (`apply_all_events(df)`) combinant tous les événements     |
| `utils.py`            | Fonctions utilitaires : types de colonnes, nettoyage, etc.                  |

---

## ⚙️ Configuration (`events.yaml`)

Voir [README_events_config.md](./README_events_config.md) pour le détail de la configuration YAML.

---

## 🧪 Tests unitaires

Les tests sont disponibles dans :
```
tests/test_generation.py
tests/test_initial_final.py
tests/test_stop_and_wait.py
tests/test_noise.py
tests/test_inject.py
```

---

## ✅ Prêt pour industrialisation

- Logging complet avec `logger.debug/info`
- Configuration centralisée
- Code modulaire, testé et documenté
