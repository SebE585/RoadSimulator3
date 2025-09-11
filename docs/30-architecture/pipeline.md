# !!! tip "À la une — Thèse RS3 et pipeline scientifique"
#     Les étapes du pipeline décrites ici s’appuient directement sur la thèse de référence : **S. Edet (2024)**, *Modélisation et simulation routière pour le projet RoadSimulator3*, Zenodo. DOI : https://zenodo.org/records/16568796. 
#     Chaque stage (génération, interpolation, injection d’événements, bruit inertiel, enrichissement, validation, export) correspond à un **module scientifique documenté** et validé.
# Pipeline

Le pipeline RS3 est modulaire et se compose d’étapes successives.

---

## Diagramme
```mermaid
flowchart TD
    A[Fetch Points GPS OSRM] --> B[Interpolation temporelle 10 Hz]
    B --> C[Injection événements (arrêts, virages, etc.)]
    C --> D[Modélisation bruit inertiel (acc, gyro)]
    D --> E[Enrichissement contexte (type de route, pente, météo)]
    E --> F[Validation (temporelle, spatiale, inertielle)]
    F --> G[Exports (CSV, JSON, HTML, PNG, Parquet)]
```

---

## Étapes principales

### Étape 1 — Récupération trajectoire GPS (OSRM)
  - **Rôle** : Obtenir les points GPS bruts de la trajectoire via OSRM.
  - **Entrée** : Requête de trajet (point de départ, point d’arrivée).
  - **Sortie** : DataFrame de points GPS avec timestamps.

### Étape 2 — Interpolation temporelle à 10 Hz
  - **Rôle** : Générer une trajectoire avec un échantillonnage temporel uniforme à 10 Hz.
  - **Entrée** : DataFrame de points GPS bruts.
  - **Sortie** : DataFrame interpolé avec positions à intervalles réguliers.

### Étape 3 — Injection événements (arrêts, virages, etc.)
  - **Rôle** : Ajouter des événements spécifiques dans la trajectoire pour simuler des comportements réalistes.
  - **Entrée** : Trajectoire interpolée.
  - **Sortie** : Trajectoire annotée avec événements.

### Étape 4 — Modélisation du bruit inertiel (accéléromètre, gyroscope)
  - **Rôle** : Simuler le bruit des capteurs inertiels pour rendre la simulation plus fidèle.
  - **Entrée** : Trajectoire annotée.
  - **Sortie** : Trajectoire avec données inertielles bruitées.

### Étape 5 — Enrichissement contexte (type de route, pente, météo)
  - **Rôle** : Ajouter des informations contextuelles qui peuvent influencer la simulation.
  - **Entrée** : Trajectoire avec données inertielles.
  - **Sortie** : Trajectoire enrichie avec données contextuelles.

### Étape 6 — Validation (temporelle, spatiale, inertielle)
  - **Rôle** : Vérifier la cohérence et la qualité des données simulées.
  - **Entrée** : Trajectoire enrichie.
  - **Sortie** : Rapport de validation et/ou DataFrame validé.

### Étape 7 — Exports (CSV, JSON, HTML, PNG, Parquet)
  - **Rôle** : Exporter les résultats dans différents formats pour analyse et visualisation.
  - **Entrée** : Trajectoire validée.
  - **Sortie** : Fichiers exportés.

---

## Extensibilité
- Le pipeline RS3 est conçu pour être extensible via une architecture de **plugins**.  
- Chaque **stage** respecte un contrat clair : il reçoit une DataFrame en entrée et produit une DataFrame en sortie, facilitant l’intégration et la substitution de modules.  
- Des plugins existants comme *Altitude* (ajout d’altitude à la trajectoire) et *Fleet* (gestion de flottes de véhicules) illustrent cette flexibilité.  
- De futurs plugins pourront intégrer des indicateurs supplémentaires, comme des mesures de qualité de conduite ou des données environnementales.
