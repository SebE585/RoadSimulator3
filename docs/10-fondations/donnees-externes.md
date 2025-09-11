---
title: Données externes utilisées par RoadSimulator3
description: Présentation des types de données externes intégrées dans RoadSimulator3, leur préparation et leur portée scientifique et industrielle.
tags: [rs3, docs]
---

# Données externes utilisées par RoadSimulator3

> **Statut** : 🧩 *Version développée* — documentation complète.  
> **Objectif** : Présenter les sources, formats, et procédures d’intégration des données externes dans RoadSimulator3.  
> **Lecteur cible** : Chercheurs, ingénieurs, développeurs impliqués dans la simulation routière et l’analyse géospatiale.

## Sommaire
- [Contexte](#contexte)
- [Types de données](#types-de-données)
- [Procédure d’intégration](#procédure-dintégration)
- [Portée scientifique et industrielle](#portée-scientifique-et-industrielle)
- [Référence](#référence)

## Contexte

RoadSimulator3 (RS3) est une plateforme avancée de simulation routière qui nécessite l’intégration de données géospatiales et environnementales externes pour assurer la précision et la pertinence des modèles. Ces données proviennent de sources ouvertes et institutionnelles, couvrant des informations topographiques, cartographiques, météorologiques et contextuelles. La diversité et la volumétrie de ces données impliquent des processus rigoureux de préparation, de validation et d’intégration afin d’assurer leur compatibilité avec les algorithmes et les modules de RS3.

## Types de données

### Données OpenStreetMap (OSM) au format PBF

Les données cartographiques vectorielles issues d’OpenStreetMap sont utilisées pour modéliser le réseau routier, les infrastructures et les points d’intérêt. Le format PBF (Protocolbuffer Binary Format) est privilégié pour son efficacité en termes de taille et de rapidité de traitement.

### GraphML via OSMnx

Pour faciliter l’analyse et la manipulation du graphe routier, RS3 exploite des fichiers GraphML générés par la bibliothèque OSMnx. Ces fichiers permettent d’obtenir une représentation structurée des réseaux routiers, avec attribution d’attributs topologiques et sémantiques.

### Modèles Numériques de Terrain (MNT) SRTM et BD ALTI

Les données d’élévation issues de la mission SRTM (Shuttle Radar Topography Mission) et de la BD ALTI (Base de Données Altimétrique) fournissent les informations nécessaires à la modélisation du relief et des pentes, indispensables aux calculs de dynamique et d’écoulement.

### Données météo et contextuelles

Les paramètres météorologiques (température, précipitations, vent) et autres données contextuelles (trafic, événements) sont intégrés pour simuler des conditions réalistes et variables dans le temps. Ces données proviennent de sources institutionnelles ou de services web spécialisés.

## Procédure d’intégration

### Préparation avec Docker

La préparation des données externes est orchestrée via des conteneurs Docker dédiés, garantissant la reproductibilité et l’isolation des processus. Ces conteneurs automatisent le téléchargement, la conversion et la validation des fichiers sources.

### Formats attendus

- OSM PBF pour les données cartographiques brutes.  
- GraphML pour les graphes routiers structurés.  
- GeoTIFF ou formats raster standard pour les MNT.  
- JSON ou CSV pour les données météo et contextuelles.

### Volumétrie

La volumétrie des données varie selon les zones géographiques couvertes et la granularité souhaitée. Typiquement, un secteur urbain moyen génère plusieurs centaines de mégaoctets de données vectorielles et raster, nécessitant une gestion optimisée en mémoire et stockage.

## Portée scientifique et industrielle

L’intégration rigoureuse des données externes dans RS3 permet d’améliorer la fidélité des simulations routières, contribuant ainsi à la recherche en mobilité, en urbanisme et en gestion des infrastructures. Sur le plan industriel, cette approche facilite le développement de solutions innovantes pour la planification, la maintenance et la sécurité routière, en s’appuyant sur des données fiables et actualisées.

## Référence

- OpenStreetMap contributors, Planet Dump, https://planet.openstreetmap.org/  
- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems, 65, 126-139.  
- NASA SRTM Data, https://www2.jpl.nasa.gov/srtm/  
- IGN BD ALTI, https://geoservices.ign.fr/documentation/donnees-de-reference/BDALTI.html  
- Météo-France Open Data, https://meteofrance.com/donnees-libres

---
### TODO
- [ ] Ajouter exemples de commandes Docker  
- [ ] Illustrations des données intégrées  
- [ ] Liens croisés vers modules RS3 associés
