#!/bin/bash
# run_tests.sh
# Script pour lancer pytest avec PYTHONPATH configuré à la racine du projet

# Récupère le chemin absolu du dossier courant (racine du projet)
PROJECT_ROOT="$(pwd)"

# Configure PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"

echo "PYTHONPATH configuré à : $PYTHONPATH"
echo "Lancement des tests pytest..."

pytest "$@"


