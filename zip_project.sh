#!/bin/bash

# Nom du fichier zip de sortie
OUTPUT_ZIP="RoadSimulator3_project.zip"

# Exclure les dossiers out/ et data/ et zipper tout le reste
zip -r $OUTPUT_ZIP . -x "out/*" -x "data/*" -x ".git/*" -x ".venv/*" -x ".pytest/*" -x ".pytest_cache/*"

echo "Archive créée : $OUTPUT_ZIP"

