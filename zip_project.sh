#!/bin/bash

# Nom du fichier zip de sortie
OUTPUT_ZIP="RoadSimulator3_project.zip"

# Exclure les dossiers out/ et data/ et zipper tout le reste
zip -r $OUTPUT_ZIP . -x "out/*" -x "data/*"

echo "Archive créée : $OUTPUT_ZIP"

