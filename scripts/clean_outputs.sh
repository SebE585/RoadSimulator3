#!/bin/bash
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name ".DS_Store" -delete
rm -rf logs/* out/* outputs/*
echo "Nettoyage terminé."
