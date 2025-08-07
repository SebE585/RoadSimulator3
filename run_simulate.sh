#!/bin/bash
# run_simulate.sh
# Script de lancement pour simulate_and_check.py avec PYTHONPATH configuré

export PYTHONPATH=$(pwd)
echo "PYTHONPATH configuré à : $PYTHONPATH"

python -m runner.simulate_and_check
