# """
# Test minimaliste pour l’événement 'dos_dane'.
#
# Ce script génère un DataFrame simulé avec données GPS/IMU de base, injecte un événement
# de type dos d’âne via `generate_dos_dane`, puis tente de le détecter avec `detect_dos_dane`.
#
# Affiche un graphique comparant acc_z et gyro_x, avec une ligne verticale sur la détection.
#
# Ce test valide que l’injection et la détection fonctionnent correctement de bout en bout.
#
# Usage :
#     python tests/test_events/test_dos_dane_minimal.py
# """
# tests/test_events/test_dos_dane_minimal.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simulator.events.generation import generate_dos_dane
from simulator.detectors import detect_dos_dane

# --- Simulation d'un DataFrame minimal ---

N = 200  # nombre de points simulés
df = pd.DataFrame({
    "lat": np.linspace(49.0, 49.001, N),
    "lon": np.linspace(1.0, 1.001, N),
    "speed": np.full(N, 30.0),  # vitesse constante
    "acc_x": np.random.normal(0, 0.1, N),
    "acc_y": np.random.normal(0, 0.1, N),
    "acc_z": np.full(N, 9.81) + np.random.normal(0, 0.1, N),
    "gyro_x": np.zeros(N),
    "gyro_y": np.zeros(N),
    "gyro_z": np.zeros(N),
    "event": pd.Series([np.nan] * N, dtype="object")
})

# --- Configuration minimaliste simulée ---

config = {
    "dos_dane": {
        "max_events": 1,
        "phase_length": 10,
        "amplitude_step": 0.4,
        "max_attempts": 5
    },
    "general": {
        "hz": 10
    }
}

# --- Injection et détection ---

df = generate_dos_dane(df, config=config)

# --- Détection et résultat ---

detected, idx = detect_dos_dane(df)

if detected:
    print(f"✅ Dos d’âne détecté à l’index {idx}")
else:
    print("❌ Aucun dos d’âne détecté.")

# --- Visualisation (optionnelle) ---

plt.figure(figsize=(10, 4))
plt.plot(df["acc_z"], label="acc_z")
plt.plot(df["gyro_x"], label="gyro_x", linestyle='--')
plt.axvline(idx, color="orange", linestyle="--", label="Détection")
plt.legend()
plt.title("Test minimal - Dos d'âne injecté et détecté")
plt.tight_layout()
plt.show()