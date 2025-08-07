import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np

from core import reprojection, kinematics
from simulator.events.generation import (
    generate_acceleration, generate_freinage, generate_dos_dane
)
from simulator.events.noise import inject_inertial_noise
from core.config_loader import load_full_config

def test_timestamps_remain_monotonic_after_full_pipeline():
    # 1. Génère un DataFrame linéaire simple
    timestamps = pd.date_range("2025-01-01", periods=1000, freq="100ms")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "lat": 49.0 + 1e-5 * np.arange(1000),
        "lon": 1.0 + 1e-5 * np.arange(1000),
        "speed": 30.0,  # Vitesse initiale pour l'étape 1
    })

    # 2. Resampling temporel
    df = reprojection.resample_time(df, freq_hz=10)
    assert df["timestamp"].is_monotonic_increasing

    # 3. Reprojection spatiale
    df = reprojection.spatial_reprojection(df, speed_target=40)
    assert df["timestamp"].is_monotonic_increasing

    # ✅ Réinjection explicite de la vitesse après reprojection
    df["speed"] = 40.0

    # 4. Calcul cinématique
    df = kinematics.calculate_heading(df)
    df = kinematics.calculate_linear_acceleration(df, freq_hz=10)
    df = kinematics.calculate_angular_velocity(df, freq_hz=10)

    # 5. Initialisation des colonnes nécessaires
    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "event"]:
        if col not in df.columns:
            df[col] = 0.0 if "acc" in col or "gyro" in col else None

    # 6. Injection de bruit inertiel
    noise_params = {
        "acc_std": 0.05,
        "gyro_std": 0.01,
        "acc_bias": 0.02,
        "gyro_bias": 0.005,
    }
    df = inject_inertial_noise(df, noise_params, seed=42)

    # 7. Injection d’événements inertiels
    config = load_full_config()
    df = generate_acceleration(df, config)
    df = generate_freinage(df, config)
    df = generate_dos_dane(df, config)

    # 8. 🔍 Vérification finale
    assert df["timestamp"].is_monotonic_increasing, "Les timestamps ne sont plus croissants après injection."
