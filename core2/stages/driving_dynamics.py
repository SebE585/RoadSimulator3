# core2/stages/driving_dynamics.py
"""
Stage — Driving Dynamics Injector.

Ajoute de la variabilité réaliste au profil de vitesse APRÈS le smoothing,
pour simuler le comportement naturel d'un conducteur humain.

3 composantes :
  1. Micro-oscillations de vitesse (le conducteur ne maintient pas une vitesse constante)
  2. Variabilité d'entrée/sortie de virage (accélération latérale non uniforme)
  3. Modulation de freinage (freinage par à-coups, pas une rampe linéaire)

Ce stage ne casse PAS le réalisme QA car :
  - Il n'altère pas la trajectoire (lat/lon inchangés)
  - Il n'altère pas les arrêts (vitesse = 0 reste 0)
  - Les variations sont bornées (paramètres configurables)
  - La validation "cadence_10hz" et "start_zero/end_zero" restent valides

Inspiré de : données réelles livraison urbaine, σ_ax ≈ 0.1-0.15g, σ_ay ≈ 0.15-0.2g
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rs3_contracts.api import Result

logger = logging.getLogger(__name__)


@dataclass
class DrivingDynamics:
    """Injecte de la variabilité de conduite réaliste dans le profil de vitesse."""

    name: str = "driving_dynamics"

    # Micro-oscillations vitesse (croisière)
    speed_oscillation_mps: float = 0.8     # ±0.8 m/s ≈ ±3 km/h
    speed_oscillation_period_s: float = 2.0  # période 2s

    # Variabilité freinage/accélération
    accel_jitter_mps2: float = 0.8          # ±0.8 m/s² — corrections pédale réalistes
    accel_jitter_freq_hz: float = 2.5       # fréquence 2.5 Hz

    # Variabilité latérale (entrée/sortie virage)
    lateral_jitter_mps2: float = 0.15       # ±0.15 m/s² de variation sur ay
    lateral_jitter_freq_hz: float = 1.5     # fréquence 1.5 Hz

    seed: int = 99

    def run(self, ctx) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(True, "skip — empty df")

        hz = float(ctx.meta.get("hz", 10))
        n = len(df)
        dt = 1.0 / hz
        rng = np.random.default_rng(self.seed)

        # Identifier les segments en mouvement (ne pas toucher les arrêts)
        speed_col = "speed" if "speed" in df.columns else "speed_mps"
        if speed_col not in df.columns:
            return Result(True, "skip — pas de colonne speed")

        v = df[speed_col].to_numpy(dtype=float).copy()
        moving = v > 0.5  # en mouvement

        if not moving.any():
            return Result(True, "skip — pas de mouvement")

        # ── 1. Micro-oscillations de vitesse en croisière ────────────────
        # Sinusoïde basse fréquence + bruit coloré
        t = np.arange(n) * dt
        osc_freq = 2 * np.pi / self.speed_oscillation_period_s
        # Sinusoïde avec phase aléatoire
        phase = rng.uniform(0, 2 * np.pi)
        speed_osc = self.speed_oscillation_mps * np.sin(osc_freq * t + phase)
        # Ajouter du bruit coloré (rolling noise)
        colored_noise = pd.Series(rng.normal(0, self.speed_oscillation_mps * 0.5, n)).rolling(
            int(hz * 0.5), center=True, min_periods=1
        ).mean().values
        speed_variation = speed_osc + colored_noise

        # Appliquer uniquement en mouvement, proportionnel à la vitesse
        v_factor = np.clip(v / 5.0, 0, 1)  # facteur 0→1 pour v entre 0 et 5 m/s
        v[moving] += (speed_variation[moving] * v_factor[moving]).astype(float)
        v = np.clip(v, 0, None)  # pas de vitesse négative

        # Les arrêts restent à 0
        v[~moving] = df[speed_col].to_numpy(dtype=float)[~moving]

        df[speed_col] = v.astype("float32") if df[speed_col].dtype == np.float32 else v

        # ── 2. Préparer les jitters acc pour l'IMUProjector ──────────────
        # On stocke les jitters dans des colonnes temporaires
        # L'IMUProjector calculera acc_x = gradient(v) + gravity + jitter_ax
        # et acc_y = v * gyro_z + jitter_ay

        # Jitter longitudinal (micro-corrections pédale)
        f_ax = 2 * np.pi * self.accel_jitter_freq_hz
        ax_jitter = self.accel_jitter_mps2 * (
            0.6 * np.sin(f_ax * t + rng.uniform(0, 2 * np.pi)) +
            0.4 * rng.normal(0, 1, n)
        )
        # Smooth légèrement (pas de discontinuité)
        ax_jitter = pd.Series(ax_jitter).rolling(
            max(1, int(hz * 0.2)), center=True, min_periods=1
        ).mean().values
        ax_jitter[~moving] = 0  # pas de jitter à l'arrêt

        # Jitter latéral (micro-corrections volant)
        f_ay = 2 * np.pi * self.lateral_jitter_freq_hz
        ay_jitter = self.lateral_jitter_mps2 * (
            0.5 * np.sin(f_ay * t + rng.uniform(0, 2 * np.pi)) +
            0.5 * rng.normal(0, 1, n)
        )
        ay_jitter = pd.Series(ay_jitter).rolling(
            max(1, int(hz * 0.3)), center=True, min_periods=1
        ).mean().values
        ay_jitter[~moving] = 0

        # Stocker dans le DataFrame pour que l'IMUProjector les récupère
        df["_driving_jitter_ax"] = ax_jitter.astype("float32")
        df["_driving_jitter_ay"] = ay_jitter.astype("float32")

        ctx.df = df

        v_std = np.std(speed_variation[moving]) if moving.sum() > 10 else 0
        return Result(True,
            f"speed_osc=±{self.speed_oscillation_mps:.1f}m/s "
            f"ax_jitter=±{self.accel_jitter_mps2:.2f}m/s² "
            f"ay_jitter=±{self.lateral_jitter_mps2:.2f}m/s²")
