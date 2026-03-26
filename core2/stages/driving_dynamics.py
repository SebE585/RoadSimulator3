# core2/stages/driving_dynamics.py
"""
Stage — Driving Dynamics Injector v2.

Modélise la variabilité naturelle de conduite via des processus
d'Ornstein-Uhlenbeck (mean-reverting random walk) sur :
  - ax : micro-corrections pédale (freinage/accélération)
  - ay : micro-corrections volant (entrée/sortie virage)
  - speed : oscillations de vitesse en croisière

Produit une distribution GAUSSIENNE centrée (pas bimodale),
conforme aux données réelles de conduite urbaine.

Paramètres calibrés sur : heatmap bi-histogramme données réelles livraison.
Cible : ax_std ≈ 0.10-0.15g, ay_std ≈ 0.15-0.20g, centre ±0.05g < 30%.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rs3_contracts.api import Result

logger = logging.getLogger(__name__)


def _ornstein_uhlenbeck(n: int, dt: float, theta: float, sigma: float, rng) -> np.ndarray:
    """
    Processus d'Ornstein-Uhlenbeck : mean-reverting random walk.
    dx = -theta * x * dt + sigma * sqrt(dt) * dW

    theta : taux de retour vers 0 (s⁻¹). Plus haut = moins de mémoire.
    sigma : amplitude du bruit (unité/s^0.5).
    Stationary std = sigma / sqrt(2*theta)
    """
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] * dt + sigma * np.sqrt(dt) * rng.normal()
    return x


@dataclass
class DrivingDynamics:
    """Variabilité de conduite réaliste via processus stochastiques."""

    name: str = "driving_dynamics"

    # Jitter longitudinal (pédale) — Ornstein-Uhlenbeck
    # Stationary std = sigma_ax / sqrt(2*theta_ax) ≈ 1.5/sqrt(6) ≈ 0.61 m/s²
    theta_ax: float = 3.0       # retour vers 0 en ~0.3s
    sigma_ax: float = 1.0       # amplitude bruit — stationary std ≈ 0.41 m/s² ≈ 0.04g

    # Jitter latéral (volant) — Ornstein-Uhlenbeck
    # Stationary std = sigma_ay / sqrt(2*theta_ay) ≈ 1.0/sqrt(4) ≈ 0.50 m/s²
    theta_ay: float = 2.0       # retour vers 0 en ~0.5s
    sigma_ay: float = 1.0       # amplitude bruit

    # Oscillations vitesse — Ornstein-Uhlenbeck
    theta_speed: float = 1.0    # retour lent (1s)
    sigma_speed: float = 1.5    # ≈ ±0.5 m/s stationary

    seed: int = 99

    def run(self, ctx) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(True, "skip — empty df")

        hz = float(ctx.meta.get("hz", 10))
        n = len(df)
        dt = 1.0 / hz
        rng = np.random.default_rng(self.seed)

        speed_col = "speed" if "speed" in df.columns else "speed_mps"
        if speed_col not in df.columns:
            return Result(True, "skip — pas de colonne speed")

        v = df[speed_col].to_numpy(dtype=float).copy()
        moving = v > 0.5

        if not moving.any():
            return Result(True, "skip — pas de mouvement")

        # ── Oscillations vitesse (Ornstein-Uhlenbeck) ────────────────
        speed_jitter = _ornstein_uhlenbeck(n, dt, self.theta_speed, self.sigma_speed, rng)
        # Proportionnel à la vitesse (pas de jitter à l'arrêt)
        v_factor = np.clip(v / 5.0, 0, 1)
        v[moving] += (speed_jitter[moving] * v_factor[moving])
        v = np.clip(v, 0, None)
        v[~moving] = df[speed_col].to_numpy(dtype=float)[~moving]
        df[speed_col] = v.astype("float32") if df[speed_col].dtype == np.float32 else v

        # ── Jitter acc longitudinal (Ornstein-Uhlenbeck) ─────────────
        ax_jitter = _ornstein_uhlenbeck(n, dt, self.theta_ax, self.sigma_ax, rng)
        ax_jitter[~moving] = 0

        # ── Jitter acc latéral (Ornstein-Uhlenbeck) ──────────────────
        ay_jitter = _ornstein_uhlenbeck(n, dt, self.theta_ay, self.sigma_ay, rng)
        ay_jitter[~moving] = 0

        df["_driving_jitter_ax"] = ax_jitter.astype("float32")
        df["_driving_jitter_ay"] = ay_jitter.astype("float32")

        ctx.df = df

        ax_std = np.std(ax_jitter[moving])
        ay_std = np.std(ay_jitter[moving])
        return Result(True, f"OU process: ax_std={ax_std:.2f} ay_std={ay_std:.2f} m/s²")
