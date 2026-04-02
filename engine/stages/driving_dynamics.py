# engine/stages/driving_dynamics.py
"""
Stage — Driving Dynamics Injector v3.

Modélise la variabilité naturelle de conduite via des processus
d'Ornstein-Uhlenbeck (mean-reverting random walk) sur :
  - ax : micro-corrections pédale (freinage/accélération)
  - ay : micro-corrections volant (entrée/sortie virage)
  - speed : oscillations de vitesse en croisière

Trois profils calibrés sur données réelles (Vision.jpg) :
  - SEVERE : Std Gx ≈ 0.07g, Std Gy ≈ 0.13g
  - MOYEN  : Std Gx ≈ 0.07g, Std Gy ≈ 0.09g
  - DOUX   : Std Gx ≈ 0.06g, Std Gy ≈ 0.05g
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rs3_contracts.api import Result

logger = logging.getLogger(__name__)

# ── Presets calibrés sur Vision.jpg ─────────────────────────────
# Chaque preset définit les paramètres DrivingDynamics + IMUProjector
PROFILES = {
    "severe": {
        "sigma_ax": 0.6,
        "sigma_ay": 0.6,
        "theta_ax": 4.0,
        "theta_ay": 2.0,
        "sigma_speed": 0.8,
        "theta_speed": 0.5,
        "smooth_speed_sigma": 18.0,
        "imu_smooth_window_s": 1.3,
        "imu_min_turn_radius_m": 22.0,
        # Multi-OU calibré sur AEGIS (conduite réelle, 24Hz)
        "multi_ou_ax": [(0.3, 0.50), (2.0, 0.02), (10.0, 1.50)],
        "multi_ou_ay": [(0.3, 0.45), (2.0, 0.02), (10.0, 1.50)],
    },
    "moyen": {
        "sigma_ax": 0.6,
        "sigma_ay": 0.3,
        "theta_ax": 4.0,
        "theta_ay": 2.0,
        "sigma_speed": 0.6,
        "theta_speed": 0.5,
        "smooth_speed_sigma": 18.0,
        "imu_smooth_window_s": 1.3,
        "imu_min_turn_radius_m": 30.0,
        "multi_ou_ax": [(0.3, 0.46), (2.0, 0.01), (10.0, 1.11)],
        "multi_ou_ay": [(0.3, 0.42), (2.0, 0.01), (10.0, 1.38)],
    },
    "doux": {
        "sigma_ax": 0.5,
        "sigma_ay": 0.1,
        "theta_ax": 4.0,
        "theta_ay": 2.0,
        "sigma_speed": 0.4,
        "theta_speed": 0.5,
        "smooth_speed_sigma": 25.0,
        "imu_smooth_window_s": 2.0,
        "imu_min_turn_radius_m": 90.0,
        "multi_ou_ax": [(0.3, 0.30), (2.0, 0.01), (10.0, 0.70)],
        "multi_ou_ay": [(0.3, 0.25), (2.0, 0.01), (10.0, 0.80)],
    },
}


def get_profile(name: str) -> dict:
    """Retourne le preset pour un profil donné (insensible à la casse)."""
    key = name.strip().lower()
    # Aliases
    aliases = {"sévère": "severe", "sévere": "severe", "severe": "severe",
               "moyen": "moyen", "medium": "moyen",
               "doux": "doux", "gentle": "doux", "soft": "doux"}
    key = aliases.get(key, key)
    if key not in PROFILES:
        raise ValueError(f"Profil inconnu: '{name}'. Choix: {list(PROFILES.keys())}")
    return PROFILES[key]


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


def _multi_ornstein_uhlenbeck(n: int, dt: float, components: list, rng) -> np.ndarray:
    """
    Superposition de processus OU à différentes échelles temporelles.

    Reproduit la structure spectrale des données réelles (AEGIS) :
      - Basse freq (θ~0.3) : dévers route, vent latéral
      - Moyenne freq (θ~2) : virages, manoeuvres
      - Haute freq (θ~10) : corrections volant, vibrations route

    components : liste de (theta, sigma) pour chaque composante.
    Calibré sur AEGIS trip 25 (24Hz, Graz Autriche).
    """
    total = np.zeros(n)
    for theta, sigma in components:
        total += _ornstein_uhlenbeck(n, dt, theta, sigma, rng)
    return total


@dataclass
class DrivingDynamics:
    """Variabilité de conduite réaliste via processus stochastiques."""

    name: str = "driving_dynamics"

    # Profil conducteur : "severe", "moyen", "doux" (ou None pour params manuels)
    profile: str | None = "moyen"

    # Params manuels (écrasés par le profil si profile != None)
    theta_ax: float = 4.0
    sigma_ax: float = 1.4
    theta_ay: float = 2.0
    sigma_ay: float = 0.8
    theta_speed: float = 0.5
    sigma_speed: float = 0.6
    smooth_speed_sigma: float = 15.0

    # Multi-OU (superposition de composantes à différentes fréquences)
    # Calibré sur AEGIS 24Hz — reproduit le spectre des données réelles
    multi_ou_ax: list = field(default_factory=lambda: [(0.3, 0.46), (2.0, 0.01), (10.0, 1.11)])
    multi_ou_ay: list = field(default_factory=lambda: [(0.3, 0.42), (2.0, 0.01), (10.0, 1.38)])
    use_multi_ou: bool = True  # False = ancien OU simple (compatibilité)

    seed: int = 99

    def __post_init__(self):
        if self.profile:
            p = get_profile(self.profile)
            self.theta_ax = p["theta_ax"]
            self.sigma_ax = p["sigma_ax"]
            self.theta_ay = p["theta_ay"]
            self.sigma_ay = p["sigma_ay"]
            self.theta_speed = p["theta_speed"]
            self.sigma_speed = p["sigma_speed"]
            self.smooth_speed_sigma = p["smooth_speed_sigma"]
            if "multi_ou_ax" in p:
                self.multi_ou_ax = p["multi_ou_ax"]
            if "multi_ou_ay" in p:
                self.multi_ou_ay = p["multi_ou_ay"]

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

        # ── Lissage des transitions discrètes (SpeedSync) ──────────
        from scipy.ndimage import gaussian_filter1d
        v_orig = v.copy()
        if self.smooth_speed_sigma > 0:
            v_smooth = gaussian_filter1d(v, sigma=self.smooth_speed_sigma)
            v[moving] = v_smooth[moving]
            v = np.clip(v, 0, None)

        # ── Oscillations vitesse (Ornstein-Uhlenbeck) ────────────────
        speed_jitter = _ornstein_uhlenbeck(n, dt, self.theta_speed, self.sigma_speed, rng)
        v_factor = np.clip(v / 5.0, 0, 1)
        v[moving] += (speed_jitter[moving] * v_factor[moving])
        v = np.clip(v, 0, None)
        v[~moving] = v_orig[~moving]
        df[speed_col] = v.astype("float32") if df[speed_col].dtype == np.float32 else v

        # ── Jitter acc longitudinal ───────────────────────────────────
        if self.use_multi_ou and self.multi_ou_ax:
            ax_jitter = _multi_ornstein_uhlenbeck(n, dt, self.multi_ou_ax, rng)
        else:
            ax_jitter = _ornstein_uhlenbeck(n, dt, self.theta_ax, self.sigma_ax, rng)
        ax_jitter[~moving] = 0

        # ── Jitter acc latéral ────────────────────────────────────────
        if self.use_multi_ou and self.multi_ou_ay:
            ay_jitter = _multi_ornstein_uhlenbeck(n, dt, self.multi_ou_ay, rng)
        else:
            ay_jitter = _ornstein_uhlenbeck(n, dt, self.theta_ay, self.sigma_ay, rng)
        ay_jitter[~moving] = 0

        df["_driving_jitter_ax"] = ax_jitter.astype("float32")
        df["_driving_jitter_ay"] = ay_jitter.astype("float32")

        ctx.df = df

        profile_name = self.profile or "custom"
        ax_std = np.std(ax_jitter[moving])
        ay_std = np.std(ay_jitter[moving])
        return Result(True, f"[{profile_name}] OU: ax_std={ax_std:.2f} ay_std={ay_std:.2f} m/s²")
