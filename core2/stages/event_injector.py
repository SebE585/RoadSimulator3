# core2/stages/event_injector.py
"""
Stage — Injection d'événements synthétiques.

Injecte un nombre configurable de chaque type d'événement dans le signal simulé.
Types supportés (issus de config/events.yaml) :
  - harsh_brake     : freinage brusque (pic acc_x négatif)
  - harsh_accel     : accélération brusque (pic acc_x positif)
  - speed_bump      : dos d'âne (oscillation acc_z)
  - pothole         : nid de poule (pic acc_z violent + rebond)
  - curb            : trottoir (pic latéral acc_y + vertical acc_z)
  - sharp_turn      : virage serré (pic acc_y + gyro_z)
  - door_open       : ouverture de porte (oscillation gyro_y + gyro_z, vitesse nulle)

Les événements sont placés aléatoirement sur les segments en mouvement
(sauf door_open : placé sur segments à l'arrêt), espacés d'au moins 5 s.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rs3_contracts.api import Result

logger = logging.getLogger(__name__)

MIN_SPACING_S = 5.0

# Profils d'injection : durée, colonnes affectées, amplitudes
_PROFILES = {
    "harsh_brake": {
        "duration_s": 0.4,
        "signals": [("acc_x", -7.0, -3.5)],  # col, amp_lo, amp_hi
        "requires_moving": True,
    },
    "harsh_accel": {
        "duration_s": 0.4,
        "signals": [("acc_x", 3.0, 5.0)],
        "requires_moving": True,
    },
    "speed_bump": {
        "duration_s": 0.6,
        "signals": [("acc_z", 3.0, 6.0)],  # oscillation
        "shape": "biphasic",  # montée puis descente
        "requires_moving": True,
    },
    "pothole": {
        "duration_s": 0.3,
        "signals": [("acc_z", 5.0, 10.0)],
        "shape": "spike",
        "gyro_signals": [("gyro_x", 0.5, 1.5)],
        "requires_moving": True,
    },
    "curb": {
        "duration_s": 0.6,
        "signals": [("acc_y", 2.5, 4.0), ("acc_z", 5.0, 8.0)],
        "gyro_signals": [("gyro_x", 0.5, 1.0), ("gyro_z", 2.0, 5.0)],
        "requires_moving": True,
    },
    "sharp_turn": {
        "duration_s": 1.0,
        "signals": [("acc_y", 2.0, 4.0)],
        "gyro_signals": [("gyro_z", 0.3, 0.8)],
        "requires_moving": True,
    },
    "door_open": {
        "duration_s": 1.0,
        "signals": [],
        "gyro_signals": [("gyro_y", 5.0, 10.0), ("gyro_z", 2.0, 5.0)],
        "requires_moving": False,  # à l'arrêt
    },
}


def _find_injection_points(
    df: pd.DataFrame, n: int, hz: float,
    exclude: set[int], rng: np.random.Generator,
    requires_moving: bool = True,
) -> list[int]:
    if n <= 0:
        return []

    speed_col = next((c for c in ("speed", "speed_mps") if c in df.columns), None)
    eligible = np.ones(len(df), dtype=bool)

    if speed_col:
        v = pd.to_numeric(df[speed_col], errors="coerce").fillna(0).to_numpy()
        if requires_moving:
            eligible &= v > 1.0
        else:
            eligible &= v < 0.5  # à l'arrêt

    if "event" in df.columns:
        eligible &= df["event"].astype(str).str.strip().eq("")

    margin = int(len(df) * 0.08)
    eligible[:margin] = False
    eligible[-margin:] = False

    spacing = int(MIN_SPACING_S * hz)
    for idx in exclude:
        lo, hi = max(0, idx - spacing), min(len(df), idx + spacing)
        eligible[lo:hi] = False

    candidates = np.where(eligible)[0]
    if len(candidates) == 0:
        return []

    n_actual = min(n, len(candidates))
    step = max(1, len(candidates) // n_actual)
    return [int(candidates[i * step]) for i in range(n_actual)]


def _inject_signal(
    df: pd.DataFrame, idx: int, col: str,
    amplitude: float, n_pts: int, shape: str,
    rng: np.random.Generator,
) -> None:
    if col not in df.columns:
        return
    end = min(len(df), idx + n_pts)
    actual = end - idx
    if actual <= 0:
        return

    t = np.linspace(0, np.pi, actual)

    if shape == "biphasic":
        profile = np.sin(2 * t)  # oscillation complète
    elif shape == "spike":
        profile = np.concatenate([np.sin(t[:actual//2]) * 2, -np.sin(t[actual//2:actual])])
        profile = profile[:actual]
    else:
        profile = np.sin(t)  # cloche standard

    vals = df[col].to_numpy(dtype=float)
    vals[idx:end] += profile * amplitude
    df[col] = vals


def _inject_event(
    df: pd.DataFrame, idx: int, event_type: str,
    profile: dict, hz: float, rng: np.random.Generator,
) -> None:
    n_pts = max(1, int(profile["duration_s"] * hz))
    shape = profile.get("shape", "bell")

    for col, amp_lo, amp_hi in profile.get("signals", []):
        amp = rng.uniform(amp_lo, amp_hi)
        sign = 1 if rng.random() > 0.5 else -1
        if amp_lo < 0:
            sign = 1  # garder le signe négatif
        _inject_signal(df, idx, col, amp * sign, n_pts, shape, rng)

    for col, amp_lo, amp_hi in profile.get("gyro_signals", []):
        amp = rng.uniform(amp_lo, amp_hi) * (1 if rng.random() > 0.5 else -1)
        _inject_signal(df, idx, col, amp, n_pts, shape, rng)

    if "event" not in df.columns:
        df["event"] = ""
    center = idx + n_pts // 2
    if center < len(df):
        df.iat[center, df.columns.get_loc("event")] = event_type


@dataclass
class EventInjector:
    """Injecte des événements synthétiques configurables."""

    name: str = "event_injector"
    n_harsh_brake: int = 0
    n_harsh_accel: int = 0
    n_speed_bump: int = 0
    n_pothole: int = 0
    n_curb: int = 0
    n_sharp_turn: int = 0
    n_door_open: int = 0
    seed: int = 42

    def run(self, ctx) -> Result:
        counts = {
            "harsh_brake": self.n_harsh_brake,
            "harsh_accel": self.n_harsh_accel,
            "speed_bump": self.n_speed_bump,
            "pothole": self.n_pothole,
            "curb": self.n_curb,
            "sharp_turn": self.n_sharp_turn,
            "door_open": self.n_door_open,
        }

        total = sum(counts.values())
        if total == 0:
            return Result(True, "aucun événement à injecter")

        df = ctx.df
        if df is None or df.empty:
            return Result(True, "skip — empty df")

        hz = float(ctx.meta.get("hz", 10))
        rng = np.random.default_rng(self.seed)
        used: set[int] = set()
        injected: dict[str, int] = {}

        for event_type, n in counts.items():
            if n <= 0:
                continue
            profile = _PROFILES[event_type]
            points = _find_injection_points(
                df, n, hz, used, rng,
                requires_moving=profile.get("requires_moving", True),
            )
            for idx in points:
                _inject_event(df, idx, event_type, profile, hz, rng)
                spacing = int(MIN_SPACING_S * hz)
                for j in range(max(0, idx - spacing), min(len(df), idx + spacing)):
                    used.add(j)
            injected[event_type] = len(points)

        ctx.df = df
        summary = ", ".join(f"{k}:{v}" for k, v in injected.items() if v > 0)
        return Result(True, summary or "aucun événement injecté")
