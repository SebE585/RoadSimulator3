# core2/stages/speed_smoother.py
from __future__ import annotations

import pandas as pd
import numpy as np

from ..contracts import Result
from ..context import Context


class SpeedSmoother:
    """
    Lisse la colonne 'speed' via une moyenne glissante temporelle centrée.
    - Utilise l'index temporel réel si possible (timestamp), sinon retombe sur hz.
    - Ne modifie aucune autre colonne.

    Paramètres
    ----------
    window_s : float
        largeur de la fenêtre en secondes (défaut 1.5 s).
    min_periods : int
        minimum d'échantillons pour calculer une moyenne (défaut 1).
    """
    name = "SpeedSmoother"

    def __init__(self, window_s: float = 1.5, min_periods: int = 1) -> None:
        self.window_s = float(window_s)
        self.min_periods = int(min_periods)

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        if "speed" not in df.columns:
            return Result(ok=False, message="speed manquante")

        out = df.copy()

        if "timestamp" in out.columns:
            ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.set_index(ts)
            # Rolling temporel centré
            win = f"{max(self.window_s, 0.05)}s"
            out["speed"] = (
                out["speed"]
                .rolling(win, center=True, min_periods=self.min_periods)
                .mean()
                .interpolate(method="time")
                .bfill()
                .ffill()
            )
            out = out.reset_index(drop=True)
        else:
            # Fallback hz si timestamp absent
            hz = int(ctx.meta.get("hz", 10))
            n = max(int(round(self.window_s * hz)), 1)
            out["speed"] = (
                pd.Series(out["speed"], dtype=float)
                .rolling(n, center=True, min_periods=self.min_periods)
                .mean()
                .bfill()
                .ffill()
                .to_numpy()
            )

        ctx.df = out
        return Result()
