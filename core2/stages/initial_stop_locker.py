# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..contracts import Result
from ..context import Context


@dataclass
class InitialStopLocker:
    """
    Fige le début de la trajectoire sur `head_s` secondes :
      - met la vitesse à 0 (colonne `speed`)
      - optionnellement, verrouille la position (lat/lon) au tout premier point
      - remet à 0 les signaux IMU sur la même fenêtre (si présents)

    Ce stage est le pendant "départ" de FinalStopLocker.
    Il est utile pour satisfaire le test 'start_zero' du validateur.

    Paramètres
    ----------
    head_s : float
        Durée (s) à figer dès le début. Exemple: 2.0.
    lock_pos : bool
        Si True, recopie lat/lon du premier échantillon sur la fenêtre.
    start_zero_thr_mps : float
        Seuil m/s pour considérer le départ déjà quasi nul. Si la médiane
        de vitesse sur la fenêtre est < seuil, on force (=0). Sinon on n'agit pas.
        Mettre None pour forcer inconditionnellement la mise à 0.
    imu_cols : Sequence[str]
        Colonnes IMU à remettre à 0 si elles existent.
    """

    head_s: float = 2.0
    lock_pos: bool = True
    start_zero_thr_mps: float | None = 0.6
    imu_cols: Sequence[str] = (
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )

    name: str = "InitialStopLocker"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        if "timestamp" not in df.columns:
            return Result(ok=False, message="timestamp manquant")
        if "speed" not in df.columns:
            # On ne fait rien si la vitesse n'existe pas encore
            return Result(ok=True, message="speed absente — InitialStopLocker ignoré")

        out = df.copy()

        # Cadence nominale attendue (déjà fixée dans SpeedSync, sinon cfg.sim.hz)
        hz = float(ctx.meta.get("hz", ctx.cfg.get("sim", {}).get("hz", 10)))
        hz = 10.0 if not np.isfinite(hz) or hz <= 0 else hz

        # Taille de fenêtre en échantillons, bornée à la taille du DF
        n = max(0, int(round(self.head_s * hz)))
        if n <= 0:
            ctx.df = out
            return Result()  # rien à faire
        n = min(n, len(out))

        # Détermine si on doit intervenir (fenêtre déjà quasi nulle ?)
        sp = pd.to_numeric(out["speed"], errors="coerce").fillna(0.0).to_numpy()
        apply_lock = True
        if self.start_zero_thr_mps is not None:
            head_med = float(np.nanmedian(sp[:n])) if n > 0 else 0.0
            apply_lock = head_med < float(self.start_zero_thr_mps)

        if apply_lock:
            # 1) Vitesse = 0 sur la fenêtre
            out.loc[: n - 1, "speed"] = 0.0

            # 2) Position verrouillée sur le tout premier point
            if self.lock_pos and {"lat", "lon"}.issubset(out.columns):
                lat0 = out.at[0, "lat"]
                lon0 = out.at[0, "lon"]
                out.loc[: n - 1, "lat"] = lat0
                out.loc[: n - 1, "lon"] = lon0

            # 3) IMU remis à 0 si colonnes présentes
            for c in self.imu_cols:
                if c in out.columns:
                    out.loc[: n - 1, c] = 0.0

            # 4) (optionnel) Tag d'événement
            if "event" in out.columns:
                out.loc[: n - 1, "event"] = out.loc[: n - 1, "event"].fillna("").astype(str)
                # n'écrase pas un éventuel tag existant, concatène proprement
                out.loc[: n - 1, "event"] = out.loc[: n - 1, "event"].replace("", "START_LOCK").where(
                    out.loc[: n - 1, "event"] != "", out.loc[: n - 1, "event"]
                )

            ctx.meta["initial_stop_locked_n"] = int(n)
        else:
            ctx.meta["initial_stop_locked_n"] = 0

        ctx.df = out
        return Result()
