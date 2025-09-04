# core2/stages/final_stop_locker.py
from __future__ import annotations
import pandas as pd
import numpy as np
from ..contracts import Result
from ..context import Context

class FinalStopLocker:
    """
    Verrouille la fin de trajectoire (dernières `tail_s` secondes) sur la
    position finale afin de garantir une vitesse terminale nulle.
    - Utilise l'axe temps réel ('timestamp').
    - Ne crée pas de nouveaux points, ne change pas la longueur.

    Paramètres
    ----------
    tail_s : float
        Durée (en secondes) à figer en fin de trajectoire. Défaut: 8.0
    """
    name = "FinalStopLocker"

    def __init__(self, tail_s: float = 8.0) -> None:
        self.tail_s = float(tail_s)

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        if "timestamp" not in df.columns:
            return Result(ok=False, message="timestamp manquant")

        out = df.copy()
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            return Result(ok=False, message="timestamps invalides")

        tsec = (ts.astype("int64").to_numpy() - ts.astype("int64").to_numpy()[0]) / 1e9
        if len(tsec) < 2:
            return Result()

        # Indices de la fenêtre terminale
        t_end = tsec[-1]
        lock_from = t_end - max(self.tail_s, 0.0)
        mask_tail = tsec >= lock_from

        # Coordonnée finale
        latN = out["lat"].iloc[-1]
        lonN = out["lon"].iloc[-1]

        # Fige la géométrie en fin de trajectoire
        out.loc[mask_tail, "lat"] = latN
        out.loc[mask_tail, "lon"] = lonN

        # Optionnel: si 'flag_stop' existe, on marque la fin comme stop
        if "flag_stop" in out.columns:
            out.loc[mask_tail, "flag_stop"] = 1
        if "event" in out.columns:
            # N'écrase pas un tag existant ; marque au moins un point comme 'freinage_final'
            idx_tail = np.where(mask_tail)[0]
            if len(idx_tail) > 0:
                out.iat[idx_tail[0], out.columns.get_loc("event")] = "freinage_final"

        ctx.df = out
        return Result()

