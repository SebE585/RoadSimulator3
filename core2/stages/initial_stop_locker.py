# core2/stages/initial_stop_locker.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from rs3_contracts.api import Result
from ..context import Context


@dataclass
class InitialStopLocker:
    """
    Verrouille le début du trajet à vitesse nulle pendant `head_s`.
    - Option lock_pos: recopie lat/lon du 1er point → vitesse recalculée = 0
    - Si start_zero_thr_mps est donné, n'applique le verrou que si la médiane
      des vitesses dans la fenêtre de tête est < ce seuil.
      Si None → verrou inconditionnel.
    """
    head_s: float = 2.0
    lock_pos: bool = True
    start_zero_thr_mps: float | None = None

    name: str = "InitialStopLocker"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result((False, "df vide"))

        if "timestamp" not in df.columns or "speed" not in df.columns:
            return Result((False, "timestamp/speed manquants"))

        hz = float(ctx.meta.get("hz", 10.0))
        n_head = max(1, int(round(self.head_s * hz)))

        out = df.copy()
        # Critère de déclenchement si demandé
        do_lock = True
        if self.start_zero_thr_mps is not None:
            sp = pd.to_numeric(out["speed"], errors="coerce").fillna(0.0).to_numpy()
            head_med = float(np.nanmedian(sp[:n_head])) if len(sp) else 0.0
            do_lock = head_med < float(self.start_zero_thr_mps)

        if not do_lock:
            return Result((True, "OK"))  # rien à faire

        # 1) vitesse = 0 en tête
        if "speed" in out.columns:
            out.loc[out.index[:n_head], "speed"] = 0.0

        # 2) option lock_pos : recopie lat/lon du 1er point => plus de déplacement
        if self.lock_pos and {"lat", "lon"}.issubset(out.columns) and len(out) >= 1:
            lat0 = float(pd.to_numeric(out.at[out.index[0], "lat"], errors="coerce"))
            lon0 = float(pd.to_numeric(out.at[out.index[0], "lon"], errors="coerce"))
            out.loc[out.index[:n_head], "lat"] = lat0
            out.loc[out.index[:n_head], "lon"] = lon0

        # 3) event: tagger STOP si présent
        if "event" in out.columns:
            ev = out["event"].astype("object").fillna("")
            ev.loc[ev.index[:n_head]] = np.where(ev.loc[ev.index[:n_head]] == "", "STOP", ev.loc[ev.index[:n_head]])
            out["event"] = ev
        else:
            out["event"] = ""
            out.loc[out.index[:n_head], "event"] = "STOP"

        ctx.df = out
        return Result((True, "OK"))