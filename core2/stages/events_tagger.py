# core2/stages/events_tagger.py
from __future__ import annotations

import numpy as np
import pandas as pd

from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context


class EventsTagger:
    """
    Tague la colonne 'event' attendue par check_realism à partir des flags et de la cinématique :
      - 'stop' si flag_stop == 1
      - 'wait' si flag_wait == 1 (et pas déjà 'stop' sur la même ligne)
      - 'acceleration_initiale' si dv/dt > seuil dans la fenêtre de tête
      - 'freinage_final' si dv/dt < -seuil dans la fenêtre de queue

    Notes:
      - Aucun postulat 1 Hz : dv/dt est calculé avec les timestamps réels.
      - N’écrase pas une étiquette existante (sauf pour la tête/queue où un unique
        échantillon-clef est forcé).
    """
    name = "EventsTagger"

    def __init__(self,
                 dvdt_thr_mps2: float = 0.5,
                 head_window_s: float = 10.0,
                 tail_window_s: float = 10.0):
        self.dvdt_thr = float(dvdt_thr_mps2)
        self.head_window_s = float(head_window_s)
        self.tail_window_s = float(tail_window_s)

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result((False, "df vide"))
        if "timestamp" not in df.columns:
            return Result((False, "timestamp manquant"))

        out = df.copy()

        # Assure la présence de 'event'
        if "event" not in out.columns:
            out["event"] = ""
        else:
            # Cast explicite en str/objet pour permettre l'écriture simple
            out["event"] = out["event"].astype("object")

        # 1) Traduction flags -> events
        if "flag_stop" in out.columns:
            out.loc[out["flag_stop"] == 1, "event"] = np.where(
                out.loc[out["flag_stop"] == 1, "event"].eq(""), "stop", out.loc[out["flag_stop"] == 1, "event"]
            )
        if "flag_wait" in out.columns:
            # Ne remplace pas 'stop'
            mask_wait = (out["flag_wait"] == 1) & (out["event"].astype(str) == "")
            out.loc[mask_wait, "event"] = "wait"

        # 2) Détection dv/dt (accélération / freinage) sur tête/queue
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            # On ne bloque pas la pipeline : tag par flags seulement
            ctx.df = out
            return Result((True, "flags-only (timestamps invalides)"))

        ns = ts.astype("int64").to_numpy()
        tsec = (ns - ns[0]) / 1e9  # secondes relatives
        v = pd.to_numeric(out.get("speed", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # dv/dt robuste (gestion dt non uniforme)
        dvdt = np.gradient(v, tsec)
        if np.any(~np.isfinite(dvdt)):
            # remplace nan/inf par 0
            dvdt = np.nan_to_num(dvdt, nan=0.0, posinf=0.0, neginf=0.0)

        # Fenêtres
        head_mask = tsec <= self.head_window_s
        tail_mask = tsec >= (tsec[-1] - self.tail_window_s)

        # Accélération initiale
        if head_mask.any():
            dvdt_head = dvdt[head_mask]
            idx_local = int(np.argmax(dvdt_head))
            if dvdt_head[idx_local] > self.dvdt_thr:
                idx_global = int(np.where(head_mask)[0][idx_local])
                out.iat[idx_global, out.columns.get_loc("event")] = "acceleration_initiale"

        # Freinage final
        if tail_mask.any():
            dvdt_tail = dvdt[tail_mask]
            idx_local = int(np.argmin(dvdt_tail))
            if dvdt_tail[idx_local] < -self.dvdt_thr:
                idx_global = int(np.where(tail_mask)[0][idx_local])
                out.iat[idx_global, out.columns.get_loc("event")] = "freinage_final"

        ctx.df = out
        return Result((True, "OK"))
