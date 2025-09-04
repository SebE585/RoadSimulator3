# core2/stages/mid_stops_locker.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..contracts import Result
from ..context import Context

@dataclass
class MidStopsLocker:
    """
    Détecte les arrêts (par vitesse) et verrouille la vitesse à 0 pendant ces fenêtres.
    Optionnel: ré-écrit 'event' à STOP sur ces fenêtres pour harmoniser l'export.
    À placer après SpeedSync.

    Paramètres:
      speed_thr_mps: seuil m/s sous lequel on considère une immobilisation
      min_stop_s: durée minimale (en s) en continu sous le seuil pour valider un stop
      margin_before_s / margin_after_s: marges (s) ajoutées autour de la fenêtre validée
      write_event: si True, affecte event='STOP' sur les fenêtres verrouillées
      respect_existing_stop: si True, traite aussi les segments event==STOP (case-insensitive)
    """
    speed_thr_mps: float = 0.5
    min_stop_s: float = 10.0
    margin_before_s: float = 1.0
    margin_after_s: float = 1.0
    write_event: bool = True
    respect_existing_stop: bool = True

    name: str = "MidStopsLocker"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        if "speed" not in df.columns or "timestamp" not in df.columns:
            return Result(ok=False, message="colonnes manquantes (speed/timestamp)")

        hz = float(ctx.meta.get("hz", 10.0))
        if hz <= 0:
            hz = 10.0

        speed = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).to_numpy()
        is_low = speed < float(self.speed_thr_mps)

        # Option: inclure segments déjà taggés STOP
        if self.respect_existing_stop and "event" in df.columns:
            ev = df["event"].astype(str).str.lower()
            is_low = np.logical_or(is_low, ev.eq("stop").to_numpy())

        # regroupe les runs continus
        runs: list[tuple[int, int]] = []
        n = len(is_low)
        i = 0
        while i < n:
            if is_low[i]:
                j = i + 1
                while j < n and is_low[j]:
                    j += 1
                runs.append((i, j - 1))
                i = j
            else:
                i += 1

        lock_mask = np.zeros(n, dtype=bool)
        min_len = int(round(self.min_stop_s * hz))
        margin_before = int(round(self.margin_before_s * hz))
        margin_after = int(round(self.margin_after_s * hz))

        for a, b in runs:
            if (b - a + 1) >= max(1, min_len):
                aa = max(0, a - margin_before)
                bb = min(n - 1, b + margin_after)
                lock_mask[aa:bb + 1] = True

        if not lock_mask.any():
            # rien à faire
            return Result()

        # Verrouille la vitesse
        speed[lock_mask] = 0.0
        df = df.copy()
        df["speed"] = speed

        # (Option) ré-écrit event
        if self.write_event:
            ev = df.get("event")
            if ev is None:
                ev = pd.Series([""] * n, index=df.index, dtype="object")
            else:
                ev = ev.astype("object").fillna("")
            ev.loc[lock_mask] = "STOP"
            df["event"] = ev

        ctx.df = df
        return Result()