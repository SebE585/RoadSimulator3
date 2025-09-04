# core2/stages/mid_stops_locker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from rs3_contracts.api import Result
from ..context import Context


def _bool_from_any(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    """Retourne une série bool à partir de colonnes potentiellement absentes."""
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce").fillna(0.0)
            return s.gt(0.0)
    return pd.Series(False, index=df.index)


def _get_stop_wait_mask(df: pd.DataFrame) -> pd.Series:
    """Construit un masque bool pour STOP/WAIT.
    - Priorité à la colonne 'event' == STOP/WAIT
    - Sinon fallback sur flags numériques (flag_stop/flag_wait/…)
    """
    if "event" in df.columns:
        ev = df["event"].astype(str).str.upper().fillna("")
        return ev.isin(["STOP", "WAIT"])
    # fallback: flags
    stop_flag = _bool_from_any(df, ("flag_stop", "stop_flag", "is_stop"))
    wait_flag = _bool_from_any(df, ("flag_wait", "wait_flag", "is_wait"))
    return stop_flag | wait_flag


def _groups_from_mask(mask: pd.Series) -> Iterable[Tuple[int, int]]:
    """Renvoie les intervalles [start_idx, end_idx] (inclus) des zones True contiguës."""
    if mask.empty:
        return []
    m = mask.to_numpy(dtype=bool)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []
    # détecter les ruptures de contiguïté
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return zip(starts.astype(int), ends.astype(int))


@dataclass
class MidStopsLocker:
    """
    Verrouille la vitesse à 0 autour des segments STOP/WAIT intermédiaires.
    Suppose une cadence régulière (post-SpeedSync). Utilise 'event' si dispo.

    Paramètres:
      head_s: extension (s) AVANT un STOP/WAIT
      tail_s: extension (s) APRÈS un STOP/WAIT
      blend_s: rampe (s) linéaire pour éviter le "mur" de vitesse (0 -> v)
      only_when_moving: si True, n’applique que si v>min_move_mps autour
      min_move_mps: vitesse considérée comme mouvement
    """
    head_s: float = 0.8
    tail_s: float = 0.8
    blend_s: float = 0.6
    only_when_moving: bool = False
    min_move_mps: float = 0.7

    name: str = "MidStopsLocker"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result((False, "df vide"))
        if "timestamp" not in df.columns:
            return Result((False, "timestamp manquant"))

        # Cadence (samples/s)
        hz = float(ctx.meta.get("hz", 10.0) or 10.0)
        hz = 10.0 if hz <= 0 else hz
        head_n = max(0, int(round(self.head_s * hz)))
        tail_n = max(0, int(round(self.tail_s * hz)))
        blend_n = max(0, int(round(self.blend_s * hz)))

        out = df.copy()

        # Colonne vitesse m/s (priorité à 'speed'), puis km/h -> m/s si besoin
        if "speed" in out.columns:
            v = pd.to_numeric(out["speed"], errors="coerce").astype(float).fillna(0.0).to_numpy()
        elif "speed_kmh" in out.columns:
            v = (pd.to_numeric(out["speed_kmh"], errors="coerce").astype(float).fillna(0.0) / 3.6).to_numpy()
        else:
            # pas de colonne vitesse -> rien à faire proprement
            return Result((True, "pas de colonne vitesse"))

        stop_mask = _get_stop_wait_mask(out)
        if not stop_mask.any():
            # Rien à verrouiller
            out["speed"] = np.maximum(v, 0.0)
            if "speed_kmh" in out.columns:
                out["speed_kmh"] = out["speed"] * 3.6
            ctx.df = out
            return Result((True, "OK"))

        n = len(v)
        v_new = v.copy()

        for s, e in _groups_from_mask(stop_mask):
            i0 = max(0, s - head_n)
            i1 = min(n - 1, e + tail_n)

            if self.only_when_moving:
                around = v[max(0, i0 - blend_n):min(n, i1 + blend_n + 1)]
                if not np.any(around > self.min_move_mps):
                    # zone déjà quasi nulle → inutile d’écraser
                    continue

            # 1) mettre à zéro la "zone lockée"
            v_new[i0:i1 + 1] = 0.0

            # 2) blending linéaire aux bords pour éviter une marche trop abrupte
            if blend_n > 0:
                # bord amont (avant i0)
                b0s = max(0, i0 - blend_n)
                b0e = i0  # non inclus
                if b0e - b0s > 0:
                    w = np.linspace(0.0, 1.0, b0e - b0s, endpoint=False)  # 0→(1-ε)
                    # on réduit progressivement la vitesse
                    v_new[b0s:b0e] = v[b0s:b0e] * (1.0 - w)

                # bord aval (après i1)
                b1s = i1 + 1
                b1e = min(n, i1 + 1 + blend_n)
                if b1e - b1s > 0:
                    w = np.linspace(0.0, 1.0, b1e - b1s, endpoint=False)
                    v_new[b1s:b1e] = v[b1s:b1e] * (w)

        # Applique et recalcule km/h si présent
        out["speed"] = np.maximum(v_new, 0.0)
        if "speed_kmh" in out.columns:
            out["speed_kmh"] = out["speed"] * 3.6

        ctx.df = out
        return Result((True, "OK"))