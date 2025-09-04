from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from rs3_contracts.api import Result
from ..context import Context


@dataclass
class StopSmoother:
    """
    Lisse les transitions de vitesse autour des arrêts (zone speed==0).

    - Amortit l'approche du STOP en ramenant progressivement la vitesse vers 0
      dans une fenêtre de tête `t_in`.
    - Ré-accélère progressivement après le STOP vers `v_out` sur `t_out`.
    - Option `lock_pos` pour figer lat/lon durant l'arrêt strict (entre s et e).

    Notes
    -----
    * Utilise la cadence `ctx.meta['hz']` (défaut 10) pour convertir secondes → samples.
    * Robuste aux cas aux bornes (début/fin de série stoppée).
    * Ne crée pas de nouvelles colonnes.
    """

    v_in: float = 0.25   # vitesse résiduelle à l'entrée du STOP (m/s)
    t_in: float = 2.0    # durée de rampe avant STOP (s)
    v_out: float = 0.6   # vitesse cible à la sortie immédiate du STOP (m/s)
    t_out: float = 2.5   # durée de rampe après STOP (s)
    lock_pos: bool = True

    name: str = "StopSmoother"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result((False, "df vide"))
        if "speed" not in df.columns:
            return Result((False, "speed manquante"))

        try:
            out = df.copy()
            hz = float(ctx.meta.get("hz", 10) or 10)
            hz = 10.0 if hz <= 0 else hz

            # Masque STOP strict
            stop_mask = pd.to_numeric(out["speed"], errors="coerce").fillna(0.0).eq(0.0)

            if not stop_mask.any():
                # Rien à lisser
                ctx.df = out
                return Result((True, "no stop zones"))

            # Détection des intervalles contigus [s, e] (inclus)
            m = stop_mask.to_numpy(dtype=bool)
            idx_true = np.flatnonzero(m)
            intervals: list[tuple[int, int]] = []
            if idx_true.size:
                brk = np.flatnonzero(np.diff(idx_true) > 1)
                starts = np.r_[idx_true[0], idx_true[brk + 1]]
                ends = np.r_[idx_true[brk], idx_true[-1]]
                intervals = [(int(s), int(e)) for s, e in zip(starts, ends)]

            # Fenêtres (échantillons)
            n_in = max(1, int(round(self.t_in * hz)))
            n_out = max(1, int(round(self.t_out * hz)))

            v = pd.to_numeric(out["speed"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            for s, e in intervals:
                # 1) Rampe avant (i0..s) : amener progressivement v -> ~0
                i0 = max(0, s - n_in)
                if s - i0 > 0:
                    L = s - i0
                    w = np.linspace(1.0, 0.0, L + 1)  # 1 → 0 sur [i0..s]
                    # garde un résiduel v_in en fin de rampe (evite mur)
                    target = self.v_in * np.ones(L + 1, dtype=float)
                    ramp = v[i0:s + 1] * w
                    v[i0:s + 1] = np.minimum(np.maximum(ramp, target), v[i0:s + 1])

                # 2) Zone d'arrêt : clamp à 0
                v[s:e + 1] = 0.0

                # 3) Rampe sortie (e..j1) : 0 → v_out
                j1 = min(len(v) - 1, e + n_out)
                L = j1 - e
                if L > 0:
                    w = np.linspace(0.0, 1.0, L + 1)  # 0 → 1 sur [e..j1]
                    ramp = self.v_out * w
                    # impose une rampe mini sans dépasser la vitesse préexistante
                    v[e:j1 + 1] = np.clip(v[e:j1 + 1], ramp, np.maximum(v[e:j1 + 1], ramp))

                # 4) Option: verrou des positions pendant l'arrêt strict
                if self.lock_pos and {"lat", "lon"}.issubset(out.columns):
                    try:
                        lat0 = float(pd.to_numeric(out.at[s, "lat"], errors="coerce"))
                        lon0 = float(pd.to_numeric(out.at[s, "lon"], errors="coerce"))
                        out.loc[out.index[s:e + 1], "lat"] = lat0
                        out.loc[out.index[s:e + 1], "lon"] = lon0
                    except Exception:
                        pass

            out["speed"] = np.maximum(v, 0.0)
            if "speed_kmh" in out.columns:
                out["speed_kmh"] = out["speed"] * 3.6

            # Expose intervals pour debug dans le runner
            ctx.artifacts["stop_smoother_intervals"] = intervals

            ctx.df = out
            return Result((True, "OK"))
        except Exception as e:
            return Result((False, f"StopSmoother error: {e}"))