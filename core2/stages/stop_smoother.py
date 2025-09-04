# core2/stages/stop_smoother.py
from __future__ import annotations
import pandas as pd
from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

class StopSmoother:
    """
    Applique un lissage aux transitions de vitesse lors des arrêts/départs.
    """
    name = "StopSmoother"

    def __init__(self, v_in=0.25, t_in=2.0, v_out=0.6, t_out=2.5, lock_pos=True):
        self.v_in = v_in
        self.t_in = t_in
        self.v_out = v_out
        self.t_out = t_out
        self.lock_pos = lock_pos

    def run(self, ctx: Context) -> Result:
        df = ctx.df.copy()
        hz = ctx.meta.get("hz", 10)

        # repère les intervalles d'arrêt
        stop_mask = (df["speed"] == 0.0).astype(int)
        diff = stop_mask.diff().fillna(0)
        starts = df.index[diff == 1].tolist()
        ends = df.index[diff == -1].tolist()

        for s, e in zip(starts, ends):
            # ramp in
            n_in = int(self.t_in * hz)
            df.loc[max(0, s-n_in):s, "speed"] = \
                pd.Series([self.v_in * (i/n_in) for i in range(n_in+1)], index=range(max(0,s-n_in), s+1))
            # ramp out
            n_out = int(self.t_out * hz)
            df.loc[e:min(len(df)-1,e+n_out), "speed"] = \
                pd.Series([self.v_out * (1 - i/n_out) for i in range(n_out+1)], index=range(e,min(len(df),e+n_out+1)))

        ctx.df = df
        return Result()