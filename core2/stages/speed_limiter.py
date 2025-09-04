# core2/stages/speed_limiter.py
from __future__ import annotations

import pandas as pd
import numpy as np

from ..contracts import Result
from ..context import Context


class SpeedLimiter:
    """
    Borne la colonne 'speed' à partir de 'target_speed' issue de l'enrichissement route.
    - Suppose que 'target_speed' est en km/h (valeur par défaut envoyée par RoadEnricher).
    - Convertit en m/s et applique une marge optionnelle.
    - Optionnellement, applique une rampe pour éviter un clip trop brutal.

    Paramètres
    ----------
    source_unit : str
        'kmh' (défaut) ou 'mps'
    margin_kmh : float
        marge ajoutée à la cible (km/h). Ex: 5.0 => +5 km/h.
    ramp_s : float
        si > 0, vitesse maximale de variation en m/s^2 autorisée pendant le clip (rampe descendante).
    """
    name = "SpeedLimiter"

    def __init__(self, source_unit: str = "kmh", margin_kmh: float = 3.0, ramp_s: float = 0.0) -> None:
        self.source_unit = source_unit.lower().strip()
        self.margin_kmh = float(margin_kmh)
        self.ramp_s = float(ramp_s)

    @staticmethod
    def _kmh_to_mps(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
        return (x * (1000.0/3600.0))

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")

        if "speed" not in df.columns:
            return Result(ok=False, message="speed manquante")

        if "target_speed" not in df.columns:
            # Rien à faire si pas de cible
            return Result()

        out = df.copy()

        # Convertit la cible en m/s
        tgt = pd.to_numeric(out["target_speed"], errors="coerce")
        if self.source_unit == "kmh":
            tgt_mps = self._kmh_to_mps(tgt)
        else:
            tgt_mps = tgt

        # Applique la marge (en km/h => convertir puis ajouter)
        if self.margin_kmh and self.margin_kmh != 0.0:
            tgt_mps = tgt_mps + self._kmh_to_mps(self.margin_kmh)

        # Base: clip vers la cible (réduction uniquement; jamais d'augmentation)
        v = pd.to_numeric(out["speed"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        tgt_arr = np.nan_to_num(tgt_mps.to_numpy(dtype=float), nan=np.inf)
        v_clip = np.minimum(v, tgt_arr)

        # Option rampe: limite la décélération max à ramp_s (m/s^2)
        if self.ramp_s and self.ramp_s > 0.0 and "timestamp" in out.columns:
            ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            ns = ts.astype("int64").to_numpy()
            tsec = (ns - ns[0]) / 1e9
            v_final = v_clip.copy()
            for i in range(1, len(v_final)):
                dt = max(tsec[i] - tsec[i-1], 1e-6)
                # borne inférieure imposée par la pente max de décélération depuis la valeur finale précédente
                lower_bound = max(v_final[i-1] - self.ramp_s * dt, 0.0)
                # on respecte la cible v_clip, mais on ne descend pas plus vite que la rampe
                if v_final[i] < lower_bound:
                    v_final[i] = lower_bound
        else:
            v_final = v_clip

        # Clamp final à [0, +inf)
        v_final = np.maximum(v_final, 0.0)

        out["speed"] = v_final
        ctx.df = out
        return Result()
