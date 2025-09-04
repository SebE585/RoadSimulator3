# core2/stages/noise_injector.py
from __future__ import annotations

import numpy as np

from rs3_contracts.api import Result
from ..context import Context


class NoiseInjector:
    """
    Ajoute un bruit blanc gaussien aux colonnes IMU si elles existent.

    - Accélérations: acc_x, acc_y, acc_z
    - Gyros: gyro_x, gyro_y, gyro_z

    Paramètres par défaut (écarts-types):
      - sigma_acc  : 0.02  (m/s²)
      - sigma_gyro : 0.001 (rad/s)

    Les paramètres peuvent être surchargés par la config YAML via:
    cfg.noise_injector: { sigma_acc: ..., sigma_gyro: ..., seed: ... }

    Notes:
      - Le stage n'échoue jamais si les colonnes sont absentes: il passe simplement.
      - Seed optionnelle pour la reproductibilité.
    """
    name = "NoiseInjector"

    def __init__(self, sigma_acc: float = 0.02, sigma_gyro: float = 0.001) -> None:
        self.sigma_acc = float(sigma_acc)
        self.sigma_gyro = float(sigma_gyro)

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            # Stage non-bloquant
            return Result((True, "df vide — skip"))

        out = df.copy()

        # surcharge éventuelle depuis la config
        cfg = {}
        try:
            cfg = ctx.cfg.get("noise_injector", {}) if isinstance(ctx.cfg, dict) else {}
        except Exception:
            cfg = {}

        sigma_acc = float(cfg.get("sigma_acc", self.sigma_acc))
        sigma_gyro = float(cfg.get("sigma_gyro", self.sigma_gyro))
        seed = cfg.get("seed", None)

        rng = np.random.default_rng(seed)

        def _add_noise(col: str, sigma: float) -> None:
            if col in out.columns:
                try:
                    n = len(out)
                    noise = rng.normal(0.0, sigma, size=n)
                    out[col] = (np.asarray(out[col], dtype=float) + noise).astype("float32")
                except Exception:
                    # on ne bloque pas le pipeline pour une colonne problématique
                    pass

        # Accélérations
        _add_noise("acc_x", sigma_acc)
        _add_noise("acc_y", sigma_acc)
        _add_noise("acc_z", sigma_acc)

        # Gyros
        _add_noise("gyro_x", sigma_gyro)
        _add_noise("gyro_y", sigma_gyro)
        _add_noise("gyro_z", sigma_gyro)

        ctx.df = out
        return Result((True, "OK"))