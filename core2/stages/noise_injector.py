# core2/stages/noise_injector.py
from __future__ import annotations
import numpy as np
from ..contracts import Result
from ..context import Context

class NoiseInjector:
    """
    Ajoute bruit gaussien réaliste aux colonnes acc/gyro.
    """
    name = "NoiseInjector"

    def __init__(self, sigma_acc=0.02, sigma_gyro=0.001):
        self.sigma_acc = sigma_acc
        self.sigma_gyro = sigma_gyro

    def run(self, ctx: Context) -> Result:
        df = ctx.df.copy()
        if "acc_x" in df: df["acc_x"] += np.random.normal(0,self.sigma_acc,len(df))
        if "acc_y" in df: df["acc_y"] += np.random.normal(0,self.sigma_acc,len(df))
        if "gyro_z" in df: df["gyro_z"] += np.random.normal(0,self.sigma_gyro,len(df))
        ctx.df = df
        return Result()