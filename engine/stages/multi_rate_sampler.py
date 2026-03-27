# engine/stages/multi_rate_sampler.py
"""
Stage — Multi-rate sampler.

Le pipeline RS3 tourne en interne à 10 Hz (physique).
Ce stage rééchantillonne la sortie pour refléter les fréquences
réelles des capteurs :
  - GPS  : 1 Hz à 10 Hz  (défaut 1 Hz)
  - IMU  : 5 Hz à 10 Hz  (acc toujours, gyro optionnel)

Les points non-GPS gardent lat/lon/speed mais marqués avec
gps_valid=False. Les colonnes gyro sont supprimées si désactivées.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rs3_contracts.api import Result

logger = logging.getLogger(__name__)

# Colonnes GPS (NaN entre les ticks GPS)
# Inclut les deux noms possibles (RS3 interne = speed, Telemachus = speed_mps)
GPS_COLS = ["lat", "lon", "speed", "speed_mps", "heading", "heading_deg",
            "altitude_m", "slope_percent"]

# Colonnes gyro (supprimées si gyro désactivé)
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]

# Colonnes acc (NaN entre les ticks IMU)
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
IMU_COLS = ACC_COLS + GYRO_COLS


@dataclass
class MultiRateSampler:
    """Rééchantillonne GPS et IMU à des fréquences différentes."""

    name: str = "multi_rate_sampler"
    gps_hz: float = 1.0
    imu_hz: float = 10.0
    gyro_enabled: bool = True

    def run(self, ctx) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(True, "skip — empty df")

        pipeline_hz = float(ctx.meta.get("hz", 10))

        # ── GPS downsampling ─────────────────────────────────────────
        gps_hz = min(self.gps_hz, pipeline_hz)
        gps_stride = max(1, int(round(pipeline_hz / gps_hz)))

        if gps_stride > 1:
            gps_mask = np.zeros(len(df), dtype=bool)
            gps_mask[::gps_stride] = True
            # Toujours garder le dernier point
            gps_mask[-1] = True

            for col in GPS_COLS:
                if col in df.columns:
                    df.loc[~gps_mask, col] = np.nan

            df["gps_valid"] = gps_mask
            logger.info("GPS: %d Hz → stride=%d (%d/%d points)",
                        int(gps_hz), gps_stride, gps_mask.sum(), len(df))
        else:
            df["gps_valid"] = True

        # ── IMU downsampling ─────────────────────────────────────────
        imu_hz = min(self.imu_hz, pipeline_hz)
        imu_stride = max(1, int(round(pipeline_hz / imu_hz)))

        if imu_stride > 1:
            imu_mask = np.zeros(len(df), dtype=bool)
            imu_mask[::imu_stride] = True

            for col in IMU_COLS:
                if col in df.columns:
                    df.loc[~imu_mask, col] = np.nan

            logger.info("IMU: %d Hz → stride=%d", int(imu_hz), imu_stride)

        # ── Gyro disable ─────────────────────────────────────────────
        if not self.gyro_enabled:
            for col in GYRO_COLS:
                if col in df.columns:
                    df[col] = np.nan
            logger.info("Gyro désactivé — colonnes gyro NaN")

        # ── Store rates in meta ──────────────────────────────────────
        ctx.set_meta("gps_hz", gps_hz)
        ctx.set_meta("imu_hz", imu_hz)
        ctx.set_meta("gyro_enabled", self.gyro_enabled)

        ctx.df = df
        return Result(True, f"GPS={gps_hz}Hz IMU={imu_hz}Hz gyro={'on' if self.gyro_enabled else 'off'}")
