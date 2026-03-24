# core2/stages/device_rotator.py
"""
Stage — Device Rotation Simulator.

Simule un capteur monté avec un angle arbitraire dans le véhicule.
Applique une rotation Euler (roll, pitch, yaw) aux colonnes acc et gyro
pour tester ensuite les algorithmes de redressement (calibration statique).

Convention :
  - roll  (φ) : rotation autour de l'axe X (longitudinal)
  - pitch (θ) : rotation autour de l'axe Y (latéral)
  - yaw   (ψ) : rotation autour de l'axe Z (vertical)

Ordre d'application : R = Rz(ψ) · Ry(θ) · Rx(φ)  (intrinsic ZYX)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from rs3_contracts.api import Result

logger = logging.getLogger(__name__)


def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Matrice de rotation 3×3 intrinsèque ZYX depuis des angles en degrés."""
    phi = np.radians(roll_deg)
    theta = np.radians(pitch_deg)
    psi = np.radians(yaw_deg)

    cx, sx = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(psi), np.sin(psi)

    # Rz · Ry · Rx
    R = np.array([
        [cz * cy,  cz * sy * sx - sz * cx,  cz * sy * cx + sz * sx],
        [sz * cy,  sz * sy * sx + cz * cx,  sz * sy * cx - cz * sx],
        [-sy,      cy * sx,                  cy * cx],
    ])
    return R


@dataclass
class DeviceRotator:
    """Applique une rotation fixe aux vecteurs acc et gyro."""

    name: str = "device_rotator"
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def run(self, ctx) -> Result:
        if self.roll_deg == 0.0 and self.pitch_deg == 0.0 and self.yaw_deg == 0.0:
            return Result(True, "skip — rotation nulle")

        df = ctx.df
        if df is None or df.empty:
            return Result(True, "skip — empty df")

        R = _rotation_matrix(self.roll_deg, self.pitch_deg, self.yaw_deg)

        # ── Rotation accéléromètre ───────────────────────────────────
        acc_cols = ["acc_x", "acc_y", "acc_z"]
        if all(c in df.columns for c in acc_cols):
            acc = df[acc_cols].to_numpy(dtype=float)  # (N, 3)
            acc_rot = (R @ acc.T).T                    # (N, 3)
            df["acc_x"] = acc_rot[:, 0]
            df["acc_y"] = acc_rot[:, 1]
            df["acc_z"] = acc_rot[:, 2]

        # ── Rotation gyroscope ───────────────────────────────────────
        gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
        if all(c in df.columns for c in gyro_cols):
            gyro = df[gyro_cols].to_numpy(dtype=float)
            gyro_rot = (R @ gyro.T).T
            df["gyro_x"] = gyro_rot[:, 0]
            df["gyro_y"] = gyro_rot[:, 1]
            df["gyro_z"] = gyro_rot[:, 2]

        # Store rotation in meta for downstream reference
        ctx.set_meta("device_rotation_deg", {
            "roll": self.roll_deg,
            "pitch": self.pitch_deg,
            "yaw": self.yaw_deg,
        })

        ctx.df = df
        logger.info("Device rotation applied: roll=%.1f° pitch=%.1f° yaw=%.1f°",
                     self.roll_deg, self.pitch_deg, self.yaw_deg)
        return Result(True, f"roll={self.roll_deg}° pitch={self.pitch_deg}° yaw={self.yaw_deg}°")
