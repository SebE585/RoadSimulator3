# core2/stages/terrain_imu_synth.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

G = 9.80665  # m/s²

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))

def _bearing_rad(lat1, lon1, lat2, lon2):
    # cap géodésique en radians (0 = Nord, +Est)
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)
    y = np.sin(Δλ) * np.cos(φ2)
    x = np.cos(φ1)*np.sin(φ2) - np.sin(φ1)*np.cos(φ2)*np.cos(Δλ)
    return np.arctan2(y, x)

def _movemean(x: np.ndarray, win_n: int) -> np.ndarray:
    if win_n <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(win_n, min_periods=1, center=True).mean().to_numpy()

@dataclass
class TerrainIMUSynth:
    """
    Synthèse IMU depuis géo + vitesse + pente.

    - Si slope_percent est présent → priorité.
    - Sinon, si altitude_m est présent → calcule la pente (dz/ds).
    - Sinon → acc_x = dv/dt (sans gravité), acc_y/gyro_z via cap, gyro_x/acc_z = 0.
    """
    smooth_window_s: float = 1.0       # lissage (s)
    include_gravity: bool = False      # acc_x inclut g*sin(theta) si True
    use_slope_percent: bool = True     # sinon dérive grade depuis altitude_m
    clamp_ax_mps2: float | None = 5.0  # limite douce |dv/dt| (~0.5 g) ; None pour désactiver

    name: str = "TerrainIMUSynth"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        need = {"timestamp", "lat", "lon", "speed"}
        if not need.issubset(df.columns):
            return Result(ok=False, message=f"colonnes manquantes: {sorted(need - set(df.columns))}")

        out = df.copy()
        idx = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if idx.isna().any():
            return Result(ok=False, message="timestamps invalides")
        out = out.set_index(idx).sort_index()
        if "timestamp" in out.columns:
            out = out.drop(columns=["timestamp"])

        # Cadence & fenêtre de lissage
        hz = float(ctx.meta.get("hz", 10))
        dt_nom = 1.0 / max(hz, 1.0)
        win_n = max(int(round(self.smooth_window_s * hz)), 1)

        # --- Signaux de base ---
        lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
        v   = pd.to_numeric(out["speed"], errors="coerce").to_numpy(dtype=float)  # m/s

        # Axe temps réel (robuste aux petites irrégularités)
        ns0 = out.index.asi8
        tsec = (ns0 - ns0[0]) / 1e9
        dt_s = np.diff(tsec, prepend=tsec[0])
        pos = dt_s > 0
        if not pos.any():
            dt_s[:] = dt_nom
        else:
            median_dt = float(np.median(dt_s[pos]))
            dt_s[~pos] = median_dt

        # --- Cap & taux de lacet (gyro_z) ---
        bearing = np.zeros_like(v)
        if len(v) > 1:
            bearing[1:] = _bearing_rad(lat[:-1], lon[:-1], lat[1:], lon[1:])
        psi = np.unwrap(bearing)
        psi = _movemean(psi, win_n)
        gyro_z = np.gradient(psi, dt_s)              # rad/s

        # --- Pente θ (rad) ---
        if self.use_slope_percent and "slope_percent" in out.columns:
            grade = pd.to_numeric(out["slope_percent"], errors="coerce").to_numpy(dtype=float) / 100.0
            grade = np.nan_to_num(grade, nan=0.0, posinf=0.0, neginf=0.0)
        elif "altitude_m" in out.columns:
            z = pd.to_numeric(out["altitude_m"], errors="coerce").to_numpy(dtype=float)
            ds = np.zeros_like(v)
            if len(v) > 1:
                ds[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
                ds[ds <= 0] = np.nan
            dz = np.gradient(z, edge_order=2)
            grade = dz / np.where(ds > 0, ds, np.nan)
            grade = np.nan_to_num(grade, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            grade = np.zeros_like(v)

        theta = np.arctan(grade)
        theta = _movemean(theta, win_n)
        gyro_y = np.gradient(theta, dt_s)            # rad/s (variation de pente)
        gyro_x = np.zeros_like(gyro_y)               # pas d’info de roulis

        # --- Accélérations spécifiques ---
        v_s = _movemean(v, win_n)
        acc_x = np.gradient(v_s, dt_s)               # m/s² (longitudinal)
        if self.clamp_ax_mps2:
            amax = abs(float(self.clamp_ax_mps2))
            acc_x = np.clip(acc_x, -amax, amax)

        if self.include_gravity:
            acc_x = acc_x + G * np.sin(theta)        # projette la gravité si demandé

        acc_y = v_s * gyro_z                          # m/s² (centripète ~ v * dψ/dt)
        acc_z = np.zeros_like(acc_x)                  # pas de bosses synthétisées

        # Nettoyage NaN/inf et cast
        for arr in (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        out["acc_x"]  = acc_x.astype("float32")
        out["acc_y"]  = acc_y.astype("float32")
        out["acc_z"]  = acc_z.astype("float32")
        out["gyro_x"] = gyro_x.astype("float32")
        out["gyro_y"] = gyro_y.astype("float32")
        out["gyro_z"] = gyro_z.astype("float32")

        # Rétablit timestamp et renvoie tout le DF (on ne casse rien)
        out = out.reset_index()
        ctx.df = out

        return Result()