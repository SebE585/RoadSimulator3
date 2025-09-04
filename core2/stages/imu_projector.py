# core2/stages/imu_projector.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

G = 9.80665  # m/s²

logger = logging.getLogger(__name__)

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
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
    return np.arctan2(y, x)

def _movemean(x: np.ndarray, win_n: int) -> np.ndarray:
    if win_n <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(win_n, min_periods=1, center=True).mean().to_numpy()

@dataclass
class IMUProjector:
    r"""
    Synthèse IMU depuis géo + vitesse + pente.

    - acc_y prend en compte les virages via \dot{ψ}.
    - acc_z ≈ g·cos(θ) si `include_gravity=True`.
    - Ajoute des garde‑fous numériques pour éviter les valeurs infinies/astronomiques sur gyro_z/acc_y.
    - Publie un diagnostic de cohérence courbure dans `ctx.artifacts['imu_coherence']`.
    """
    smooth_window_s: float = 1.0
    include_gravity: bool = True
    use_slope_percent: bool = True
    clamp_ax_mps2: float | None = 5.0
    min_speed_for_lateral: float = 0.3
    min_turn_radius_m: float = 8.0          # rayon mini plausible → borne gyro_z/acc_y
    pitch_rate_max_rad_s: float = 0.7       # borne douce pour gyro_y (variation pente)
    name: str = "IMUProjector"

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

        # Paramètres (priorité YAML cfg > valeurs de l'instance)
        cfg_stage = {}
        try:
            cfg_stage = ctx.cfg.get("imu_projector", {}) if isinstance(ctx.cfg, dict) else {}
        except Exception:
            cfg_stage = {}

        smooth_window_s = float(cfg_stage.get("smooth_window_s", self.smooth_window_s))
        include_gravity = bool(cfg_stage.get("include_gravity", self.include_gravity))
        use_slope_percent = bool(cfg_stage.get("use_slope_percent", self.use_slope_percent))
        clamp_ax_mps2 = cfg_stage.get("clamp_ax_mps2", self.clamp_ax_mps2)
        clamp_ax_mps2 = float(clamp_ax_mps2) if clamp_ax_mps2 is not None else None
        min_speed_for_lateral = float(cfg_stage.get("min_speed_for_lateral", self.min_speed_for_lateral))
        min_turn_radius_m = float(cfg_stage.get("min_turn_radius_m", self.min_turn_radius_m))
        pitch_rate_max_rad_s = float(cfg_stage.get("pitch_rate_max_rad_s", self.pitch_rate_max_rad_s))

        # Petit log de configuration effective
        logger.info(
            "[IMU] cfg: smooth_window_s=%.2f, include_gravity=%s, use_slope_percent=%s, clamp_ax=%.2f, v_min=%.2f m/s, Rmin=%.1f m, pitch_rate_max=%.2f rad/s",
            smooth_window_s,
            include_gravity,
            use_slope_percent,
            (clamp_ax_mps2 if clamp_ax_mps2 is not None else float('nan')),
            min_speed_for_lateral,
            min_turn_radius_m,
            pitch_rate_max_rad_s,
        )

        # Cadence & fenêtre de lissage
        hz = float(ctx.meta.get("hz", 10))
        dt_nom = 1.0 / max(hz, 1.0)
        win_n = max(int(round(smooth_window_s * hz)), 1)

        # Signaux de base
        lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
        v   = pd.to_numeric(out["speed"], errors="coerce").to_numpy(dtype=float)

        # Axe temps robuste → dt_s strictement positif et fini
        ns = out.index.asi8.astype(np.float64)
        dt_s = np.diff(ns, prepend=ns[0]) / 1e9
        pos = np.isfinite(dt_s) & (dt_s > 0)
        if not pos.any():
            dt_s[:] = dt_nom
        else:
            med = float(np.median(dt_s[pos]))
            dt_s[~pos] = med
            dt_s = np.maximum(dt_s, max(1e-3, 0.2 * dt_nom))
        # temps cumulé strictement croissant
        tsec = np.cumsum(dt_s)
        tsec[0] = 0.0

        # Seuil de vitesse pour activer les grandeurs latérales
        v_min = float(min_speed_for_lateral)
        moving = v > v_min

        # Cap & gyro_z
        bearing = np.zeros_like(v)
        if len(v) > 1:
            bearing[1:] = _bearing_rad(lat[:-1], lon[:-1], lat[1:], lon[1:])
        psi = np.unwrap(bearing)
        psi = _movemean(psi, win_n)
        # utiliser l'axe temps robuste
        gyro_z = np.gradient(psi, tsec, edge_order=1)   # rad/s
        gyro_z[~moving] = 0.0

        # Pente θ
        if use_slope_percent and "slope_percent" in out.columns:
            grade = pd.to_numeric(out["slope_percent"], errors="coerce").to_numpy(dtype=float) / 100.0
            grade = np.nan_to_num(grade, nan=0.0, posinf=0.0, neginf=0.0)
        elif "altitude_m" in out.columns:
            z = pd.to_numeric(out["altitude_m"], errors="coerce").to_numpy(dtype=float)
            ds = np.zeros_like(v)
            if len(v) > 1:
                ds[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
                ds[ds <= 0] = np.nan
            dz = np.gradient(z, edge_order=1)
            grade = dz / np.where(ds > 0, ds, np.nan)
            grade = np.nan_to_num(grade, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            grade = np.zeros_like(v)

        theta = np.arctan(grade)
        theta = _movemean(theta, win_n)
        gyro_y = np.gradient(theta, tsec, edge_order=1)
        # Borne douce sur gyro_y (évite explosions rares)
        gyro_y = np.clip(gyro_y, -pitch_rate_max_rad_s, pitch_rate_max_rad_s)
        gyro_x = np.zeros_like(gyro_y)

        # Accélérations spécifiques
        v_s = _movemean(v, win_n)
        acc_x = np.gradient(v_s, tsec, edge_order=1)
        if clamp_ax_mps2:
            amax = abs(float(clamp_ax_mps2))
            acc_x = np.clip(acc_x, -amax, amax)
        if include_gravity:
            acc_x = acc_x + G * np.sin(theta)

        # Garde‑fou courbure: |psi_dot| <= v / R_min
        Rmin = max(1.0, float(min_turn_radius_m))
        psi_dot_max = np.where(v_s > 0, v_s / Rmin, 0.0)
        # clamp gyro_z puis acc_y dérivé
        gyro_z = np.clip(gyro_z, -psi_dot_max, psi_dot_max)
        acc_y = v_s * gyro_z
        acc_y[~moving] = 0.0

        # acc_z spécifique
        if include_gravity:
            acc_z = (G * np.cos(theta)).astype(np.float64)
        else:
            acc_z = np.zeros_like(acc_x)

        # Nettoyage NaN/inf
        for arr in (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Diagnostic cohérence
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.where(v_s > v_min, gyro_z / np.where(v_s > 0, v_s, np.nan), 0.0)
            R = np.where(np.abs(gyro_z) > 0, v_s / np.where(np.abs(gyro_z) > 0, np.abs(gyro_z), np.nan), np.nan)
        ay_curv = v_s**2 * kappa
        resid = acc_y - ay_curv
        mask = np.isfinite(resid) & moving
        if mask.any():
            rmse = float(np.sqrt(np.mean(resid[mask]**2)))
            p95  = float(np.percentile(np.abs(resid[mask]), 95))
            bad_ratio = float(np.mean(np.abs(resid[mask]) > 0.2))
            R_valid = R[mask & np.isfinite(R)]
            R_p50 = float(np.nanmedian(R_valid)) if R_valid.size else float('nan')
            R_p10 = float(np.nanpercentile(R_valid, 10)) if R_valid.size else float('nan')
            R_p90 = float(np.nanpercentile(R_valid, 90)) if R_valid.size else float('nan')
        else:
            rmse = p95 = bad_ratio = R_p50 = R_p10 = R_p90 = float('nan')
        ctx.artifacts["imu_coherence"] = {
            "rmse_ay_vs_vpsi": rmse,
            "p95_abs_resid": p95,
            "bad_ratio_gt_0p2": bad_ratio,
            "R_median": R_p50,
            "R_p10": R_p10,
            "R_p90": R_p90,
        }

        logger.info(
            "[IMU] coherence: rmse_ay_vs_vpsi=%.3f m/s^2, p95=%.3f, bad>0.2=%.1f%%, R[p10,p50,p90]=[%.1f, %.1f, %.1f] m",
            rmse,
            p95,
            (bad_ratio * 100.0 if np.isfinite(bad_ratio) else float('nan')),
            R_p10,
            R_p50,
            R_p90,
        )

        out["acc_x"]  = acc_x.astype("float32")
        out["acc_y"]  = acc_y.astype("float32")
        out["acc_z"]  = acc_z.astype("float32")
        out["gyro_x"] = gyro_x.astype("float32")
        out["gyro_y"] = gyro_y.astype("float32")
        out["gyro_z"] = gyro_z.astype("float32")

        out = out.reset_index()
        ctx.df = out
        return Result()