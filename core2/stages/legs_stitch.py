# core2/stages/legs_stitch.py
from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from ..contracts import Result
from ..context import Context

logger = logging.getLogger(__name__)

def _haversine_series_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(p1) * np.cos(p2) * np.sin(dlmb/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

class LegsStitch:
    """
    Concatène les legs routés, pose un axe temps continu, puis resample à hz.
    """
    name = "LegsStitch"

    def run(self, ctx: Context) -> Result:
        traces = ctx.artifacts.get("legs_traces", [])
        if not traces:
            return Result(ok=False, message="legs_traces manquant")

        hz = int(ctx.cfg.get("sim", {}).get("hz", 10))
        dt = pd.to_timedelta(1 / hz, unit="s")

        # concat brut (on conserve l'ordre des legs)
        df = pd.concat(traces, ignore_index=True)

        plan = ctx.artifacts.get("legs_plan", {})
        start_iso = plan.get("start_time_utc")
        if start_iso:
            # Robust parse supporting both '...Z' and '+00:00'
            t0 = pd.to_datetime(start_iso, utc=True)
        else:
            t0 = pd.Timestamp.utcnow().tz_localize("UTC")

        # --- Nouvelle stratégie de timeline ---
        # 1) Durée totale: on privilégie la somme des durées OSRM si dispo; sinon fallback = 1s/échantillon
        summaries = ctx.artifacts.get("legs_summary", [])
        total_dur_s = None
        try:
            if summaries and isinstance(summaries, list):
                total_dur_s = float(sum(float(s.get("duration_s", 0.0)) for s in summaries))
        except Exception:
            total_dur_s = None

        n = len(df)
        df = df.copy()

        # 2) Paramètre "temps brut" par distance cumulée (évite artefacts liés au nombre de points)
        lat_raw = df["lat"].to_numpy(dtype=float)
        lon_raw = df["lon"].to_numpy(dtype=float)
        dmeters = np.zeros(n, dtype=float)
        if n > 1:
            dmeters[1:] = _haversine_series_m(lat_raw[:-1], lon_raw[:-1], lat_raw[1:], lon_raw[1:])
        cum_m = np.cumsum(dmeters)
        total_m = float(cum_m[-1]) if n > 0 else 0.0
        # Si total_m ~ 0 (trajet dégénéré), on fabrique un cum linéaire pour éviter les divisions par zéro
        if total_m <= 0:
            cum_m = np.linspace(0.0, 1.0, n)
            total_m = 1.0

        # --- Contrôle longueur géo (Haversine) vs somme OSRM (si disponible) ---
        osrm_total_m = None
        try:
            if summaries and isinstance(summaries, list):
                osrm_total_m = float(sum(float(s.get("distance_m", 0.0)) for s in summaries))
        except Exception:
            osrm_total_m = None

        if osrm_total_m and osrm_total_m > 0:
            ratio = float(total_m) / osrm_total_m
            ctx.artifacts["legs_geom_vs_osrm"] = {
                "geo_total_m": float(total_m),
                "osrm_total_m": float(osrm_total_m),
                "ratio": float(ratio),
                "threshold": 0.9,
                "under_sampled": bool(ratio < 0.9),
            }
            if ratio < 0.9:
                logger.warning(
                    "[LegsStitch] Tracé sous-échantillonné: géo=%.1f m, OSRM=%.1f m (ratio=%.3f &lt; 0.9)",
                    total_m, osrm_total_m, ratio
                )
        else:
            # Stocke tout de même l’info géo pour diagnostic
            ctx.artifacts["legs_geom_vs_osrm"] = {
                "geo_total_m": float(total_m),
                "osrm_total_m": None,
                "ratio": None,
                "threshold": 0.9,
                "under_sampled": None,
            }

        # 3) Détermine la durée à utiliser
        if total_dur_s is None or total_dur_s <= 0:
            # fallback: 1 seconde par échantillon (héritage), mais on le note
            total_dur_s = float(max(n - 1, 1))
            ctx.artifacts["legs_stitch_duration_fallback_s"] = total_dur_s

        # 4) Construit un axe temps en secondes basé sur la progression spatiale
        t_lin = (cum_m / total_m) * total_dur_s

        # 5) Indexe par ce temps et reindexe exactement à dt
        df["timestamp"] = pd.to_datetime(t0) + pd.to_timedelta(t_lin, unit="s")
        df = df.set_index("timestamp").sort_index()
        # Déduplique prudemment si nécessaire
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        # Grille régulière exacte à hz, calée sur la durée totale théorique
        step_ms = int(round(1000.0 / hz))  # ex: 10 Hz -> 100 ms
        if step_ms <= 0:
            step_ms = 100  # fallback 10 Hz
        
        # Nombre d'échantillons: plancher pour ne jamais dépasser la durée (inclut t0)
        n_samples = int(np.floor(total_dur_s * hz)) + 1
        t0_utc = pd.to_datetime(t0, utc=True)
        target_index = pd.date_range(start=t0_utc, periods=n_samples, freq=pd.to_timedelta(step_ms, unit="ms"))
        df = df.reindex(target_index).interpolate(method="time")
        # S'assure que l'index est bien UTC
        df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        # Diagnostics
        ctx.meta["hz_expected"] = float(hz)
        ctx.meta["samples_expected"] = int(n_samples)
        ctx.meta["duration_expected_s"] = float(total_dur_s)

        # 6) Recalcule systématiquement la vitesse depuis lat/lon et le dt réel (index temps)
        lat = df["lat"].to_numpy(dtype=float)
        lon = df["lon"].to_numpy(dtype=float)
        ns = df.index.asi8  # int64 ns
        tsec = (ns - ns[0]) / 1e9

        # Diagnostic de cadence
        if len(tsec) > 1:
            dt_obs = np.diff(tsec)
            hz_obs = 1.0 / np.median(dt_obs)
            ctx.meta["hz_observed"] = float(hz_obs)

        dmeters = np.zeros_like(lat, dtype=float)
        if len(lat) > 1:
            dmeters[1:] = _haversine_series_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

        dt_s = np.diff(tsec, prepend=tsec[0])
        pos = dt_s > 0
        median_dt = float(np.median(dt_s[pos])) if pos.any() else float(dt.total_seconds())
        dt_s[~pos] = median_dt

        speed = dmeters / dt_s  # m/s
        speed = np.maximum(speed, 0.0)
        df["speed"] = speed

        # Lissage robuste (médiane centrée ~300 ms) pour éliminer les à-coups numériques
        k = max(1, int(round(0.3 * hz)))
        if k > 1:
            df["speed"] = df["speed"].rolling(window=k, center=True, min_periods=1).median()

        df = df.reset_index()
        ctx.df = df[["timestamp","lat","lon","speed"]]
        ctx.meta["hz"] = hz
        return Result()