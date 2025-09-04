# core2/stages/legs_route.py
from __future__ import annotations
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np

# Optional: precise polyline decoding (polyline6)
try:
    import polyline as _pl  # pip install polyline
except Exception:
    _pl = None

# Optionnel: profils cinématiques si disponibles
try:
    from core.kinematics_speed import retime_polyline_by_speed  # adapte au nom réel si différent
except Exception:
    retime_polyline_by_speed = None  # fallback si module absent

from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(p1) * np.cos(p2) * np.sin(dlmb/2)**2)
    return float(2 * R * np.arcsin(np.sqrt(a)))

def _fallback_route(src: Tuple[float,float], dst: Tuple[float,float]) -> Tuple[pd.DataFrame, float, float]:
    n_min = 200
    dist_m = _haversine_m(src[0], src[1], dst[0], dst[1])
    v_nom = 13.9  # ~50 km/h
    dur_s = max(dist_m / max(v_nom, 0.1), 1.0)
    n = max(n_min, int(dur_s) + 1)
    lat = np.linspace(src[0], dst[0], n)
    lon = np.linspace(src[1], dst[1], n)
    t_rel = np.linspace(0.0, dur_s, n, dtype=float)
    df = pd.DataFrame({"lat": lat, "lon": lon, "t_rel_s": t_rel})
    # déduplique les points strictement identiques (sécurité)
    df = df.loc[~((df["lat"].diff().fillna(1.0) == 0.0) & (df["lon"].diff().fillna(1.0) == 0.0))].reset_index(drop=True)
    return df, float(dist_m), float(dur_s)

def _make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "rs3/legs-route (python-requests)"})
    retry = Retry(total=3, backoff_factor=0.2, status_forcelist=[429, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def _http_osrm_route(session: requests.Session, base_url: str, profile: str, src: Tuple[float,float], dst: Tuple[float,float], timeout: int = 10) -> Tuple[pd.DataFrame, float, float]:
    geom = "polyline6" if _pl is not None else "geojson"
    url = (
        f"{base_url.rstrip('/')}/route/v1/{profile}/"
        f"{src[1]},{src[0]};{dst[1]},{dst[0]}"
        f"?overview=full&geometries={geom}&steps=true&annotations=duration,distance"
    )
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        route = data["routes"][0]
        if _pl is not None:
            # Decode polyline6 to (lat, lon)
            coords = _pl.decode(route["geometry"], precision=6)
            lats = np.array([c[0] for c in coords], dtype=float)
            lons = np.array([c[1] for c in coords], dtype=float)
        else:
            # Fallback: geojson coordinates are [lon, lat]
            coords = route["geometry"]["coordinates"]
            lats = np.array([c[1] for c in coords], dtype=float)
            lons = np.array([c[0] for c in coords], dtype=float)

        dur_s = float(route["duration"])
        dist_m = float(route["distance"])

        # Try to align per-segment durations with geometry points
        seg_dur = None
        try:
            leg0 = route.get("legs", [{}])[0]
            ann = leg0.get("annotation", {}) if isinstance(leg0, dict) else {}
            seg_dur = np.asarray(ann.get("duration", []), dtype=float)
        except Exception:
            seg_dur = None

        if seg_dur is not None and seg_dur.size == (len(lats) - 1):
            t_rel = np.zeros(len(lats), dtype=float)
            t_rel[1:] = np.cumsum(seg_dur)
        else:
            # uniform fallback
            t_rel = np.linspace(0.0, dur_s, len(lats), dtype=float)

        df = pd.DataFrame({"lat": lats, "lon": lons, "t_rel_s": t_rel})
        return df, dist_m, dur_s
    except Exception:
        return _fallback_route(src, dst)

class LegsRoute:
    """
    Route chaque leg via:
      - client interne si cfg.osrm.client == 'internal'
      - HTTP OSRM si cfg.osrm.base_url est défini
      - fallback synthétique sinon
    """
    name = "LegsRoute"
    def __init__(self, profile: str = None, max_workers: int = 4):
        self.profile = profile
        self.max_workers = max_workers

    def run(self, ctx: Context) -> Result:
        plan = ctx.artifacts.get("legs_plan")
        if not plan:
            return Result(ok=False, message="legs_plan manquant")

        legs = plan["legs"]
        osrm_cfg = ctx.cfg.get("osrm", {})
        base_url = osrm_cfg.get("base_url")
        client_mode = osrm_cfg.get("client", "http")  # 'internal' | 'http'
        profile = self.profile or osrm_cfg.get("profile", "driving")

        traces: List[pd.DataFrame] = []
        summaries: List[Dict[str, Any]] = []

        t_accum = 0.0

        # client interne optionnel: core.osrm.client.route_leg(src, dst, profile) -> (df, dist_m, dur_s)
        internal = None
        if client_mode == "internal":
            try:
                from core.osrm.client import route_leg as internal_route_leg  # adapte à ton code
                internal = internal_route_leg
            except Exception:
                internal = None

        if internal:
            # Séquentiel avec client interne
            for leg in legs:
                src = (float(leg["from"]["lat"]), float(leg["from"]["lon"]))
                dst = (float(leg["to"]["lat"]),   float(leg["to"]["lon"]))
                try:
                    df_leg, dist_m, dur_s = internal(src, dst, profile=profile)
                except Exception:
                    df_leg, dist_m, dur_s = _fallback_route(src, dst)
                # 1) Évite doublon à la couture: supprime le premier point des legs suivants
                if traces and not df_leg.empty:
                    df_leg = df_leg.iloc[1:].reset_index(drop=True)
                # 2) Supprime les sauts nuls (lat/lon identiques consécutifs)
                if not df_leg.empty:
                    dup = (df_leg["lat"].diff().fillna(1.0) == 0.0) & (df_leg["lon"].diff().fillna(1.0) == 0.0)
                    if dup.any():
                        df_leg = df_leg.loc[~dup].reset_index(drop=True)
                for col in ("lat","lon","t_rel_s"):
                    if col not in df_leg.columns:
                        return Result(ok=False, message=f"Leg {leg['idx']}: colonne manquante {col}")
                if "speed" not in df_leg.columns:
                    df_leg["speed"] = 0.0
                if "t_rel_s" not in df_leg.columns:
                    n_local = len(df_leg)
                    df_leg["t_rel_s"] = np.linspace(0.0, float(dur_s), n_local, dtype=float)

                kin_cfg = ctx.cfg.get("kinematics", {}) if isinstance(ctx.cfg, dict) else {}
                if retime_polyline_by_speed and kin_cfg.get("use_speed_profile", False):
                    # Attendu: retime_polyline_by_speed(lat, lon, speed_kmh or mps) -> ndarray seconds
                    # Ici on ne connaît pas encore road_type; on peut passer une valeur cible 'nominale'
                    # ou laisser le module décider. On garde un fallback sur t_rel_s existant.
                    try:
                        t_rel_new = retime_polyline_by_speed(
                            df_leg["lat"].to_numpy(), df_leg["lon"].to_numpy(),
                            kin_cfg.get("default_speed_kmh", 50.0)
                        )
                        if t_rel_new is not None and len(t_rel_new) == len(df_leg):
                            df_leg["t_rel_s"] = np.asarray(t_rel_new, dtype=float)
                    except Exception:
                        pass

                # curseur absolu
                df_leg["t_abs_s"] = float(t_accum) + df_leg["t_rel_s"].to_numpy(dtype=float)
                df_leg["leg_idx"] = leg["idx"] if isinstance(leg, dict) and "idx" in leg else None
                t_accum += float(dur_s)
                traces.append(df_leg)
                summaries.append({"idx": leg["idx"], "distance_m": dist_m, "duration_s": dur_s})
        elif base_url:
            # Parallélise les appels HTTP OSRM avec session keep-alive + retries
            session = _make_session()

            def route_one(leg):
                idx = leg["idx"]
                src = (float(leg["from"]["lat"]), float(leg["from"]["lon"]))
                dst = (float(leg["to"]["lat"]),   float(leg["to"]["lon"]))
                try:
                    df_leg, dist_m, dur_s = _http_osrm_route(session, base_url, profile, src, dst, timeout=10)
                except Exception:
                    df_leg, dist_m, dur_s = _fallback_route(src, dst)
                # assurance colonnes
                for col in ("lat","lon","t_rel_s"):
                    if col not in df_leg.columns:
                        raise RuntimeError(f"Leg {idx}: colonne manquante {col}")
                if "speed" not in df_leg.columns:
                    df_leg["speed"] = 0.0
                return (idx, df_leg, dist_m, dur_s)

            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = [ex.submit(route_one, leg) for leg in legs]
                for f in as_completed(futs):
                    results.append(f.result())
            # reorder by idx
            for idx, df_leg, dist_m, dur_s in sorted(results, key=lambda x: x[0]):
                # 1) Évite doublon à la couture: supprime le premier point des legs suivants
                if traces and not df_leg.empty:
                    df_leg = df_leg.iloc[1:].reset_index(drop=True)
                # 2) Supprime les sauts nuls (lat/lon identiques consécutifs)
                if not df_leg.empty:
                    dup = (df_leg["lat"].diff().fillna(1.0) == 0.0) & (df_leg["lon"].diff().fillna(1.0) == 0.0)
                    if dup.any():
                        df_leg = df_leg.loc[~dup].reset_index(drop=True)
                if "t_rel_s" not in df_leg.columns:
                    n_local = len(df_leg)
                    df_leg["t_rel_s"] = np.linspace(0.0, float(dur_s), n_local, dtype=float)

                kin_cfg = ctx.cfg.get("kinematics", {}) if isinstance(ctx.cfg, dict) else {}
                if retime_polyline_by_speed and kin_cfg.get("use_speed_profile", False):
                    # Attendu: retime_polyline_by_speed(lat, lon, speed_kmh or mps) -> ndarray seconds
                    # Ici on ne connaît pas encore road_type; on peut passer une valeur cible 'nominale'
                    # ou laisser le module décider. On garde un fallback sur t_rel_s existant.
                    try:
                        t_rel_new = retime_polyline_by_speed(
                            df_leg["lat"].to_numpy(), df_leg["lon"].to_numpy(),
                            kin_cfg.get("default_speed_kmh", 50.0)
                        )
                        if t_rel_new is not None and len(t_rel_new) == len(df_leg):
                            df_leg["t_rel_s"] = np.asarray(t_rel_new, dtype=float)
                    except Exception:
                        pass

                # curseur absolu
                df_leg["t_abs_s"] = float(t_accum) + df_leg["t_rel_s"].to_numpy(dtype=float)
                df_leg["leg_idx"] = idx
                t_accum += float(dur_s)
                traces.append(df_leg)
                summaries.append({"idx": idx, "distance_m": dist_m, "duration_s": dur_s})
        else:
            # Fallback synthétique (séquentiel)
            for leg in legs:
                src = (float(leg["from"]["lat"]), float(leg["from"]["lon"]))
                dst = (float(leg["to"]["lat"]),   float(leg["to"]["lon"]))
                df_leg, dist_m, dur_s = _fallback_route(src, dst)
                # 1) Évite doublon à la couture: supprime le premier point des legs suivants
                if traces and not df_leg.empty:
                    df_leg = df_leg.iloc[1:].reset_index(drop=True)
                # 2) Supprime les sauts nuls (lat/lon identiques consécutifs)
                if not df_leg.empty:
                    dup = (df_leg["lat"].diff().fillna(1.0) == 0.0) & (df_leg["lon"].diff().fillna(1.0) == 0.0)
                    if dup.any():
                        df_leg = df_leg.loc[~dup].reset_index(drop=True)
                for col in ("lat","lon","t_rel_s"):
                    if col not in df_leg.columns:
                        return Result(ok=False, message=f"Leg {leg['idx']}: colonne manquante {col}")
                if "speed" not in df_leg.columns:
                    df_leg["speed"] = 0.0
                if "t_rel_s" not in df_leg.columns:
                    n_local = len(df_leg)
                    df_leg["t_rel_s"] = np.linspace(0.0, float(dur_s), n_local, dtype=float)

                kin_cfg = ctx.cfg.get("kinematics", {}) if isinstance(ctx.cfg, dict) else {}
                if retime_polyline_by_speed and kin_cfg.get("use_speed_profile", False):
                    # Attendu: retime_polyline_by_speed(lat, lon, speed_kmh or mps) -> ndarray seconds
                    # Ici on ne connaît pas encore road_type; on peut passer une valeur cible 'nominale'
                    # ou laisser le module décider. On garde un fallback sur t_rel_s existant.
                    try:
                        t_rel_new = retime_polyline_by_speed(
                            df_leg["lat"].to_numpy(), df_leg["lon"].to_numpy(),
                            kin_cfg.get("default_speed_kmh", 50.0)
                        )
                        if t_rel_new is not None and len(t_rel_new) == len(df_leg):
                            df_leg["t_rel_s"] = np.asarray(t_rel_new, dtype=float)
                    except Exception:
                        pass

                # curseur absolu
                df_leg["t_abs_s"] = float(t_accum) + df_leg["t_rel_s"].to_numpy(dtype=float)
                df_leg["leg_idx"] = leg["idx"] if isinstance(leg, dict) and "idx" in leg else None
                t_accum += float(dur_s)
                traces.append(df_leg)
                summaries.append({"idx": leg["idx"], "distance_m": dist_m, "duration_s": dur_s})

        ctx.artifacts["legs_traces"] = traces
        ctx.artifacts["legs_summary"] = summaries
        return Result()