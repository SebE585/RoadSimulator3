# core2/stages/legs_retimer.py
from __future__ import annotations
import numpy as np
import pandas as pd

from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

# Optionnel: utilise le module cinématique si dispo
try:
    # Le module de référence: core/kinematics_speed.py
    from core.kinematics_speed import retime_polyline_by_speed  # type: ignore
except Exception:
    retime_polyline_by_speed = None  # type: ignore


def _haversine_series_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(p1) * np.cos(p2) * np.sin(dlmb/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


class LegsRetimer:
    """
    Recalage temporel de la trajectoire selon un **profil de vitesse cible**
    (dérivé du type de route ou fourni via `target_speed`).

    Entrée (df):
      - 'lat','lon','timestamp' obligatoires
      - 'road_type' (catégorie OSM) et/ou 'target_speed' (km/h)
    Sortie:
      - 'timestamp' retimé (monotone croissant)
      - 't_abs_s' (secondes depuis t0)
      - 'speed' sera recalculée plus tard par SpeedSync

    Paramètres:
      - speed_by_type: mapping {road_type: km/h}
      - default_kmh: km/h par défaut si inconnu
      - use_column_target_speed: si True, priorise df['target_speed'] (km/h) quand dispo
      - min_dt: pas de temps minimal en secondes (anti-dt=0)
    """
    name = "LegsRetimer"

    def __init__(
        self,
        speed_by_type: dict | None = None,
        default_kmh: float = 50.0,
        use_column_target_speed: bool = True,
        min_dt: float = 0.05,
    ) -> None:
        if speed_by_type is None:
            speed_by_type = {
                "motorway": 110.0,
                "trunk": 100.0,
                "primary": 70.0,
                "secondary": 50.0,
                "tertiary": 40.0,
                "residential": 30.0,
                "service": 20.0,
                "unclassified": 50.0,
                "unknown": 50.0,
            }
        self.speed_by_type = dict(speed_by_type)
        self.default_kmh = float(default_kmh)
        self.use_column_target_speed = bool(use_column_target_speed)
        self.min_dt = float(min_dt)

    @staticmethod
    def _kmh_to_mps(x: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(x, dtype=float) * (1000.0/3600.0))

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result((False, "df vide"))

        req_cols = {"lat", "lon", "timestamp"}
        if not req_cols.issubset(df.columns):
            return Result((False, f"colonnes manquantes: {list(req_cols - set(df.columns))}"))

        out = df.copy()

        # 1) Fabrique un vecteur de vitesses cibles (km/h)
        if self.use_column_target_speed and "target_speed" in out.columns:
            v_kmh = pd.to_numeric(out["target_speed"], errors="coerce").to_numpy(dtype=float)
        else:
            rt = out["road_type"].astype("object") if "road_type" in out.columns else None
            if rt is not None:
                v_kmh = np.array([self.speed_by_type.get(str(t) if t is not None else "unknown", self.default_kmh) for t in rt], dtype=float)
            else:
                v_kmh = np.full(len(out), self.default_kmh, dtype=float)

        # Remplace les valeurs non valides
        v_kmh = np.nan_to_num(v_kmh, nan=self.default_kmh, posinf=self.default_kmh, neginf=self.default_kmh)
        v_mps = np.maximum(self._kmh_to_mps(v_kmh), 0.1)  # borne > 0

        # 2) Calcule t_rel en s (cumul de d / v_cible)
        lat = out["lat"].to_numpy(dtype=float)
        lon = out["lon"].to_numpy(dtype=float)
        dmeters = np.zeros(len(out), dtype=float)
        if len(out) > 1:
            dmeters[1:] = _haversine_series_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt = dmeters / v_mps  # secondes par segment
        # Borne minimale dt (évite dt=0)
        dt = np.maximum(dt, self.min_dt)
        t_rel = np.cumsum(dt)
        t_rel[0] = 0.0

        # 3) Si le module cinématique existe, on peut raffiner
        if retime_polyline_by_speed is not None:
            try:
                # L'API peut attendre des km/h ou m/s selon l'impl; ici on passe km/h
                t_rel2 = retime_polyline_by_speed(lat, lon, v_kmh)
                if t_rel2 is not None and len(t_rel2) == len(out):
                    t_rel = np.asarray(t_rel2, dtype=float)
                    # Sécurité monotonicité + min_dt
                    t_rel = np.maximum.accumulate(np.maximum(t_rel, 0.0))
                    t_rel = np.maximum(t_rel, np.linspace(0.0, self.min_dt*(len(out)-1), len(out)))
            except Exception:
                pass

        # 4) Reconstruit l'axe de temps absolu depuis t0 (sans resampling)
        ts0 = pd.to_datetime(out["timestamp"].iloc[0], utc=True, errors="coerce")
        if pd.isna(ts0):
            ts0 = pd.Timestamp.utcnow().tz_localize("UTC")

        out["t_abs_s"] = t_rel
        out["timestamp"] = ts0 + pd.to_timedelta(out["t_abs_s"], unit="s")

        # 5) Garde-fou: s'assurer que les timestamps sont strictement croissants
        idx = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        ns = idx.astype("int64").to_numpy()
        dtn = np.diff(ns, prepend=ns[0])  # int64 ns step

        pos = dtn > 0
        if not pos.all():
            # médiane des pas strictement positifs ; fallback 100ms si nécessaire
            if pos.any():
                med = int(np.median(dtn[pos]))
                if med <= 0:
                    med = 100_000_000  # 0.1 s en ns
            else:
                med = 100_000_000

            dtn[~pos] = med
            ns = np.cumsum(dtn)
            out["timestamp"] = pd.to_datetime(ns, utc=True, unit="ns")

        # Note: 'speed' sera recalculée plus tard par SpeedSync (post-retiming)
        ctx.df = out
        return Result((True, "OK"))
