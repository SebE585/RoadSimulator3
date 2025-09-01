# simulator/pipeline_utils.py
from __future__ import annotations

import logging
import inspect
from typing import Dict, Callable, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [pipeline_utils] %(message)s')


# ---------------------------- GEO HELPERS ----------------------------

def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distance Haversine (m). Entrées en degrés, supporte ndarrays."""
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _compute_rolling_sinuosity(df: pd.DataFrame, hz: int, window_sec: float = 10.0) -> pd.Series:
    """
    Sinuosité locale ~ (longueur chemin) / (corde) dans une fenêtre centrée.
    Fenêtre courte (≈10s) pour rester stable sur 10 Hz.
    """
    n = len(df)
    if n < 3:
        return pd.Series(np.ones(n), index=df.index, dtype="float64")

    k = max(1, int((window_sec * hz) // 2))
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy()

    # Distances élémentaires (entre i-1 -> i)
    d = np.zeros(n, dtype=float)
    d[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    s = np.cumsum(d)

    sinu = np.ones(n, dtype=float)
    # Boucle raisonnable (O(n))
    for i in range(k, n - k):
        s_path = s[i + k] - s[i - k]
        chord = _haversine_m(lat[i - k], lon[i - k], lat[i + k], lon[i + k])
        sinu[i] = (s_path / chord) if chord > 1e-3 else 1.0

    # Bords : propage vers l'intérieur
    sinu[:k] = sinu[k]
    sinu[-k:] = sinu[-k-1]
    return pd.Series(sinu, index=df.index, dtype="float64")


def _complete_altitude_and_slope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Altitude : si absente/NaN -> 0. Slope % à partir de dz/ds (centré).
    """
    n = len(df)
    if "altitude" not in df.columns:
        df["altitude"] = np.nan
    alt = pd.to_numeric(df["altitude"], errors="coerce")
    if alt.isna().all():
        df["altitude"] = 0.0
        alt = df["altitude"]
        log.debug("[postfill] altitude manquante → 0.0 (fallback plat)")
    else:
        df["altitude"] = alt.interpolate("linear").bfill().ffill()

    # slope % = 100 * dz / ds
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy()
    ds = np.zeros(n); ds[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dz = np.zeros(n); dz[1:-1] = (df["altitude"].to_numpy()[2:] - df["altitude"].to_numpy()[:-2]) / 2.0
    # éviter divisions par 0
    slope = np.zeros(n)
    # pente locale basée sur ds centrée : approx dz/ds
    ds_c = np.zeros(n)
    ds_c[1:-1] = (ds[1:-1] + ds[2:]) / 2.0
    mask = ds_c > 1e-3
    slope[mask] = 100.0 * (dz[mask] / ds_c[mask])
    df["slope_percent"] = slope
    return df


# ---------------------------- POST-FILL EXPORT ----------------------------

def complete_trajectory_fields(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Remplit/calcul les champs manquants pour export “propre” :
      - heading, omega_z, curvature, sinuosité
      - altitude (+ slope%)
      - gyro/acc par défaut si NaN
      - event 'unknown' si demandé
      - osm_highway / road_type harmonisés
    """
    if df is None or df.empty:
        return df

    # Horloge
    hz = int(cfg.get("simulation", {}).get("hz", 10))
    window_sec = float(cfg.get("metrics", {}).get("sinuosity_window_sec", 10.0))
    fill_unknown_events = bool(cfg.get("policy", {}).get("fill_unknown_events", True))

    # Assurer timestamp dtype
    if "timestamp" in df.columns and not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # --- Heading (°) & omega_z (rad/s)
    if "heading" not in df.columns or df["heading"].isna().any():
        try:
            # si le module cinématique est déjà passé, on devrait l'avoir
            from core import kinematics as _k
            df = _k.calculate_heading(df)
        except Exception:
            # fallback simple à partir des deltas lat/lon (grossier)
            lat = np.radians(pd.to_numeric(df["lat"], errors="coerce").to_numpy())
            lon = np.radians(pd.to_numeric(df["lon"], errors="coerce").to_numpy())
            dlon = np.diff(lon, prepend=lon[0])
            dlat = np.diff(lat, prepend=lat[0])
            hdg = (np.degrees(np.arctan2(dlon*np.cos((lat+np.roll(lat,1))/2.0), dlat)) + 360.0) % 360.0
            df["heading"] = hdg

    # omega_z : si absent, approx par dérivée de heading
    if "omega_z" not in df.columns or df["omega_z"].isna().any():
        hdg_rad = np.radians(pd.to_numeric(df["heading"], errors="coerce").fillna(0.0).to_numpy())
        # unwrap pour éviter les sauts 359->0
        hdg_unwrap = np.unwrap(hdg_rad)
        dt = 1.0 / max(hz, 1)
        omega = np.gradient(hdg_unwrap, dt, edge_order=1)
        df["omega_z"] = omega

    # --- Curvature (1/m) : kappa ≈ |omega_z| / v
    v_mps = (pd.to_numeric(df.get("speed"), errors="coerce").fillna(0.0) / 3.6).to_numpy()
    omega = pd.to_numeric(df["omega_z"], errors="coerce").fillna(0.0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.abs(omega) / np.clip(v_mps, 1e-3, None)
    df["curvature"] = kappa

    # --- Sinuosity locale
    df["sinuosity"] = _compute_rolling_sinuosity(df, hz=hz, window_sec=window_sec)

    # --- Altitude & pente
    df = _complete_altitude_and_slope(df)

    # --- Gyros/acc : remplir les vides (plan)
    for g in ("gyro_x", "gyro_y", "gyro_z"):
        if g not in df.columns:
            df[g] = np.nan
        df[g] = pd.to_numeric(df[g], errors="coerce").fillna(0.0)

    for a in ("acc_x", "acc_y"):
        if a not in df.columns:
            df[a] = 0.0
        else:
            df[a] = pd.to_numeric(df[a], errors="coerce").fillna(0.0)
    # acc_z : gravité si absent
    if "acc_z" not in df.columns:
        df["acc_z"] = 9.81
    else:
        df["acc_z"] = pd.to_numeric(df["acc_z"], errors="coerce").fillna(9.81)

    # --- Event : option ‘unknown’ plutôt que NaN
    if "event" not in df.columns:
        df["event"] = np.nan
    df["event"] = df["event"].astype("object")
    if fill_unknown_events:
        df["event"] = df["event"].where(~df["event"].isna(), "unknown")

    # --- osm_highway / road_type : harmoniser
    if "road_type" not in df.columns and "osm_highway" in df.columns:
        df["road_type"] = df["osm_highway"]
    if "osm_highway" not in df.columns and "road_type" in df.columns:
        df["osm_highway"] = df["road_type"]
    # rester prudent : si l'un est NaN -> remplir depuis l'autre
    if "road_type" in df.columns and "osm_highway" in df.columns:
        rt = df["road_type"].astype("object")
        oh = df["osm_highway"].astype("object")
        df["road_type"] = rt.where(~rt.isna(), oh)
        df["osm_highway"] = oh.where(~oh.isna(), df["road_type"])

    # --- Types/dtypes finaux
    numeric_cols = ["speed", "heading", "sinuosity", "curvature", "slope_percent"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Heading borné [0,360)
    if "heading" in df.columns:
        df["heading"] = (df["heading"] % 360.0).fillna(0.0)

    return df


# ---------------------------- EVENTS INJECTION ----------------------------

def _safe_call(func: Callable, **kwargs):
    """
    Appelle `func` en ne passant que les kwargs supportés.
    """
    sig = inspect.signature(func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**allowed)

def _event_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "event" not in df.columns or df.empty:
        return {}
    vc = df["event"].astype("object").value_counts(dropna=False)
    res: Dict[str, int] = {}
    for k, v in vc.items():
        key = "nan" if pd.isna(k) else str(k)
        res[key] = int(v)
    return res

def _delta_for(event_name: str, before: Dict[str, int], after: Dict[str, int]) -> int:
    b = before.get(event_name, 0)
    a = after.get(event_name, 0)
    return max(0, a - b)

def _try_import(*candidates: Tuple[str, str]) -> Optional[Callable]:
    """
    candidates: tuples (module, attr)
    Retourne la première fonction importable, sinon None.
    """
    for mod, attr in candidates:
        try:
            m = __import__(mod, fromlist=[attr])
            fn = getattr(m, attr, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def inject_all_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Injecte tous les événements ponctuels disponibles de façon robuste
    (compat signatures différentes entre branches).
    """
    if df is None or df.empty:
        return df

    # Pré-comptage
    before = _event_counts(df)

    # Liste des générateurs (module, fonction, label event, kwargs de base)
    specs = [
        # Accélération
        (
            [("simulator.events.generation", "generate_acceleration")],
            "acceleration",
            {"df": df, "config": cfg, "cfg": cfg}
        ),
        # Freinage
        (
            [("simulator.events.generation", "generate_freinage")],
            "freinage",
            {"df": df, "config": cfg, "cfg": cfg}
        ),
        # Dos d'âne
        (
            [("simulator.events.generation", "generate_dos_dane")],
            "dos_dane",
            {"df": df, "config": cfg, "cfg": cfg}
        ),
        # Nid de poule
        (
            [("simulator.events.generation", "generate_nid_de_poule")],
            "nid_de_poule",
            {"df": df, "config": cfg, "cfg": cfg}
        ),
        # Trottoir
        (
            [("simulator.events.generation", "generate_trottoir")],
            "trottoir",
            {"df": df, "config": cfg, "cfg": cfg}
        ),
    ]

    # Ouvertures de portes / livraisons (facultatif)
    deliveries_fn = _try_import(
        ("simulator.events.deliveries", "inject_opening_for_deliveries"),
    )

    def _apply(fn: Callable, label: str, base_kwargs: Dict):
        nonlocal df, before
        try:
            # Certains générateurs retournent (df, meta) / (df, n) / df
            out = _safe_call(fn, **base_kwargs)
            if isinstance(out, tuple):
                df_out = out[0]
            else:
                df_out = out
            if isinstance(df_out, pd.DataFrame):
                after = _event_counts(df_out)
                delta = _delta_for(label, before, after)
                df = df_out
                before = after
                log.info("✅ %s injectés : %d", label.replace("_", " ").capitalize(), delta)
            else:
                log.warning("⚠️ %s : retour inattendu (%s)", label, type(out).__name__)
        except Exception as e:
            log.warning("⚠️ %s : injection échouée (%s)", label, e)

    # Appliquer les specs de génération
    for cand_list, label, base_kwargs in specs:
        fn = _try_import(*cand_list)
        if fn is None:
            log.warning("⚠️ %s : générateur indisponible", label)
            continue
        _apply(fn, label, base_kwargs)

    # Ouvertures de portes (signature souvent: inject_opening_for_deliveries(df) -> df)
    if deliveries_fn is not None:
        try:
            out = _safe_call(deliveries_fn, df=df, config=cfg, cfg=cfg)
            if isinstance(out, pd.DataFrame):
                after = _event_counts(out)
                delta = _delta_for("ouverture_porte", before, after)
                df = out
                before = after
                if delta > 0:
                    log.info("✅ Ouverture_porte injectés : %d", delta)
                else:
                    log.info("⚠️ Aucun événement ouverture_porte injecté.")
            else:
                log.warning("⚠️ ouverture_porte : retour inattendu (%s)", type(out).__name__)
        except TypeError:
            # réessayer strictement avec df seul (certaines branches n'acceptent pas de cfg)
            try:
                out = deliveries_fn(df)
                if isinstance(out, pd.DataFrame):
                    after = _event_counts(out)
                    delta = _delta_for("ouverture_porte", before, after)
                    df = out
                    before = after
                    if delta > 0:
                        log.info("✅ Ouverture_porte injectés : %d", delta)
                    else:
                        log.info("⚠️ Aucun événement ouverture_porte injecté.")
                else:
                    log.warning("⚠️ ouverture_porte : retour inattendu (%s)", type(out).__name__)
            except Exception as e:
                log.warning("⚠️ ouverture_porte : injection échouée (%s)", e)
        except Exception as e:
            log.warning("⚠️ ouverture_porte : injection échouée (%s)", e)
    else:
        log.info("⚠️ Aucun événement ouverture_porte injecté.")

    return df