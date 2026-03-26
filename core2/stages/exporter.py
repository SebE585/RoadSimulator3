# core2/stages/exporter.py
from __future__ import annotations
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rs3_contracts.api import Result
from ..context import Context


logger = logging.getLogger(__name__)

# --- distance helper pour le résumé ---
def _haversine_series_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

def _compute_summary(df: pd.DataFrame, ctx: Context, hz_meta: float | None) -> dict:
    """Retourne un dict KPI: duration_s, distance_km, avg_kmh, stops_count, stop_time_s."""
    if df is None or df.empty:
        return dict(duration_s=0, distance_km=0, avg_kmh=0, stops_count=0, stop_time_s=0)

    # Durée
    t = pd.to_datetime(df.get("timestamp", []), utc=True, errors="coerce")
    if len(t) >= 2 and t.notna().any():
        duration_s = float((t.iloc[-1] - t.iloc[0]).total_seconds())
    else:
        duration_s = float(ctx.meta.get("duration_after_speed_sync_s") or ctx.meta.get("duration_expected_s") or 0)

    # Distance (haversine cumulée)
    lat = pd.to_numeric(df.get("lat", pd.Series([], dtype=float)), errors="coerce").astype(float).to_numpy()
    lon = pd.to_numeric(df.get("lon", pd.Series([], dtype=float)), errors="coerce").astype(float).to_numpy()
    dist_m = 0.0
    if len(lat) > 1:
        d = _haversine_series_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dist_m = float(np.nansum(d))
    distance_km = dist_m / 1000.0

    # Vitesse moyenne (GPS)
    avg_kmh = (distance_km / (duration_s / 3600.0)) if duration_s > 0 else 0.0

    # Stops (compte blocs STOP contigus) + temps à l'arrêt
    stops_count = 0
    stop_time_s = 0.0
    if "event" in df.columns and len(df):
        ev = df["event"].astype(str).fillna("").to_numpy()
        is_stop = (ev == "STOP")
        if is_stop.any():
            hz = float(hz_meta) if hz_meta and hz_meta > 0 else float(ctx.meta.get("hz", 10) or 10)
            stop_time_s = float(np.sum(is_stop)) / hz
            prev = False
            for cur in is_stop:
                if cur and not prev:
                    stops_count += 1
                prev = cur

    return dict(
        duration_s=max(0.0, duration_s),
        distance_km=max(0.0, distance_km),
        avg_kmh=max(0.0, avg_kmh),
        stops_count=int(stops_count),
        stop_time_s=max(0.0, stop_time_s),
    )

# ========= Templates =========
#
# Note: _write_report_inline injects Plotly via either a local asset or CDN using the {{PLOTLY_TAG}} placeholder.

def _templates_dir() -> Path:
    """
    Localise le répertoire 'templates' du dépôt.
    Fallback: un dossier 'templates' au voisinage du fichier courant.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "templates",
        here.parent.parent / "templates",
        here.parent / "templates",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return here.parent / "templates"

def _read_template(name: str) -> str:
    """
    Lit un template texte depuis le dossier templates.
    Lève une exception s'il est introuvable.
    """
    path = _templates_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"Template manquant: {path}")
    return path.read_text(encoding="utf-8")

def _render_template(tpl: str, mapping: dict[str, str]) -> str:
    """
    Remplacement ultra-simple: {{KEY}} -> mapping['KEY'] (converti en str).
    """
    out = tpl
    for k, v in mapping.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out

# ========= JSON / Schema helpers =========

def _sanitize(o: Any):
    """Convertit récursivement en objets JSON-compatibles pour artifacts/meta."""
    if isinstance(o, (datetime, pd.Timestamp)):
        s = o.isoformat()
        return s.replace("+00:00", "Z")
    if isinstance(o, pd.Timedelta):
        return o.isoformat()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, pd.DataFrame):
        return [_sanitize(rec) for rec in o.to_dict(orient="records")]
    if isinstance(o, pd.Series):
        return _sanitize(o.to_dict())
    if isinstance(o, (pd.Index, pd.RangeIndex, pd.DatetimeIndex)):
        return _sanitize(list(o))
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o

_DEFAULT_SCHEMA_PATHS = [
    "config/rs3_dataset_schema.yaml",
    "config/dataset_schema.yaml",
]

_DTYPE_MAP = {
    "float32": "float32",
    "float64": "float64",
    "category": "category",
    "datetime": "datetime64[ns, UTC]",
}

def _load_schema(cfg: dict) -> dict | None:
    """Charge le schéma RS3 depuis cfg['dataset_schema'] ou chemins par défaut."""
    path = None
    if isinstance(cfg, dict):
        path = cfg.get("dataset_schema")
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(_DEFAULT_SCHEMA_PATHS)
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or None
        except Exception:
            continue
    return None

def _ensure_utc_iso_z(ts: pd.Series) -> pd.Series:
    """Force les timestamps en UTC et au format ISO avec 'Z' final (string)."""
    s = pd.to_datetime(ts, utc=True, errors="coerce")
    return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _apply_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Applique le schéma: colonnes, ordre, cast, timestamps ISO-Z, valeurs par défaut."""
    if not schema or "columns" not in schema:
        return df

    df_out = df.copy()
    order = []
    for col in schema["columns"]:
        name = col.get("name")
        dtype = _DTYPE_MAP.get(col.get("dtype", ""), None)
        nullable = bool(col.get("nullable", True))
        order.append(name)

        if name not in df_out.columns:
            if dtype in ("float32", "float64"):
                df_out[name] = 0.0
            elif dtype == "category":
                df_out[name] = pd.Series([None] * len(df_out), dtype="object")
            elif dtype and dtype.startswith("datetime"):
                df_out[name] = pd.NaT
            else:
                df_out[name] = pd.NA

        # Cast
        if dtype == "float32":
            df_out[name] = pd.to_numeric(df_out[name], errors="coerce").astype("float32")
            if not nullable:
                df_out[name] = df_out[name].fillna(0.0)
        elif dtype == "float64":
            df_out[name] = pd.to_numeric(df_out[name], errors="coerce").astype("float64")
            if not nullable:
                df_out[name] = df_out[name].fillna(0.0)
        elif dtype == "category":
            df_out[name] = df_out[name].astype("object")
        elif dtype == "datetime64[ns, UTC]":
            df_out[name] = _ensure_utc_iso_z(df_out[name])

        # Non-nullable
        if not nullable:
            if dtype in ("float32", "float64"):
                df_out[name] = df_out[name].fillna(0.0)
            elif dtype == "category":
                df_out[name] = df_out[name].fillna("")
            elif dtype == "datetime64[ns, UTC]":
                mask = df_out[name].astype(str).eq("NaT")
                if mask.any():
                    df_out.loc[mask, name] = "1970-01-01T00:00:00.000000Z"

    # Ordre
    extras = [c for c in df_out.columns if c not in order]
    df_out = df_out[order + extras]

    # Catégories à la fin
    for col in schema["columns"]:
        name = col["name"]
        if col.get("dtype") == "category":
            df_out[name] = df_out[name].astype("category")

    return df_out

def _apply_start_time(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Décale toute la timeline pour aligner le premier timestamp sur cfg.start_time_utc."""
    if not isinstance(cfg, dict):
        return df
    start_s = cfg.get("start_time_utc")
    if not start_s:
        return df
    out = df.copy()
    t = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if t.isna().all():
        return df
    t0 = t.iloc[0]
    try:
        t_target = pd.to_datetime(start_s, utc=True, errors="raise")
    except Exception:
        return df
    delta = t_target - t0
    out["timestamp"] = (t + delta).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return out

# ========= Telemachus (v0.1) inline export =========

def _build_telemachus_samples_no_derive(df: pd.DataFrame, tele_cfg: dict) -> pd.DataFrame:
    """Mappe les colonnes présentes vers le schéma Telemachus 0.1 sans dérivation.
    Conversions minimales autorisées: km/h -> m/s pour speed.
    """
    out = pd.DataFrame(index=df.index)

    # ts (UTC ISO Z)
    if "timestamp" in df.columns:
        out["ts"] = _ensure_utc_iso_z(df["timestamp"])
    else:
        out["ts"] = pd.Series([None] * len(df), dtype="object")

    # lat/lon
    for src, dst in (("lat", "lat"), ("lon", "lon")):
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dst] = pd.Series([None] * len(df), dtype="float64")

    # speed_mps: priorité aux colonnes déjà en m/s; conversion simple si *_kmh existe
    speed_mps = None
    for cand in ("speed_mps", "speed", "v_mps", "v"):
        if cand in df.columns:
            speed_mps = pd.to_numeric(df[cand], errors="coerce")
            break
    if speed_mps is None:
        for cand in ("speed_kmh", "v_kmh", "speed_km_h"):
            if cand in df.columns:
                speed_mps = pd.to_numeric(df[cand], errors="coerce") * (1000.0/3600.0)
                break
    out["speed_mps"] = speed_mps if speed_mps is not None else pd.Series([None] * len(df), dtype="float64")

    # acc
    mapping_acc = {"acc_x": "ax_mps2", "acc_y": "ay_mps2", "acc_z": "az_mps2"}
    for src, dst in mapping_acc.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dst] = pd.Series([None] * len(df), dtype="float64")

    # gyro (on suppose déjà en rad/s dans la pipeline RS3)
    mapping_gyro = {"gyro_x": "gx_rad_s", "gyro_y": "gy_rad_s", "gyro_z": "gz_rad_s"}
    for src, dst in mapping_gyro.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dst] = pd.Series([None] * len(df), dtype="float64")

    # heading / altitude / slope
    if "heading" in df.columns:
        out["heading_deg"] = pd.to_numeric(df["heading"], errors="coerce")
    elif "heading_deg" in df.columns:
        out["heading_deg"] = pd.to_numeric(df["heading_deg"], errors="coerce")
    else:
        out["heading_deg"] = pd.Series([None] * len(df), dtype="float64")

    # altitude
    alt_src = None
    for cand in ("altitude_m", "z_m", "z", "elevation", "altitude"):
        if cand in df.columns:
            alt_src = cand; break
    if alt_src:
        out["altitude_m"] = pd.to_numeric(df[alt_src], errors="coerce")
    else:
        out["altitude_m"] = pd.Series([None] * len(df), dtype="float64")

    # slope
    slope_src = None
    for cand in ("slope_percent", "slope_pct", "grade_percent"):
        if cand in df.columns:
            slope_src = cand; break
    if slope_src:
        out["slope_percent"] = pd.to_numeric(df[slope_src], errors="coerce")
    else:
        out["slope_percent"] = pd.Series([None] * len(df), dtype="float64")

    # Identifiants
    vehicle_id = (tele_cfg or {}).get("vehicle_id") or "VL-UNKNOWN"
    trip_id = (tele_cfg or {}).get("trip_id")
    out["vehicle_id"] = vehicle_id
    if trip_id:
        out["trip_id"] = trip_id

    # Ordre recommandé
    cols_order = [
        "ts", "lat", "lon", "speed_mps",
        "ax_mps2", "ay_mps2", "az_mps2",
        "gx_rad_s", "gy_rad_s", "gz_rad_s",
        "heading_deg", "altitude_m", "slope_percent",
        "vehicle_id", "trip_id"
    ]
    # garde l'ordre connu puis ajoute extras si existants
    known = [c for c in cols_order if c in out.columns]
    extras = [c for c in out.columns if c not in known]
    return out[known + extras]


def _export_telemachus_inline(ctx: Context, df: pd.DataFrame, sim_outdir: str, tele_cfg: dict | None) -> None:
    if not isinstance(tele_cfg, dict) or not tele_cfg.get("enabled", False):
        return
    version = str(tele_cfg.get("version", "0.1"))
    vehicle_id = tele_cfg.get("vehicle_id", "VL-UNKNOWN")
    trip_prefix = tele_cfg.get("trip_prefix", "TR")
    write_parquet = bool(tele_cfg.get("write_parquet", True))
    write_csv = bool(tele_cfg.get("write_csv", True))

    # out dir
    out_dir = tele_cfg.get("out_dir")
    if out_dir:
        out_dir = os.path.join(sim_outdir, out_dir) if not os.path.isabs(out_dir) else out_dir
    else:
        out_dir = os.path.join(sim_outdir, "telemachus")
    os.makedirs(out_dir, exist_ok=True)

    # trip_id par défaut
    if "trip_id" in tele_cfg and tele_cfg["trip_id"]:
        trip_id = str(tele_cfg["trip_id"]) 
    else:
        trip_id = f"{trip_prefix}-{Path(sim_outdir).name}"

    # construit samples sans dérivation
    tele_local_cfg = {"vehicle_id": vehicle_id, "trip_id": trip_id}
    samples = _build_telemachus_samples_no_derive(df, tele_local_cfg)

    # écritures
    artifacts = {"parquet": None, "csv": None}
    if write_parquet:
        try:
            pqt = os.path.join(out_dir, "samples.parquet")
            samples.to_parquet(pqt, index=False)
            artifacts["parquet"] = pqt
        except Exception as e:
            logger.info("[Telemachus] Parquet indisponible (%s)", e)
    if write_csv or not artifacts["parquet"]:
        csvp = os.path.join(out_dir, "samples.csv")
        samples.to_csv(csvp, index=False)
        artifacts["csv"] = csvp

    # dataset.json
    meta = {
        "format": "Telemachus",
        "version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "artifacts": artifacts,
        "source": {
            "project": "RoadSimulator3",
            "rs3_outdir": sim_outdir,
            "rs3_timeline": os.path.join(sim_outdir, "timeline.csv"),
        },
    }
    with open(os.path.join(out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Telemachus] samples → {artifacts['parquet'] or artifacts['csv']}")

def _ensure_event_column(df: pd.DataFrame) -> pd.DataFrame:
    """Crée 'event' à partir de flags (STOP/WAIT) si absent."""
    if "event" in df.columns and df["event"].notna().any():
        return df
    out = df.copy()
    stop_flag = None
    wait_flag = None
    for name in ("flag_stop", "stop_flag", "is_stop"):
        if name in out.columns:
            stop_flag = pd.to_numeric(out[name], errors="coerce").fillna(0.0).astype(float)
            break
    for name in ("flag_wait", "wait_flag", "is_wait"):
        if name in out.columns:
            wait_flag = pd.to_numeric(out[name], errors="coerce").fillna(0.0).astype(float)
            break
    if stop_flag is None and wait_flag is None:
        out["event"] = ""
        return out
    ev = np.array([""] * len(out), dtype=object)
    if wait_flag is not None:
        ev = np.where(wait_flag > 0, "WAIT", ev)
    if stop_flag is not None:
        ev = np.where(stop_flag > 0, "STOP", ev)  # STOP prioritaire
    out["event"] = ev
    return out

# ========= Map & Charts (inline data, no CORS) =========

def _median_dt_seconds_from_index(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 0.0
    ns = idx.asi8
    dt = np.diff((ns - ns[0]) / 1e9, prepend=0.0)
    pos = dt > 0
    if not pos.any():
        return 0.0
    return float(np.median(dt[pos]))

def _to_float_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df[name], errors="coerce").astype("float64")

def _sample_track_for_map(
    df: pd.DataFrame,
    hz_meta: float | None = None,
    sample_hz: float | None = 1.0,
    max_points: int = 20000
) -> dict:
    """
    Prépare les données de carte intégrées :
      - pts: [[lat, lon], ...] sous-échantillonné (~sample_hz) / limité à max_points
      - events: [{lat, lon, type}, ...]
    """
    if df is None or df.empty:
        return {"pts": [], "events": []}

    lat = pd.to_numeric(df.get("lat", pd.Series([], dtype=float)), errors="coerce").astype(float)
    lon = pd.to_numeric(df.get("lon", pd.Series([], dtype=float)), errors="coerce").astype(float)
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        t = pd.date_range(periods=len(lat), freq="100ms", start=pd.Timestamp.now(tz="UTC").floor("s"))
    n = len(lat)

    # Stride depuis Hz meta/timeline
    stride = 1
    if n > 1:
        hz_src = None
        if hz_meta and hz_meta > 0:
            hz_src = float(hz_meta)
        else:
            dt_med = _median_dt_seconds_from_index(t)
            if dt_med > 0:
                hz_src = 1.0 / dt_med
        if sample_hz and hz_src and hz_src > 0 and sample_hz > 0:
            stride = max(1, int(round(hz_src / float(sample_hz))))
    if n // max(1, stride) > max_points:
        stride = max(1, int(np.ceil(n / max_points)))

    pts = []
    for i in range(0, n, stride):
        try:
            la = float(lat.iloc[i]); lo = float(lon.iloc[i])
            if np.isfinite(la) and np.isfinite(lo):
                pts.append([la, lo])
        except Exception:
            continue

    # Fallback si trop peu de points
    if len(pts) < 2:
        pts = []
        for i in range(0, n):
            try:
                la = float(lat.iloc[i]); lo = float(lon.iloc[i])
                if np.isfinite(la) and np.isfinite(lo):
                    pts.append([la, lo])
            except Exception:
                continue

    events = []
    if "event" in df.columns:
        ev = df["event"].astype(str).fillna("")
        for i, e in ev.items():
            if e and e.upper() in ("STOP", "WAIT"):
                try:
                    la = float(df.at[i, "lat"]); lo = float(df.at[i, "lon"])
                    if np.isfinite(la) and np.isfinite(lo):
                        events.append({"lat": la, "lon": lo, "type": e.upper()})
                except Exception:
                    pass

    return {"pts": pts, "events": events}

def _build_chart_json_inline(
    df: pd.DataFrame,
    charts_sample_hz: float | None,
    hz_meta: float | None
) -> str:
    """
    Construit un JSON compact pour les graphiques (sans fetch CSV).
    - Downsample via charts_sample_hz (défaut: 1.0 si fourni)
    - vitesses (km/h), acc en g, gyros rad/s, altitude (m), pente (%)
    - on conserve aussi les arrays complets vitesse_kmh & road_type pour le violon
    """
    if df is None or df.empty:
        return json.dumps({"T": []}, ensure_ascii=False)

    # Stride pour downsample
    hz_src = float(hz_meta) if hz_meta and hz_meta > 0 else None
    if hz_src is None:
        t_tmp = pd.to_datetime(df.get("timestamp", []), utc=True, errors="coerce")
        dt_med = _median_dt_seconds_from_index(pd.DatetimeIndex(t_tmp))
        if dt_med > 0:
            hz_src = 1.0 / dt_med
    stride = 1
    if charts_sample_hz and hz_src and hz_src > 0 and charts_sample_hz > 0:
        stride = max(1, int(round(hz_src / float(charts_sample_hz))))

    cols = df.columns
    t = pd.to_datetime(df.get("timestamp", []), utc=True, errors="coerce")
    T_all = t.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").tolist()

    def _num(name, default=None):
        if name in cols:
            return pd.to_numeric(df[name], errors="coerce").astype(float).tolist()
        return default

    # speed km/h
    speed_mps = _num("speed", None)
    speed_kmh = _num("speed_kmh", None)
    if speed_kmh is None and speed_mps is not None:
        speed_kmh = [(v * 3.6 if np.isfinite(v) else None) for v in speed_mps]

    # acc (g)
    g0 = 9.80665
    acc_x = _num("acc_x", [])
    acc_y = _num("acc_y", [])
    acc_z = _num("acc_z", [])
    acc_x_g = [(v / g0 if np.isfinite(v) else None) for v in acc_x] if acc_x else []
    acc_y_g = [(v / g0 if np.isfinite(v) else None) for v in acc_y] if acc_y else []
    acc_z_g = [(v / g0 if np.isfinite(v) else None) for v in acc_z] if acc_z else []

    # gyro
    gyro_x = _num("gyro_x", [])
    gyro_y = _num("gyro_y", [])
    gyro_z = _num("gyro_z", [])

    # altitude / pente
    altitude = _num("altitude_m", [])
    slope = _num("slope_percent", [])

    # type route (pour violon)
    if "road_type" in cols:
        rtype_all = df["road_type"].astype(str).fillna("unknown").tolist()
    elif "osm_highway" in cols:
        rtype_all = df["osm_highway"].astype(str).fillna("unknown").tolist()
    else:
        rtype_all = ["unknown"] * len(df)

    # Downsample indices
    idxs = list(range(0, len(T_all), stride))

    data = {
        "T": [T_all[i] for i in idxs],
        "speed_kmh": [speed_kmh[i] for i in idxs] if speed_kmh else [],

        "acc_x_g": [acc_x_g[i] for i in idxs] if acc_x_g else [],
        "acc_y_g": [acc_y_g[i] for i in idxs] if acc_y_g else [],
        "acc_z_g": [acc_z_g[i] for i in idxs] if acc_z_g else [],

        "gyro_x": [gyro_x[i] for i in idxs] if gyro_x else [],
        "gyro_y": [gyro_y[i] for i in idxs] if gyro_y else [],
        "gyro_z": [gyro_z[i] for i in idxs] if gyro_z else [],

        "altitude_m": [altitude[i] for i in idxs] if altitude else [],
        "slope_percent": [slope[i] for i in idxs] if slope else [],

        # non downsamplés (violon)
        "road_type": rtype_all,
        "speed_kmh_full": speed_kmh or [],
    }
    return json.dumps(data, ensure_ascii=False)

def _write_map_embedded(outdir: str, track_json: str) -> str:
    """
    Rend 'map.html' à partir du template 'map_embedded.html' en y injectant DATA_JSON.
    """
    tpl = _read_template("map_embedded.html")
    html = _render_template(tpl, {"DATA_JSON": track_json})
    path = Path(outdir) / "map.html"
    path.write_text(html, encoding="utf-8")
    return str(path)

def _write_report_inline(ctx: Context, outdir: str, title: str, chart_json: str) -> str:
    """
    Renders 'report.html' from the 'report.html' template.
    Placeholders :
      - {{TITLE}}
      - {{QA_BLOCK}}
      - {{PLOTLY_TAG}}
      - {{INLINE_BLOCK}}
    """
    # Bloc QA (markup non stylé → styles dans le template)
    qa_html = ""
    try:
        qa = ctx.artifacts.get("qa_pretty") or {}
        if isinstance(qa, dict):
            status = (qa.get("status") or "").strip()
            block = (qa.get("text") or "").strip()
            if status or block:
                qa_html = (
                    "<div class='qa-panel'>"
                    f"<div class='qa-status'>{status}</div>"
                    f"<pre class='qa-body'>{block}</pre>"
                    "</div>"
                )
    except Exception:
        pass

    # Plotly local si dispo sinon CDN
    assets_dir = _templates_dir() / "assets"
    plotly_path = assets_dir / "plotly-2.32.0.min.js"
    if plotly_path.exists():
        try:
            content = plotly_path.read_text(encoding="utf-8")
            plotly_tag = "<script>" + content + "</script>"
        except Exception:
            plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
    else:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'

    # Injecte données inline (charts + summary)
    hz_meta = float(ctx.meta.get("hz", 10.0)) if ctx.meta.get("hz") else None
    df = ctx.df if ctx.df is not None else pd.DataFrame()
    summary = _compute_summary(df, ctx, hz_meta)
    summary_json = json.dumps(summary, ensure_ascii=False)
    inline_block = (
        f"<script>window.RS3_CHART = {chart_json};"
        f"window.RS3_SUMMARY = {summary_json};</script>"
    )

    # Rend le template
    tpl = _read_template("report.html")
    html = _render_template(tpl, {
        "TITLE": title,
        "QA_BLOCK": qa_html,
        "PLOTLY_TAG": plotly_tag,
        "INLINE_BLOCK": inline_block,
    })
    path = Path(outdir) / "report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)

# ========= Orchestrateur (Exporter) =========

def _generate_outputs(ctx: Context, df: pd.DataFrame, outdir: str, cfg: dict | None = None) -> tuple[str | None, str | None]:
    """
    Génère:
      - map.html (embed JSON → pas de CORS)
      - report.html (inline JSON → pas de CORS)
    Renvoie (report_path, map_path).
    """
    rep_cfg = (cfg or {})
    title = rep_cfg.get("title") or "RoadSimulator3 — Rapport"

    # Paramètres par défaut anti-CORS (tout inline)
    charts_sample_hz = rep_cfg.get("charts_sample_hz", 1.0)   # downsample visible
    map_sample_hz = rep_cfg.get("map_sample_hz", 1.0)         # trace ~1 Hz
    map_max_points = int(rep_cfg.get("map_max_points", 20000))

    hz_meta = float(ctx.meta.get("hz", 10.0)) if ctx.meta.get("hz") else None

    # -- Map embedded
    try:
        track = _sample_track_for_map(df, hz_meta=hz_meta, sample_hz=map_sample_hz, max_points=map_max_points)
        # Stops planifiés (facultatif) ajoutés dans le JSON
        planned = None
        for k in ("stops", "plan_stops", "stops_plan"):
            if k in ctx.artifacts and isinstance(ctx.artifacts[k], (list, tuple)):
                planned = ctx.artifacts[k]
                break
        track["planned"] = planned or []
        map_json = json.dumps(track, ensure_ascii=False)
        map_path = _write_map_embedded(outdir, map_json)
    except Exception as e:
        logger.exception("[Exporter] map.html non générée: %s", e)
        map_path = None

    # -- Charts inline
    try:
        chart_json = _build_chart_json_inline(df, charts_sample_hz=charts_sample_hz, hz_meta=hz_meta)
        report_path = _write_report_inline(ctx, outdir, title, chart_json)
    except Exception as e:
        logger.exception("[Exporter] report.html non généré: %s", e)
        report_path = None

    return report_path, map_path


# ═══════════════════════════════════════════════════════════════════════════
# D0 Export — RFC-0013 compliant
# ═══════════════════════════════════════════════════════════════════════════

def _export_d0(ctx, df: pd.DataFrame, outdir: str) -> str | None:
    """
    Exporte un fichier D0 conforme RFC-0013.
    Colonnes D0 uniquement — pas de road_type, event, target_speed.
    Format : Parquet + CSV.
    """
    d0 = pd.DataFrame()

    # Timestamp (obligatoire)
    if "timestamp" in df.columns:
        d0["ts"] = df["timestamp"]
    elif "ts" in df.columns:
        d0["ts"] = df["ts"]

    # GPS (obligatoire, NaN entre ticks si multi-rate)
    for src, dst in [("lat", "lat"), ("lon", "lon")]:
        d0[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan

    # Speed (obligatoire)
    for cand in ("speed_mps", "speed", "v_mps"):
        if cand in df.columns:
            d0["speed_mps"] = pd.to_numeric(df[cand], errors="coerce")
            break
    if "speed_mps" not in d0.columns:
        d0["speed_mps"] = np.nan

    # Heading (recommandé) — calculer sur les ticks GPS valides uniquement
    if "heading" in df.columns:
        d0["heading_deg"] = pd.to_numeric(df["heading"], errors="coerce")
    elif "heading_deg" in df.columns:
        d0["heading_deg"] = pd.to_numeric(df["heading_deg"], errors="coerce")
    else:
        d0["heading_deg"] = np.nan
        gps_valid = d0["lat"].notna()
        if gps_valid.sum() > 1:
            lat_gps = d0.loc[gps_valid, "lat"]
            lon_gps = d0.loc[gps_valid, "lon"]
            dlat = lat_gps.diff()
            dlon = lon_gps.diff()
            heading = np.degrees(np.arctan2(dlon.values, dlat.values)) % 360
            d0.loc[gps_valid, "heading_deg"] = heading
            # NaN à l'arrêt (speed < 0.5 m/s)
            if "speed_mps" in d0.columns:
                stopped = gps_valid & (d0["speed_mps"].fillna(0) < 0.5)
                d0.loc[stopped, "heading_deg"] = np.nan

    # Altitude GPS : PAS dans D0 — c'est la promesse du D1 (DEM enrichment)
    # Même si le FMC880 remonte une altitude NMEA, elle est trop imprécise (±30m)
    # Le DEM à 1m de résolution est la source de vérité → D1

    # HDOP simulé (recommandé) — valeurs réalistes FMC880
    rng_hdop = np.random.default_rng(43)
    hdop_base = rng_hdop.lognormal(mean=0.3, sigma=0.3, size=len(d0)).clip(0.8, 5.0)
    d0["hdop"] = np.nan
    d0.loc[d0["lat"].notna(), "hdop"] = hdop_base[:d0["lat"].notna().sum()].round(1)

    # Satellites simulé (recommandé) — 6-12 typique FMC880
    rng_sat = np.random.default_rng(44)
    n_sat = rng_sat.integers(6, 13, size=d0["lat"].notna().sum())
    d0["n_satellites"] = np.nan
    d0.loc[d0["lat"].notna(), "n_satellites"] = n_sat

    # Bruit GPS (configurable)
    gps_cfg = (ctx.cfg if isinstance(ctx.cfg, dict) else {}).get("gps_noise", {})
    sigma_pos_m = float(gps_cfg.get("sigma_pos_m", 2.0))
    sigma_speed = float(gps_cfg.get("sigma_speed_mps", 0.1))
    jump_prob = float(gps_cfg.get("jump_probability", 0.001))  # 0.1% des ticks GPS = saut
    jump_max_m = float(gps_cfg.get("jump_max_m", 50.0))

    # Cold start drift (décroissance exponentielle en début de trace)
    cold_start_drift_m = float(gps_cfg.get("cold_start_drift_m", 0))
    cold_start_decay_s = float(gps_cfg.get("cold_start_decay_s", 30))

    # Tunnel blackout (segments de NaN GPS)
    blackout_count = int(gps_cfg.get("blackout_count", 0))
    blackout_min_s = float(gps_cfg.get("blackout_min_s", 5))
    blackout_max_s = float(gps_cfg.get("blackout_max_s", 30))

    gps_mask = d0["lat"].notna()
    if gps_mask.sum() > 0:
        rng_gps = np.random.default_rng(45)
        n_gps = gps_mask.sum()
        gps_indices = d0.index[gps_mask]

        # HDOP-correlated position noise: σ = sigma_pos_m * hdop / 1.5
        hdop_vals = d0.loc[gps_mask, "hdop"].fillna(1.5).values
        sigma_per_tick_m = sigma_pos_m * hdop_vals / 1.5
        sigma_lat_deg = sigma_per_tick_m / 111_000
        sigma_lon_deg = sigma_per_tick_m / (111_000 * np.cos(np.radians(49.0)))
        d0.loc[gps_mask, "lat"] += rng_gps.normal(0, 1, n_gps) * sigma_lat_deg
        d0.loc[gps_mask, "lon"] += rng_gps.normal(0, 1, n_gps) * sigma_lon_deg

        # Sauts GPS (multipath / coupure tunnel)
        n_jumps = int(n_gps * jump_prob)
        if n_jumps > 0:
            jump_idx = rng_gps.choice(gps_indices, size=n_jumps, replace=False)
            jump_deg = jump_max_m / 111_000
            d0.loc[jump_idx, "lat"] += rng_gps.uniform(-jump_deg, jump_deg, n_jumps)
            d0.loc[jump_idx, "lon"] += rng_gps.uniform(-jump_deg, jump_deg, n_jumps)

        # Cold start drift : biais décroissant sur les premières secondes
        if cold_start_drift_m > 0 and cold_start_decay_s > 0:
            drift_deg = cold_start_drift_m / 111_000
            # Compute time offset from first GPS tick (in seconds)
            ts_col = d0["ts"]
            ts_parsed = pd.to_datetime(ts_col, utc=True, errors="coerce")
            t0 = ts_parsed.loc[gps_indices[0]]
            dt_s = (ts_parsed.loc[gps_mask] - t0).dt.total_seconds().values
            decay = np.exp(-dt_s / cold_start_decay_s)
            # Random fixed direction for the drift
            drift_angle = rng_gps.uniform(0, 2 * np.pi)
            d0.loc[gps_mask, "lat"] += drift_deg * decay * np.cos(drift_angle)
            d0.loc[gps_mask, "lon"] += drift_deg * decay * np.sin(drift_angle) / np.cos(np.radians(49.0))

        # Bruit vitesse
        d0.loc[gps_mask, "speed_mps"] += rng_gps.normal(0, sigma_speed, n_gps).astype("float32")
        d0.loc[gps_mask, "speed_mps"] = d0.loc[gps_mask, "speed_mps"].clip(0)

        # Tunnel blackout : mettre lat/lon/speed/heading à NaN sur des segments
        if blackout_count > 0:
            hz_est = float((ctx.cfg if isinstance(ctx.cfg, dict) else {}).get("sensors", {}).get("gps_hz", 10) or 10)
            for _ in range(blackout_count):
                dur_s = rng_gps.uniform(blackout_min_s, blackout_max_s)
                dur_ticks = int(dur_s * hz_est)
                max_start = max(0, len(gps_indices) - dur_ticks)
                if max_start <= 0:
                    continue
                start_pos = rng_gps.integers(0, max_start)
                bo_idx = gps_indices[start_pos : start_pos + dur_ticks]
                d0.loc[bo_idx, ["lat", "lon", "speed_mps"]] = np.nan
                if "heading_deg" in d0.columns:
                    d0.loc[bo_idx, "heading_deg"] = np.nan
                if "hdop" in d0.columns:
                    d0.loc[bo_idx, "hdop"] = np.nan
                if "n_satellites" in d0.columns:
                    d0.loc[bo_idx, "n_satellites"] = np.nan

    # Accéléromètre (obligatoire)
    for src, dst in [("acc_x", "ax_mps2"), ("acc_y", "ay_mps2"), ("acc_z", "az_mps2")]:
        if src in df.columns:
            d0[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            d0[dst] = 0.0

    # Gyroscope (optionnel — absent si désactivé)
    for src, dst in [("gyro_x", "gx_rad_s"), ("gyro_y", "gy_rad_s"), ("gyro_z", "gz_rad_s")]:
        if src in df.columns and not df[src].isna().all():
            d0[dst] = pd.to_numeric(df[src], errors="coerce")

    # Écrire le D0 (signal pur — pas de device_id/trip_id dans le parquet)
    d0_parquet = os.path.join(outdir, "d0.parquet")
    d0.to_parquet(d0_parquet, index=False)

    logger.info("[D0 Export] %d lignes, %d colonnes → %s", len(d0), len(d0.columns), d0_parquet)

    # Manifest (métadonnées d'enveloppe — séparé du signal)
    cfg_dict = ctx.cfg if isinstance(ctx.cfg, dict) else {}
    tele_cfg = cfg_dict.get("telemachus", {}) or {}
    sensors_cfg = cfg_dict.get("sensors", {}) or {}

    manifest = {
        "format": "telemachus-d0",
        "version": "0.2",
        "device_id": tele_cfg.get("vehicle_id", cfg_dict.get("vehicle_id", "RS3-SIM")),
        "trip_id": tele_cfg.get("trip_id", f"RS3_{os.path.basename(outdir)}"),
        "source": "RoadSimulator3",
        "sensors": {
            "gps_hz": sensors_cfg.get("gps_hz", 10),
            "imu_hz": sensors_cfg.get("imu_hz", 10),
            "gyro_enabled": sensors_cfg.get("gyro_enabled", True),
        },
        "gps_noise": {
            "sigma_pos_m": sigma_pos_m,
            "sigma_speed_mps": sigma_speed,
            "jump_probability": jump_prob,
            "cold_start_drift_m": cold_start_drift_m,
            "cold_start_decay_s": cold_start_decay_s,
            "blackout_count": blackout_count,
            "blackout_min_s": blackout_min_s,
            "blackout_max_s": blackout_max_s,
        },
        "n_points": len(d0),
        "columns": list(d0.columns),
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Ground truth (RS3 seulement — pour validation P014)
    gt = {}
    if ctx.meta.get("device_rotation_deg"):
        gt["device_rotation"] = ctx.meta["device_rotation_deg"]
    if isinstance(cfg_dict.get("inject_events"), dict):
        gt["inject_events"] = cfg_dict["inject_events"]
    if isinstance(cfg_dict.get("sensors"), dict):
        gt["sensors"] = cfg_dict["sensors"]
    if gt:
        gt_path = os.path.join(outdir, "ground_truth.json")
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2, default=str)

    return d0_parquet


class Exporter:
    """
    Exporte:
      - CSV 'timeline.csv' (source brute, utile hors CORS)
      - JSON 'artifacts.json'
      - JSON 'meta.json'
      - HTML 'map.html' (embedded, file:// OK)
      - HTML 'report.html' (inline charts, file:// OK)
    """
    name = "Exporter"

    def run(self, ctx: Context) -> Result:
        outdir = ctx.cfg.get("outdir")
        if not outdir:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            outdir = f"data/simulations/simulated_{ts}"
        os.makedirs(outdir, exist_ok=True)

        try:
            df = ctx.df
            # Harmonise la timeline + colonnes minimales
            if isinstance(ctx.cfg, dict):
                df = _apply_start_time(df, ctx.cfg)
            df = _ensure_event_column(df)

            # Schéma RS3 (si dispo)
            schema = _load_schema(ctx.cfg if isinstance(ctx.cfg, dict) else {})
            if schema:
                df = _apply_schema(df, schema)

            # Sauvegardes tabulaires / méta
            csv_path = os.path.join(outdir, "timeline.csv")
            df.to_csv(csv_path, index=False)

            # ── D0 export (RFC-0013 compliant) ──────────────────────────
            _export_d0(ctx, df, outdir)

            art = {k: _sanitize(v) for k, v in ctx.artifacts.items()}
            with open(os.path.join(outdir, "artifacts.json"), "w", encoding="utf-8") as f:
                json.dump(art, f, ensure_ascii=False, indent=2)

            meta = _sanitize({**ctx.meta, "outdir": outdir})
            with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Rapports (100% inline → pas de CORS)
            try:
                rep_cfg = {}
                if isinstance(ctx.cfg, dict):
                    rep_cfg = (ctx.cfg.get("exporter", {}).get("report", {})) or {}
                report_path, map_path = _generate_outputs(ctx, df, outdir, cfg=rep_cfg)
                if report_path:
                    print(f"[Report] HTML → {report_path}")
                if map_path:
                    print(f"[Report] Map  → {map_path}")
            except Exception as e:
                logger.debug("[Exporter] Rapport non généré: %s", e)

            # -- Telemachus inline export (option B: unit conversion only)
            try:
                tele_cfg = {}
                if isinstance(ctx.cfg, dict):
                    tele_cfg = (ctx.cfg.get("telemachus") or {})
                _export_telemachus_inline(ctx, df, outdir, tele_cfg)
            except Exception as e:
                logger.exception("[Exporter] Telemachus export error: %s", e)

            # Met à jour le df dans le contexte (post-cast schéma)
            ctx.df = df
            return Result((True, "OK"))
        except Exception as e:
            logger.exception("[Exporter] Crash: %s", e)
            return Result((False, f"Exception in Exporter: {e}"))