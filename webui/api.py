"""
RS3 Web API — FastAPI backend pour simulate.roadsimulator3.fr

POST /simulate  → lance une simulation RS3, retourne le chemin de sortie
GET  /status/{job_id} → état d'un job async
GET  /download/{job_id}/{fmt} → télécharge le résultat (parquet, csv, json)
"""
from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Ajoute RS3 root au path pour import engine/runner
import os

RS3_ROOT = str(Path(__file__).resolve().parent.parent)
OSRM_URL = os.environ.get("OSRM_URL", "http://localhost:5003")
if RS3_ROOT not in sys.path:
    sys.path.insert(0, RS3_ROOT)

app = FastAPI(title="RS3 Simulator API", version="0.3.0")

# ── Rate limiting simple par IP ──────────────────────────────────────────────
_rate_store: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60  # secondes
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "10"))  # simulations par minute par IP


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Limite le nombre de simulations par IP (POST /simulate uniquement)."""
    import time
    if request.method == "POST" and "/simulate" in str(request.url):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        hits = _rate_store.get(ip, [])
        hits = [t for t in hits if now - t < RATE_LIMIT_WINDOW]
        if len(hits) >= RATE_LIMIT_MAX:
            from starlette.responses import JSONResponse as JR
            return JR({"detail": f"Rate limit: max {RATE_LIMIT_MAX} simulations per minute"}, status_code=429)
        hits.append(now)
        _rate_store[ip] = hits
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir le frontend statique
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

# Store des jobs (en mémoire pour le POC)
_jobs: dict[str, dict[str, Any]] = {}


def _resolve_outdir(job_id: str) -> Path:
    """Résout le dossier de sortie depuis job_id."""
    if job_id in _jobs:
        return Path(_jobs[job_id].get("outdir", ""))
    # Fallback : chercher dans data/simulations/
    candidate = Path(f"data/simulations/{job_id}")
    if candidate.exists():
        return candidate
    # Chercher par prefix WEB-
    for d in Path("data/simulations").glob("WEB-*"):
        if job_id in d.name:
            return d
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.post("/simulate")
async def simulate(cfg: dict[str, Any]) -> JSONResponse:
    """Lance une simulation RS3 synchrone et retourne le résultat."""
    from engine.context import Context
    from runner.simulate import build_pipeline

    job_id = uuid.uuid4().hex[:12]

    # Expand strftime
    now = datetime.now(timezone.utc)
    if isinstance(cfg.get("outdir"), str):
        try:
            cfg["outdir"] = now.strftime(cfg["outdir"])
        except Exception:
            pass

    try:
        ctx = Context(cfg=cfg)
        pipeline = build_pipeline(cfg)
        result = pipeline.run(ctx)

        outdir = ctx.meta.get("outdir", cfg.get("outdir", ""))
        _jobs[job_id] = {
            "status": "done" if result.ok else "error",
            "outdir": outdir,
            "message": result.msg,
        }

        return JSONResponse({
            "job_id": job_id,
            "status": "done" if result.ok else "error",
            "outdir": outdir,
            "message": result.msg,
        })

    except Exception as exc:
        logger.exception("Simulation failed")
        _jobs[job_id] = {"status": "error", "message": str(exc)}
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(_jobs[job_id])


@app.get("/download/{job_id}/{fmt}")
async def download(job_id: str, fmt: str) -> FileResponse:
    """Télécharge le résultat au format demandé (parquet, csv, json)."""
    outdir = _resolve_outdir(job_id)
    if not outdir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    ext_map = {"parquet": "*.parquet", "csv": "*.csv", "json": "*.json"}
    pattern = ext_map.get(fmt)
    if not pattern:
        raise HTTPException(status_code=400, detail=f"Format must be one of: {list(ext_map.keys())}")

    files = list(outdir.glob(pattern))

    # Si le format n'existe pas nativement, convertir depuis Parquet
    if not files and fmt in ("csv", "json"):
        import pandas as pd
        pq_files = list(outdir.glob("*.parquet"))
        if pq_files:
            src = pq_files[0]
            df = pd.read_parquet(src)
            if fmt == "csv":
                out = outdir / f"{src.stem}.csv"
                df.to_csv(out, index=False)
            else:
                out = outdir / f"{src.stem}.json"
                df.to_json(out, orient="records", date_format="iso")
            files = [out]

    if not files:
        raise HTTPException(status_code=404, detail=f"No {fmt} file found in output")

    f = files[0]
    media = {
        "parquet": "application/octet-stream",
        "csv": "text/csv",
        "json": "application/json",
    }
    return FileResponse(f, filename=f.name, media_type=media.get(fmt, "application/octet-stream"))


@app.get("/results/{job_id}/trace")
async def trace(job_id: str, hz: int = 1, source: str = "matched") -> JSONResponse:
    """Retourne la trace pour affichage carte.

    source=matched : trace map-matchée (legs_traces depuis artifacts.json) — route réelle
    source=raw     : trace GPS brute (d0.parquet) — points avec bruit
    """
    outdir = _resolve_outdir(job_id)
    if not outdir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    import json as json_mod

    # Trace map-matchée depuis artifacts.json (trajectoire routière réelle)
    if source == "matched":
        art_file = outdir / "artifacts.json"
        if art_file.exists():
            arts = json_mod.loads(art_file.read_text())
            legs_traces = arts.get("legs_traces", [])
            if legs_traces:
                # Assembler toutes les coords de tous les legs
                all_coords = []
                for leg in legs_traces:
                    for pt in leg:
                        all_coords.append([round(pt["lat"], 6), round(pt["lon"], 6)])

                # Sous-échantillonner
                stride = max(1, len(all_coords) // 2000)
                coords = all_coords[::stride]

                return JSONResponse({
                    "coords": coords,
                    "n_points": len(coords),
                    "n_legs": len(legs_traces),
                    "start": coords[0] if coords else None,
                    "end": coords[-1] if coords else None,
                    "source": "matched",
                })

    # Fallback : trace GPS brute depuis D0
    import pandas as pd
    df = _load_df(outdir)

    sim_hz = 10
    stride = max(1, sim_hz // hz)
    ds = df.iloc[::stride]

    lat_col = next((c for c in ["lat", "latitude"] if c in ds.columns), None)
    lon_col = next((c for c in ["lon", "lng", "longitude"] if c in ds.columns), None)
    if not lat_col or not lon_col:
        raise HTTPException(status_code=500, detail="No lat/lon columns found")

    valid = ds[lat_col].notna() & ds[lon_col].notna()
    pts = ds[valid]

    coords = list(zip(pts[lat_col].round(6).tolist(), pts[lon_col].round(6).tolist()))
    return JSONResponse({
        "coords": coords,
        "n_points": len(coords),
        "start": coords[0] if coords else None,
        "end": coords[-1] if coords else None,
        "source": "raw",
    })


@app.get("/results/{job_id}/sensors")
async def sensors(job_id: str, max_points: int = 3000) -> JSONResponse:
    """Retourne les signaux capteurs pour les graphiques."""
    outdir = _resolve_outdir(job_id)

    import pandas as pd
    df = _load_df(outdir)

    stride = max(1, len(df) // max_points)
    ds = df.iloc[::stride]

    result: dict[str, Any] = {"n_points": len(ds)}

    # Timestamp → secondes relatives (colonne "ts" ou "timestamp")
    ts_col = next((c for c in ["ts", "timestamp"] if c in ds.columns), None)
    if ts_col:
        ts = pd.to_datetime(ds[ts_col], utc=True, errors="coerce")
        t0 = ts.dropna().iloc[0] if ts.notna().any() else pd.Timestamp.now(tz="UTC")
        result["time_s"] = ((ts - t0).dt.total_seconds()).round(2).tolist()
    else:
        result["time_s"] = list(range(len(ds)))

    # Mapping colonnes D0 → noms normalisés pour le frontend
    col_map = {
        "acc_x": ["ax_mps2", "acc_x"],
        "acc_y": ["ay_mps2", "acc_y"],
        "acc_z": ["az_mps2", "acc_z"],
        "gyro_x": ["gx_rad_s", "gyro_x"],
        "gyro_y": ["gy_rad_s", "gyro_y"],
        "gyro_z": ["gz_rad_s", "gyro_z"],
        "speed": ["speed_mps", "speed"],
    }
    for out_name, candidates in col_map.items():
        src = next((c for c in candidates if c in ds.columns), None)
        if src:
            result[out_name] = ds[src].fillna(0).round(4).tolist()

    # time_s peut aussi contenir des NaN
    if "time_s" in result:
        result["time_s"] = [0 if pd.isna(v) else v for v in result["time_s"]]

    return JSONResponse(result)


@app.get("/results/{job_id}/meta")
async def meta(job_id: str) -> JSONResponse:
    """Retourne les métadonnées de la simulation (meta.json)."""
    outdir = _resolve_outdir(job_id)
    meta_file = outdir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")

    import json
    return JSONResponse(json.loads(meta_file.read_text()))


@app.get("/results/{job_id}/qa")
async def qa(job_id: str) -> JSONResponse:
    """Retourne le rapport QA (extraits d'artifacts.json)."""
    outdir = _resolve_outdir(job_id)
    art_file = outdir / "artifacts.json"
    if not art_file.exists():
        raise HTTPException(status_code=404, detail="artifacts.json not found")

    import json
    arts = json.loads(art_file.read_text())

    # Enrichir legs_summary avec vitesse moyenne calculée
    legs = arts.get("legs_summary", [])
    for leg in legs:
        d = leg.get("distance_m", 0)
        t = leg.get("duration_s", 0)
        leg["mean_speed_mps"] = d / t if t > 0 else 0

    return JSONResponse({
        "qa_pretty": arts.get("qa_pretty", {}),
        "qa_realism": arts.get("qa_realism", {}),
        "qa_checklist": arts.get("qa_checklist", {}),
        "imu_coherence": arts.get("imu_coherence", {}),
        "severity_criteria": arts.get("severity_criteria", {}),
        "legs_summary": legs,
    })


@app.get("/results/{job_id}/severity")
async def severity(job_id: str, max_points: int = 5000) -> JSONResponse:
    """Retourne les données pour le bi-histogramme de sévérité."""
    outdir = _resolve_outdir(job_id)

    import pandas as pd
    import numpy as np
    df = _load_df(outdir)

    # Colonnes accéléro
    ax_col = next((c for c in ["ax_mps2", "acc_x"] if c in df.columns), None)
    ay_col = next((c for c in ["ay_mps2", "acc_y"] if c in df.columns), None)
    speed_col = next((c for c in ["speed_mps", "speed"] if c in df.columns), None)

    if not ax_col or not ay_col:
        raise HTTPException(status_code=500, detail="No accelerometer columns found")

    # Filtrer vitesse > 1 m/s (pas à l'arrêt)
    mask = df[ax_col].notna() & df[ay_col].notna()
    if speed_col:
        mask = mask & (df[speed_col] > 1.0)
    filtered = df[mask]

    # Sous-échantillonner
    stride = max(1, len(filtered) // max_points)
    ds = filtered.iloc[::stride]

    # Convertir en mG (1 m/s² ≈ 101.97 mG)
    mg_factor = 1000.0 / 9.81
    gx = (ds[ay_col] * mg_factor).round(1).tolist()  # latéral
    gy = (ds[ax_col] * mg_factor).round(1).tolist()  # longitudinal

    # Charger meta pour rotation info
    meta_file = outdir / "meta.json"
    rotation = {}
    if meta_file.exists():
        import json
        m = json.loads(meta_file.read_text())
        rotation = m.get("device_rotation_deg", {})

    return JSONResponse({
        "gx_mg": gx,
        "gy_mg": gy,
        "n_points": len(gx),
        "rotation": rotation,
    })


def _load_df(outdir: Path):
    """Charge le DataFrame depuis le dossier de sortie."""
    import pandas as pd
    pq = list(outdir.glob("d0.parquet")) or list(outdir.glob("*.parquet"))
    csv = list(outdir.glob("timeline.csv")) or list(outdir.glob("*.csv"))
    if pq:
        return pd.read_parquet(pq[0])
    elif csv:
        return pd.read_csv(csv[0])
    raise HTTPException(status_code=404, detail="No data file found")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rs3-simulator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8100, reload=True)
