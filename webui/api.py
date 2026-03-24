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
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

# Ajoute RS3 root au path pour import core2/runner
RS3_ROOT = str(Path(__file__).resolve().parent.parent)
if RS3_ROOT not in sys.path:
    sys.path.insert(0, RS3_ROOT)

app = FastAPI(title="RS3 Simulator API", version="0.1.0")

# Store des jobs (en mémoire pour le POC)
_jobs: dict[str, dict[str, Any]] = {}


@app.post("/simulate")
async def simulate(cfg: dict[str, Any]) -> JSONResponse:
    """Lance une simulation RS3 synchrone et retourne le résultat."""
    from core2.context import Context
    from runner.run_simulation2 import build_pipeline

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
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    outdir = Path(job.get("outdir", ""))
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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rs3-simulator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8100, reload=True)
