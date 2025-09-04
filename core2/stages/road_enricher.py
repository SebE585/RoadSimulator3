# core2/stages/road_enricher.py
from __future__ import annotations

import logging
import time
import json
from typing import Optional, Dict, Any

import pandas as pd

from ..contracts import Result
from ..context import Context

# Déps réseau (requests) et barre de progression (tqdm) — tqdm est optionnel
try:
    import requests  # type: ignore
except Exception as _e:  # pragma: no cover
    requests = None  # fallback sans réseau

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, total=None, desc=None):  # type: ignore
        return x

# Mapping OSM → type canonique
try:
    from core.osmnx.mapping import get_edge_type_nearest  # type: ignore
except Exception:
    # Fallback minimal si le mapping n'est pas disponible
    def get_edge_type_nearest(attrs: Dict[str, Any]) -> str:  # type: ignore
        raw = (attrs or {}).get("highway") or (attrs or {}).get("osm_highway")
        return str(raw or "unknown")


logger = logging.getLogger(__name__)
DEFAULT_STREAM_BASE = "http://localhost:5002/nearest_road_batch_stream"


def _iter_sse(url: str, timeout: int = 300):
    """
    Minimal SSE parser based on requests.iter_lines.
    Yields dicts with 'data' entries (joined multi-line).
    """
    if requests is None:
        raise RuntimeError("requests module not available")

    with requests.get(
        url,
        stream=True,
        timeout=timeout,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        buffer = []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                if buffer:
                    yield {"data": "\n".join(buffer)}
                    buffer = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                buffer.append(line[5:].lstrip())
        if buffer:
            yield {"data": "\n".join(buffer)}


def enrich_road_type_stream(
    df: pd.DataFrame,
    stream_url_base: str = DEFAULT_STREAM_BASE,
    max_attempts: int = 5,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Enrichit un DataFrame (lat/lon) avec:
      - road_type (canonique)
      - osm_highway (valeur brute OSM)
      - target_speed (km/h, heuristique simple par type)
    via un service SSE local:
      POST   {base}/start
      GET    {base}/stream/{stream_id}
    """
    if df is None or df.empty:
        return df
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("enrich_road_type_stream: colonnes 'lat' et 'lon' requises")

    latitudes = df["lat"].tolist()
    longitudes = df["lon"].tolist()
    n = len(df)

    df = df.copy()
    df["road_type"] = None
    df["osm_highway"] = None
    df["target_speed"] = None

    if requests is None:
        logger.warning("[RoadEnricher] 'requests' indisponible — fallback local.")
        return df

    # 1) Démarrage du stream
    stream_url_start = f"{stream_url_base.rstrip('/')}/start"
    stream_url_template = f"{stream_url_base.rstrip('/')}/stream/{{stream_id}}"
    payload = {"lat": latitudes, "lon": longitudes}

    stream_id: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"[OSMNX] Démarrage stream avec {n} points → {stream_url_start}")
            response = requests.post(stream_url_start, json=payload, timeout=30)
            response.raise_for_status()
            stream_id = response.json().get("stream_id")
            if not stream_id:
                raise ValueError("Aucun stream_id reçu du serveur.")
            break
        except Exception as e:
            logger.warning(f"[OSMNX] Tentative {attempt}/{max_attempts} échec: {e}")
            time.sleep(delay)
    else:
        raise RuntimeError(f"[OSMNX] Échec de la connexion SSE après {max_attempts} tentatives")

    # 2) Lecture du flux
    stream_url = stream_url_template.format(stream_id=stream_id)
    logger.info(f"[OSMNX] Connexion SSE → {stream_url}")

    speed_by_type = {
        "residential": 30,
        "secondary": 50,
        "primary": 70,
        "motorway": 110,
        "tertiary": 40,
        "unclassified": 50,
        "service": 20,
        "unknown": 50,
    }

    count = 0
    for event in tqdm(_iter_sse(stream_url), total=n, desc="⏳ road_type"):
        try:
            payload = event.get("data", "")
            if isinstance(payload, str) and payload.strip():
                try:
                    data = json.loads(payload)
                except Exception as e:
                    logger.error(f"[OSMNX] JSON decode error: {e} — raw: {payload[:200]}")
                    continue
            else:
                data = payload

            idx = data["index"]
            raw_highway = data.get("osm_highway", data.get("highway", None))
            road_type = get_edge_type_nearest({"highway": raw_highway})

            df.at[idx, "road_type"] = road_type
            df.at[idx, "osm_highway"] = raw_highway or "unknown"
            df.at[idx, "target_speed"] = speed_by_type.get(road_type, 50)

            logger.debug(f"[OSMNX] idx={idx} highway={raw_highway} → road_type={road_type}")
            count += 1
        except Exception as e:
            logger.error(f"[OSMNX] Erreur parsing SSE (idx~{count}): {e}")
        if count >= n:
            break

    return df


class RoadEnricher:
    """
    Branche l'enrichissement road_type via core.osmnx (SSE).
    - lit la base URL dans cfg.road_enrich.stream_url_base (optionnel)
    - en cas d'erreur réseau, ne casse pas la pipeline (fallback)
    - ajoute: road_type, osm_highway, target_speed
    """
    name = "RoadEnricher"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")

        # Base URL configurable
        stream_base = (
            ctx.cfg.get("road_enrich", {}).get("stream_url_base", DEFAULT_STREAM_BASE)
            if isinstance(ctx.cfg, dict) else DEFAULT_STREAM_BASE
        )

        try:
            enriched = enrich_road_type_stream(df, stream_url_base=stream_base)
            # Merge: on préserve toutes les colonnes existantes et on ajoute/écrase les 3 colonnes d'enrichissement
            cols = ["road_type", "osm_highway", "target_speed"]
            # si enrich renvoie moins de lignes (ne devrait pas), on safe-merge par index
            for c in cols:
                if c in enriched.columns:
                    df[c] = enriched[c]
                else:
                    df[c] = df.get(c, None)
            ctx.df = df
            return Result()
        except Exception as e:
            logger.warning(f"[RoadEnricher] Fallback (erreur enrichissement): {e}")
            # Fallback: a minima road_type="unknown"
            df = df.copy()
            if "road_type" not in df.columns:
                df["road_type"] = "unknown"
            if "osm_highway" not in df.columns:
                df["osm_highway"] = "unknown"
            if "target_speed" not in df.columns:
                df["target_speed"] = None
            ctx.df = df
            return Result()