import logging
import time
import json
import requests
import pandas as pd
from tqdm import tqdm

from core.osmnx.mapping import get_edge_type_nearest

logger = logging.getLogger(__name__)
OSMNX_BASE_URL = "http://localhost:5002/nearest_road_batch_stream"


def _iter_sse(url: str, timeout: int = 300):
    """Minimal SSE line parser using requests.iter_lines.

    Yields dicts with a single key 'data' containing the concatenated data lines
    for each SSE event. This avoids third-party client issues (bytes/str mix).
    """
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
            # Blank line separates events
            if not line:
                if buffer:
                    yield {"data": "\n".join(buffer)}
                    buffer = []
                continue
            # Comments start with ':' per SSE spec – ignore
            if line.startswith(":"):
                continue
            # Only collect data lines (we ignore 'event:' etc. for now)
            if line.startswith("data:"):
                buffer.append(line[5:].lstrip())
        # Flush last event if stream ends without trailing blank line
        if buffer:
            yield {"data": "\n".join(buffer)}


def enrich_road_type_stream(
    df: pd.DataFrame,
    stream_url_base: str = OSMNX_BASE_URL,
    max_attempts: int = 5,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Enrichit un DataFrame contenant des colonnes 'lat' et 'lon' en ajoutant :
    - 'road_type'      : type standardisé (ex: 'residential', 'tertiary', etc.)
    - 'osm_highway'    : valeur brute retournée par OSM (ex: 'primary_link')

    Les données sont obtenues en appelant une API locale via un flux SSE
    (`/nearest_road_batch_stream/start` puis `/stream/<id>`).

    Args:
        df (pd.DataFrame): Doit contenir les colonnes 'lat' et 'lon'.
        stream_url_base (str): Base URL de l’API SSE locale (par défaut `http://localhost:5002/...`).
        max_attempts (int): Nombre maximal de tentatives en cas d’erreur réseau.
        delay (float): Délai (en secondes) entre deux tentatives.

    Returns:
        pd.DataFrame: Le DataFrame original, avec deux colonnes enrichies :
                      'road_type' et 'osm_highway'.
    """
    df = df.copy()
    df["target_speed"] = None
    latitudes = df["lat"].tolist()
    longitudes = df["lon"].tolist()
    n = len(df)

    df["road_type"] = None
    df["osm_highway"] = None

    # 🔹 Étape 1 — Lancement de la session
    stream_url_start = f"{stream_url_base}/start"
    stream_url_template = f"{stream_url_base}/stream/{{stream_id}}"
    payload = {"lat": latitudes, "lon": longitudes}

    stream_id = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"[OSMNX] Initialisation du stream avec {n} points...")
            response = requests.post(stream_url_start, json=payload, timeout=30)
            response.raise_for_status()
            stream_id = response.json().get("stream_id")
            if not stream_id:
                raise ValueError("Aucun stream_id reçu du serveur.")
            break
        except requests.RequestException as e:
            logger.warning(f"[Tentative {attempt}/{max_attempts}] Échec : {e}")
            time.sleep(delay)
    else:
        raise RuntimeError(f"[OSMNX] Échec de la connexion SSE après {max_attempts} tentatives.")

    # 🔹 Étape 2 — Lecture du flux
    stream_url = stream_url_template.format(stream_id=stream_id)
    logger.info(f"[OSMNX] Connexion à {stream_url}...")

    iterable = _iter_sse(stream_url)
    count = 0

    speed_by_type = {
        "residential": 30,
        "secondary": 50,
        "primary": 70,
        "motorway": 110,
        "tertiary": 40,
        "unclassified": 50,
        "service": 20,
        "unknown": 50
    }

    for event in tqdm(iterable, total=n, desc="⏳ Enrichissement road_type"):
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

            # ✅ Compatibilité highway / osm_highway
            raw_highway = data.get("osm_highway", data.get("highway", None))

            road_type = get_edge_type_nearest({"highway": raw_highway})

            df.at[idx, "road_type"] = road_type
            df.at[idx, "osm_highway"] = raw_highway or "unknown"
            df.at[idx, "target_speed"] = speed_by_type.get(road_type, 50)

            logger.debug(f"[DEBUG] Index {idx} : highway={raw_highway} → road_type={road_type}")
            count += 1
        except Exception as e:
            logger.error(f"[OSMNX] Erreur de parsing SSE à l’index {count} : {e}")

        if count >= n:
            break

    return df