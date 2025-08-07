import logging
import time
import requests
from typing import List
import pandas as pd
from sseclient import SSEClient
from tqdm import tqdm

from core.osmnx.mapping import get_edge_type_nearest

logger = logging.getLogger(__name__)
OSMNX_BASE_URL = "http://localhost:5002/nearest_road_batch_stream"


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

    client = SSEClient(stream_url)
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

    for event in tqdm(client, total=n, desc="⏳ Enrichissement road_type"):
        try:
            if not event.data:
                continue
            data = eval(event.data) if isinstance(event.data, str) else event.data
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