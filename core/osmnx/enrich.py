import pandas as pd
from .client import stream_osmnx_batch
from .mapping import get_edge_type_nearest


@deprecated
def enrich_road_type(df: pd.DataFrame) -> pd.DataFrame:
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Enrichit un DataFrame avec les colonnes 'road_type' et 'osm_highway'
    en interrogeant l'API OSMnx locale via `stream_osmnx_batch`.

    Pour chaque point (lat, lon), l'API retourne les attributs OSM du tronçon le plus proche,
    à partir desquels on déduit un `road_type` (résidentiel, secondaire, etc.).

    Args:
        df (pd.DataFrame): DataFrame contenant au minimum les colonnes ['lat', 'lon']

    Returns:
        pd.DataFrame: Le même DataFrame, avec 2 colonnes supplémentaires :
            - 'road_type'     : type générique standardisé (ex: 'residential', 'secondary', etc.)
            - 'osm_highway'   : valeur brute OSM (ex: 'primary_link', 'unclassified', etc.)
    """
    coords = df[["lat", "lon"]].values.tolist()
    results = stream_osmnx_batch(coords)

    road_types = []
    osm_highways = []

    for i in range(len(df)):
        info = results.get(i, {})
        raw_highway = info.get("highway", None)
        road_types.append(get_edge_type_nearest(raw_highway))
        osm_highways.append(raw_highway if raw_highway else "unknown")

    df["road_type"] = road_types
    df["osm_highway"] = osm_highways
    return df
