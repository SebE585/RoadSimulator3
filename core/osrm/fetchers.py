# core/osrm/fetchers.py (simplifié)
from typing import Any, Iterable, List, Tuple
import pandas as pd

try:
    # Optionnel selon ta version
    from core.osrm.client import decode_polyline  # type: ignore
except Exception:  # pragma: no cover
    decode_polyline = None  # type: ignore

from core.osrm.client import get_route_from_coords


def _normalize_points(points: Iterable[Any]) -> List[Tuple[float, float]]:
    """Convertit une séquence quelconque de points en paires (lon, lat).
    Accepte: [[lon,lat], [lat,lon], [lon,lat,alt], ( (lon,lat), ...), {"lon":..,"lat":..}].
    """
    out: List[Tuple[float, float]] = []
    for pt in points:
        lon = lat = None
        if isinstance(pt, dict) and "lon" in pt and "lat" in pt:
            lon, lat = float(pt["lon"]), float(pt["lat"])
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            a, b = pt[0], pt[1]
            # Autoriser des points imbriqués ou surdimensionnés (ex: [lon,lat,alt])
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    a, b = a[0], a[1]
                elif isinstance(b, (list, tuple)) and len(b) >= 2:
                    a, b = b[0], b[1]
            ax, by = float(a), float(b)
            # Heuristique: si le premier ressemble à une latitude, on inverse
            if abs(ax) <= 90 and abs(by) <= 180:
                lat, lon = ax, by
            else:
                lon, lat = ax, by
        if lon is not None and lat is not None:
            out.append((lon, lat))
    return out


def _extract_lonlat_pairs(container: Any) -> List[Tuple[float, float]]:
    """Extrait une liste [(lon,lat),...] depuis les formats connus renvoyés par le client OSRM.
    Formats supportés:
      • DataFrame (déjà géré ailleurs)
      • dict avec clé top-level "coordinates": {"coordinates": [[lon,lat], ...]}
      • dict GeoJSON OSRM: {"routes":[{"geometry": {"coordinates":[[lon,lat],...]}}]}
      • dict OSRM avec polyline encodé: {"routes":[{"geometry":"<polyline>"}]}
      • list/tuple de points
      • tuple (payload, meta) → on lit payload
    """
    # Déballer (payload, meta)
    if isinstance(container, tuple) and len(container) >= 1:
        container = container[0]

    # 1) dict top-level {"coordinates": ...}
    if isinstance(container, dict) and "coordinates" in container:
        return _normalize_points(container["coordinates"])

    # 2) dict OSRM classiqe {routes:[{geometry: ...}]}
    if isinstance(container, dict) and "routes" in container:
        try:
            route0 = container["routes"][0]
            geom = route0.get("geometry")
            if isinstance(geom, dict) and "coordinates" in geom:
                return _normalize_points(geom["coordinates"])  # GeoJSON
            if isinstance(geom, str) and decode_polyline:
                return _normalize_points(decode_polyline(geom))  # polyline
            # Fallback legs/steps
            legs = route0.get("legs") or []
            if legs and isinstance(legs, list) and legs[0].get("steps"):
                acc: List[Tuple[float, float]] = []
                for st in legs[0]["steps"]:
                    g = st.get("geometry")
                    if isinstance(g, dict) and "coordinates" in g:
                        acc.extend(_normalize_points(g["coordinates"]))
                    elif isinstance(g, str) and decode_polyline:
                        acc.extend(_normalize_points(decode_polyline(g)))
                return acc
        except Exception:
            pass

    # 3) liste / tuple de points
    if isinstance(container, (list, tuple)):
        return _normalize_points(container)

    # Format inconnu
    return []


def fetch_route_standard(*, cities_coords, **cfg) -> pd.DataFrame:
    """Fetcher standard pour le pipeline RS3.
    Appelle la fonction existante `get_route_from_coords(cities_coords)` et renvoie
    un DataFrame [lat, lon]. Aucune autre dépendance n'est imposée au client.
    """
    res = get_route_from_coords(cities_coords)

    # Directement un DataFrame ?
    if isinstance(res, pd.DataFrame):
        df = res
    else:
        pairs = _extract_lonlat_pairs(res)
        if not pairs:
            raise ValueError("fetch_route_standard: aucune coordonnée exploitable renvoyée par le client OSRM")
        # (lon,lat) → DataFrame [lat, lon]
        df = pd.DataFrame(({"lat": lat, "lon": lon} for lon, lat in pairs), columns=["lat", "lon"])  # type: ignore

    if df.empty:
        raise ValueError("fetch_route_standard: DataFrame vide")
    return df