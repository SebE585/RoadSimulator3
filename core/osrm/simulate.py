"""
simulate.py – Pipeline principal de simulation via OSRM.

Enchaîne l’appel OSRM, l’interpolation à pas fixe, la génération des timestamps,
et retourne un DataFrame exploitable pour enrichissement inertiel.
"""

from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple, Optional
from .client import get_route_from_coords

def simulate_route_via_osrm(cities_coords: List[Tuple[float, float]], hz: int = 10, start_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Génère un DataFrame simulé à partir d’un itinéraire OSRM interpolé à pas fixe.

    Args:
        cities_coords (List[Tuple[float, float]]): Liste ordonnée des coordonnées (lat, lon) des villes ou points à relier.
        hz (int, optional): Fréquence d’échantillonnage en Hz. Défaut : 10.
        start_time (Optional[datetime], optional): Date et heure de début de la simulation. Si None, prend la date courante.

    Returns:
        pd.DataFrame: DataFrame contenant les colonnes :
            - lat (float) : Latitude
            - lon (float) : Longitude
            - timestamp (datetime) : Horodatage simulé à fréquence fixe
            - road_type (str) : Type de route extrait depuis OSM

    Raises:
        ValueError: Si la liste est vide, contient des NaN ou des coordonnées hors limites géographiques.
    """
    if start_time is None:
        start_time = datetime.now()

    if not cities_coords or len(cities_coords) < 2:
        raise ValueError("La liste des villes est vide ou trop courte.")
    if any(pd.isna(lat) or pd.isna(lon) for lat, lon in cities_coords):
        raise ValueError("Coordonnées contenant NaN.")
    if any(not (-90 <= lat <= 90 and -180 <= lon <= 180) for lat, lon in cities_coords):
        raise ValueError("Coordonnées hors bornes géographiques.")

    print(f"[SIMULATE] Récupération du trajet OSRM entre {len(cities_coords)} points...")
    _, osrm_coords = get_route_from_coords(cities_coords)
    print(f"[DEBUG] Nombre de points bruts retournés par OSRM : {len(osrm_coords)}")
    if len(osrm_coords) < 2:
        raise ValueError("OSRM a retourné un trajet trop court (moins de 2 points).")

    from core.osrm.interpolation import interpolate_route_at_fixed_step
    df = pd.DataFrame(osrm_coords, columns=["lat", "lon"])
    step_m = 0.83  # default interpolation step in meters
    df_interp = pd.DataFrame(interpolate_route_at_fixed_step(df[["lat", "lon"]].to_records(index=False).tolist(), step_m=step_m), columns=["lat", "lon"])

    timestamps = [start_time + timedelta(seconds=i * (1 / hz)) for i in range(len(df_interp))]

    df_interp["timestamp"] = timestamps

    from geopy.distance import geodesic
    distance_km = sum(geodesic(df_interp.iloc[i][["lat", "lon"]], df_interp.iloc[i+1][["lat", "lon"]]).km for i in range(len(df_interp)-1))
    print(f"[DEBUG] Distance totale estimée du trajet : {distance_km:.2f} km")

    from core.osmnx.client import enrich_road_type_stream
    df_osrm = enrich_road_type_stream(df_interp)

    print(f"[SIMULATE] ✅ {len(df_osrm)} points interpolés générés à {hz} Hz.")

    # Ajout du calcul du heading avant l'export CSV
    from core.kinematics import compute_heading
    lat1 = df_osrm["lat"][:-1].to_numpy()
    lon1 = df_osrm["lon"][:-1].to_numpy()
    lat2 = df_osrm["lat"][1:].to_numpy()
    lon2 = df_osrm["lon"][1:].to_numpy()

    headings = compute_heading(lat1, lon1, lat2, lon2)
    df_osrm["heading"] = [headings[0]] + list(headings)  # égalise la taille en dupliquant le premier heading

    # Export CSV output_osrm_trajectory.csv in RS3_OUTPUT_DIR or latest simulation dir
    import os
    from core.utils import get_latest_simulation_dir, get_simulation_output_dir, save_dataframe_as_csv

    output_dir = os.environ.get("RS3_OUTPUT_DIR")
    if output_dir is None:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        output_dir = get_simulation_output_dir(timestamp)

    output_path = os.path.join(output_dir, "output_osrm_trajectory.csv")
    save_dataframe_as_csv(df_osrm, output_path)
    print(f"[EXPORT] Trajectoire OSRM exportée : {output_path}")

    return df_osrm


# Nouvelle fonction : simulation à partir des événements
def simulate_route_via_osrm_from_events(df: pd.DataFrame, hz: int = 10, start_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Reconstruit un trajet complet via OSRM en utilisant les points marqués par des événements,
    puis utilise les points bruts OSRM à fréquence fixe.

    Args:
        df (pd.DataFrame): DataFrame source contenant au moins les colonnes 'lat', 'lon', 'event'.
        hz (int): Fréquence d’échantillonnage souhaitée.
        start_time (Optional[datetime], optional): Date et heure de début de la simulation. Si None, prend la date courante.

    Returns:
        pd.DataFrame: Trajet brut OSRM avec horodatage et événements repositionnés.
    """
    if start_time is None:
        start_time = datetime.now()

    if "event" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'event'.")

    df = df.copy()
    df["latlon"] = list(zip(df["lat"], df["lon"]))

    # Points avec événements
    points_event = df[df["event"].notna()][["latlon"]]

    # Points tous les 500 mètres environ (~1 point tous les 600 index à 0.83m)
    sampling_interval = 600
    sampled_idx = list(range(0, len(df), sampling_interval))
    points_regular = df.iloc[sampled_idx][["latlon"]]

    # Concaténer et dédupliquer
    key_points = pd.concat([points_event, points_regular]).drop_duplicates()
    event_points = key_points["latlon"].tolist()
    event_points = [(lat, lon) for lat, lon in event_points]

    if len(event_points) < 2:
        raise ValueError("Pas assez de points événements pour reconstruire un trajet.")

    print(f"[SIMULATE] OSRM recalculé sur {len(event_points)} points (événements + régulier).")

    # Découpage en segments pour appel OSRM par morceaux
    MAX_SEGMENT_POINTS = 200
    segments = [event_points[i:i+MAX_SEGMENT_POINTS]
                for i in range(0, len(event_points) - 1, MAX_SEGMENT_POINTS - 1)]

    dfs = []
    for segment in segments:
        if len(segment) < 2:
            continue
        try:
            df_part = simulate_route_via_osrm(segment, hz=hz, start_time=start_time)
            dfs.append(df_part)
        except Exception as e:
            print(f"[ERROR] Segment skipped: {e}")

    if not dfs:
        raise RuntimeError("Aucun segment OSRM n'a pu être simulé.")
    df_interp = pd.concat(dfs, ignore_index=True)

    from core.kinematics import recompute_speed
    df_interp = recompute_speed(df_interp)

    from geopy.distance import geodesic

    # Réinjection des événements par proximité géographique (tolérance : 3 m)
    events = df.dropna(subset=["event"])[["lat", "lon", "event"]].drop_duplicates()
    df_interp["event"] = None

    for i, interp_row in df_interp.iterrows():
        lat_interp, lon_interp = interp_row["lat"], interp_row["lon"]
        for _, event_row in events.iterrows():
            lat_ev, lon_ev = event_row["lat"], event_row["lon"]
            if geodesic((lat_interp, lon_interp), (lat_ev, lon_ev)).meters < 3:
                df_interp.at[i, "event"] = event_row["event"]
                break

    df_interp.drop(columns=["latlon"], inplace=True)

    from core.osmnx.client import enrich_road_type_stream
    df_interp = enrich_road_type_stream(df_interp)
    return df_interp
