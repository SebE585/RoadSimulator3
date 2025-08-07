import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def load_osm_zones(filepath):
    """
    Charge un fichier GeoJSON ou shapefile contenant les zones OSM annotées :
    Exemple de colonnes attendues : ['zone_type', 'name', 'geometry']
    """
    zones_gdf = gpd.read_file(filepath)
    print(f"[INFO] {len(zones_gdf)} zones chargées depuis {filepath}")
    return zones_gdf


def enrich_with_osm_zones(df, zones_gdf):
    """
    Enrichit le DataFrame avec les types de zones OSM (urbain, zone industrielle, piéton, etc.)
    en associant chaque point GPS à une zone par inclusion spatiale.
    """
    # Convertir df en GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    # Initialiser la colonne
    gdf['osm_zone'] = 'inconnu'

    # Parcourir les zones pour enrichir par jointure spatiale
    for idx, zone in zones_gdf.iterrows():
        mask = gdf.within(zone['geometry'])
        gdf.loc[mask, 'osm_zone'] = zone['zone_type']

    # Retirer la colonne geometry pour retour DataFrame classique
    gdf = gdf.drop(columns='geometry')

    return pd.DataFrame(gdf)
