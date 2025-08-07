import geopandas as gpd

ZONES_GEOJSON = 'data/zone_typology.geojson'
ZONES_GDF = None

def load_zones():
    global ZONES_GDF
    if ZONES_GDF is None:
        try:
            ZONES_GDF = gpd.read_file(ZONES_GEOJSON)
            print("[INFO] zone_typology.geojson chargé avec succès.")
        except Exception as e:
            print(f"[WARNING] Impossible de charger {ZONES_GEOJSON} : {e}")
            ZONES_GDF = None

@deprecated
def enrich_with_h3_and_zones(df, config):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    load_zones()
    if ZONES_GDF is None:
        print("[WARNING] Pas de données de zones, enrichissement ignoré.")
        return df
    # Logique d'enrichissement H3 + zones
    ...
