import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyrosm import OSM


def normalize_geometry(geom):
    if geom is None:
        return None
    if geom.geom_type == 'LineString':
        return geom
    if geom.geom_type == 'MultiLineString':
        return list(geom.geoms)[0]
    return None


class LocalOSMIndex:
    def __init__(self, pbf_path, network_type='driving'):
        print(f"[DEBUG] Initialisation de LocalOSMIndex avec {pbf_path} (network_type={network_type})")
        osm = OSM(pbf_path)
        self.edges = osm.get_network(network_type=network_type)

        if self.edges is None or self.edges.empty:
            raise ValueError(f"[ERROR] Aucune donnée routière extraite du fichier {pbf_path}")

        print(f"[DEBUG] {len(self.edges)} routes extraites.")
        self.edges['geometry'] = self.edges['geometry'].apply(normalize_geometry)
        self.edges = self.edges[self.edges['geometry'].notnull()]

        print("[DEBUG] Construction de l'index spatial STRtree...")
        self.tree = STRtree(self.edges['geometry'].tolist())
        self.geom_to_highway = dict(zip(self.edges['geometry'], self.edges['highway']))
        print("[DEBUG] Index spatial prêt.")

    def query_point(self, lat, lon, buffer_m=10):
        point = Point(lon, lat)
        buffer_deg = buffer_m / 111320
        candidates = self.tree.query(point.buffer(buffer_deg))
        candidates = [g for g in candidates if hasattr(g, 'distance')]

        if not candidates:
            return 'inconnu'

        nearest_geom = min(candidates, key=lambda g: g.distance(point))
        highway = self.geom_to_highway.get(nearest_geom, 'inconnu')
        return highway if isinstance(highway, str) else highway[0]

    def annotate_dataframe(self, df, buffer_m=10):
        print("[DEBUG] Annotation du DataFrame avec les types de route depuis PBF...")
        df['road_type'] = df.apply(
            lambda row: self.query_point(row['lat'], row['lon'], buffer_m), axis=1
        )
        print("[DEBUG] Annotation terminée.")
        return df


@deprecated
def enrich_road_type_pbf(df, pbf_index: LocalOSMIndex, buffer_m=10):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    print("[INFO] Attribution des types de routes via fichier PBF local...")
    return pbf_index.annotate_dataframe(df, buffer_m=buffer_m)
