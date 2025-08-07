import logging
import osmium
import networkx as nx
from shapely.geometry import LineString
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("osmium-pbf")

PBF_PATH = "data/osmnx/haute-normandie-latest.osm.pbf"
OUT_GRAPHML = "data/osmnx/graph_haute_normandieC.graphml"
OUT_PICKLE = OUT_GRAPHML.replace(".graphml", ".graph.pkl")


class RoadGraphHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.G = nx.DiGraph()

    @deprecated
    def way(self, w):
        logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
        highway = w.tags.get("highway")
        if not highway:
            return

        oneway = w.tags.get("oneway", "no")
        maxspeed = w.tags.get("maxspeed")
        name = w.tags.get("name")

        for i in range(len(w.nodes) - 1):
            u = w.nodes[i]
            v = w.nodes[i + 1]
            u_id = u.ref
            v_id = v.ref
            u_coords = (u.lon, u.lat)
            v_coords = (v.lon, v.lat)

            self.G.add_node(u_id, x=u.lon, y=u.lat)
            self.G.add_node(v_id, x=v.lon, y=v.lat)

            self.G.add_edge(
                u_id, v_id,
                highway=highway,
                maxspeed=maxspeed or "",
                name=name or "",
                geometry=LineString([u_coords, v_coords])
            )

            if oneway in ["no", "false", "0"]:
                self.G.add_edge(
                    v_id, u_id,
                    highway=highway,
                    maxspeed=maxspeed or "",
                    name=name or "",
                    geometry=LineString([v_coords, u_coords])
                )


logger.info(f"📥 Lecture du fichier PBF : {PBF_PATH}")
handler = RoadGraphHandler()
handler.apply_file(PBF_PATH, locations=True)
G = handler.G
G.graph["crs"] = "EPSG:4326"

logger.info(f"✅ Graphe construit : {len(G.nodes)} nœuds / {len(G.edges)} arêtes")

# Nettoyage des attributs non sérialisables
for u, v, data in G.edges(data=True):
    if "geometry" in data:
        del data["geometry"]

# Export
os.makedirs(os.path.dirname(OUT_GRAPHML), exist_ok=True)

logger.info(f"💾 Sauvegarde GraphML : {OUT_GRAPHML}")
nx.write_graphml(G, OUT_GRAPHML)

logger.info(f"💾 Sauvegarde Pickle : {OUT_PICKLE}")
nx.write_gpickle(G, OUT_PICKLE)

logger.info("✅ Export terminé avec succès.")