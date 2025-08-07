import logging
import osmium
import networkx as nx
from shapely.geometry import Point, LineString

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PBF_PATH = "data/osmnx/haute-normandie-latest.osm.pbf"
OUT_PATH = "data/osmnx/graph_haute_normandie.graphml"

class RoadGraphHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.G = nx.DiGraph()

    @deprecated
    def way(self, w):
        logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
        highway = w.tags.get("highway")
        maxspeed = w.tags.get("maxspeed")
        name = w.tags.get("name")

        if highway:
            coords = [(n.lon, n.lat) for n in w.nodes]
            if len(coords) >= 2:
                for u, v in zip(coords[:-1], coords[1:]):
                    self.G.add_edge(
                        u, v,
                        highway=highway or "unknown",
                        maxspeed=maxspeed or "unknown",
                        name=name or "",
                        geometry=LineString([u, v])
                    )
                    logger.debug(f"Added edge from {u} to {v} | highway: {highway}, maxspeed: {maxspeed}, name: {name}")

handler = RoadGraphHandler()
logger.info(f"Lecture du PBF : {PBF_PATH}")
handler.apply_file(PBF_PATH, locations=True)
G = handler.G

logger.info(f"Nombre de noeuds : {len(G.nodes)}")
logger.info(f"Nombre d'arêtes : {len(G.edges)}")

logger.info("Nettoyage des attributs non sérialisables")
for u, v, data in G.edges(data=True):
    for key in list(data.keys()):
        if data[key] is None:
            data[key] = ""
        elif key == 'geometry':
            data[key] = ""

logger.info(f"Sauvegarde du graphe au format GraphML : {OUT_PATH}")
nx.write_graphml(G, OUT_PATH)
logger.info("Graphe sauvegardé avec succès.")
