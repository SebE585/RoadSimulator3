from pyrosm import OSM

PBF_FILE = "data/osmnx/haute-normandie-latest.osm.pbf"
NETWORK_TYPE = "driving"

print(f"[INFO] Chargement du PBF local : {PBF_FILE}")
osm = OSM(PBF_FILE)

print(f"[INFO] Extraction des edges et nodes pour : {NETWORK_TYPE}")
edges, nodes = osm.get_network(NETWORK_TYPE, nodes=True)

print(f"[DEBUG] nodes.columns: {nodes.columns}")
print(f"[DEBUG] nodes.head():\n{nodes.head()}")
print(f"[DEBUG] nodes.geometry types:\n{nodes.geometry.apply(lambda g: g.geom_type if g else None).value_counts()}")

if nodes.empty:
    print("[ERREUR] Le GeoDataFrame des nodes est vide après extraction.")
else:
    print(f"[INFO] Nombre total de nodes : {len(nodes)}")
