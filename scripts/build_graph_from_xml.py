import osmnx as ox
import networkx as nx

OSM_PATH = "data/osmnx/roads-car.osm"
OUT_PATH = "data/osmnx/graph_haute_normandieD.graphml"

print(f"📥 Chargement XML filtré depuis {OSM_PATH}")
G = ox.graph_from_xml(OSM_PATH)

# === PATCH: suppression manuelle des composants isolés ===
print("🧼 Suppression des nœuds isolés...")
largest_cc = max(nx.weakly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

print(f"💾 Sauvegarde finale dans : {OUT_PATH}")
ox.save_graphml(G, OUT_PATH)

print("✅ Graphe exporté avec succès.")