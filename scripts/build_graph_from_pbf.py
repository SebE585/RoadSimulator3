import osmnx as ox
import networkx as nx

OSM_PATH = "data/osmnx/haute-normandie.osm"
OUT_PATH = "data/osmnx/graph_haute_normandieA.graphml"

print(f"📥 Chargement depuis le fichier OSM : {OSM_PATH}")
G = ox.graph_from_xml(OSM_PATH)

# 🔧 Suppression manuelle des nœuds isolés
print("🧼 Suppression des nœuds isolés...")
largest_cc = max(nx.weakly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

print(f"💾 Sauvegarde finale dans : {OUT_PATH}")
ox.save_graphml(G, OUT_PATH)

print("✅ Graphe exporté avec succès.")