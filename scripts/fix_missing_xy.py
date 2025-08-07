import os
import pickle
import networkx as nx

INPUT_GRAPHML = "data/osmnx/graph_haute_normandie_fixed.graphml"
OUTPUT_PICKLE = "data/osmnx/graph_haute_normandie_fixed.graph.pkl"

print(f"📥 Chargement du fichier GraphML : {INPUT_GRAPHML}")
G = nx.read_graphml(INPUT_GRAPHML, node_type=str)

# ✅ Ajout CRS si manquant
if "crs" not in G.graph:
    print("⚠️ CRS manquant → ajout manuel EPSG:4326")
    G.graph["crs"] = "EPSG:4326"

# ✅ Réparation des x/y manquants
for node, data in G.nodes(data=True):
    if "x" not in data or "y" not in data:
        if "lon" in data and "lat" in data:
            data["x"] = float(data["lon"])
            data["y"] = float(data["lat"])
        else:
            raise ValueError(f"Nœud {node} sans coordonnées valides : {data}")

print(f"💾 Sauvegarde au format Pickle : {OUTPUT_PICKLE}")
with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(G, f)

print("✅ Graphe corrigé et sauvegardé.")