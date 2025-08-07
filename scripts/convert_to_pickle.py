import os
import argparse
import pickle
import networkx as nx

def convert_to_pickle(input_path, output_path=None, compress=False):
    print(f"📥 Chargement du fichier GraphML : {input_path}")

    # Chargement du graphe avec identifiants de nœuds en chaîne
    G = nx.read_graphml(input_path, node_type=str)

    # Ajout manuel du CRS s’il est absent
    if "crs" not in G.graph:
        print("⚠️ CRS manquant dans G.graph → Ajout manuel EPSG:4326")
        G.graph["crs"] = "epsg:4326"

    # Chemin de sortie
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".graph.pkl"

    print(f"💾 Sauvegarde au format Pickle : {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(G, f)

    print("✅ Conversion terminée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True, help="Chemin vers le fichier .graphml")
    parser.add_argument("--out", dest="output_path", help="Chemin de sortie .pkl (facultatif)")
    parser.add_argument("--compress", action="store_true", help="Compression (non utilisé ici)")
    args = parser.parse_args()

    convert_to_pickle(args.input_path, args.output_path, args.compress)