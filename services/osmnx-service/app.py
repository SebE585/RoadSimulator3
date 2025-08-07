import os
import json
import uuid
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, Response
import networkx as nx
import osmnx as ox
from scipy.spatial import cKDTree

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 📦 Chargement du graphe OSM ===
GRAPH_PATH_GML = "/data/graph_haute_normandie.graphml"
GRAPH_PATH_PKL = "/data/graph_haute_normandie.graph.pkl"

if os.path.exists(GRAPH_PATH_PKL):
    logger.info(f"📦 Chargement rapide depuis Pickle : {GRAPH_PATH_PKL}")
    with open(GRAPH_PATH_PKL, "rb") as f:
        G = pickle.load(f)
else:
    logger.info(f"📦 Chargement depuis GraphML : {GRAPH_PATH_GML}")
    G = ox.load_graphml(GRAPH_PATH_GML)
    # Sauvegarde immédiate au format Pickle
    try:
        with open(GRAPH_PATH_PKL, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"💾 Graphe sauvegardé au format Pickle : {GRAPH_PATH_PKL}")
    except Exception as e:
        logger.warning(f"⚠️ Échec de la sauvegarde Pickle : {e}")

# === 📌 Conversion en GeoDataFrame ===
nodes, edges = ox.graph_to_gdfs(G)

# 🔄 Reprojection en EPSG:4326
if nodes.crs.to_epsg() != 4326:
    nodes_proj = nodes.to_crs(epsg=4326)
    logger.info("🌐 Graphe reprojeté en EPSG:4326")
else:
    nodes_proj = nodes
    logger.info("✅ Graphe déjà en CRS WGS84 (EPSG:4326)")

# 🌳 Initialisation BallTree pour recherche rapide
balltree = cKDTree(np.vstack([nodes_proj['y'], nodes_proj['x']]).T)
logger.info(f"🗺️ BallTree construit avec {len(nodes_proj)} nœuds.")

# === Sessions actives ===
active_streams = {}

# 🔍 Récupération du tag highway
def get_osm_highway_from_node(G, node_id):
    edges = list(G.out_edges(node_id, data=True)) or list(G.in_edges(node_id, data=True))
    for _, _, data in edges:
        if "highway" in data:
            return data["highway"]
    return "unknown"

@app.route('/nearest_road_batch_stream/start', methods=['POST'])
@deprecated
def start_stream():
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    data = request.json
    if not data or 'lat' not in data or 'lon' not in data:
        return {"error": "Missing 'lat' or 'lon'"}, 400

    lats = data['lat']
    lons = data['lon']
    stream_id = str(uuid.uuid4())
    active_streams[stream_id] = list(zip(lats, lons))
    logger.info(f"[STREAM] Initialisation stream {stream_id} avec {len(lats)} points.")
    return {"stream_id": stream_id}, 200

@app.route('/nearest_road_batch_stream/stream/<stream_id>')
@deprecated
def stream_results(stream_id):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    if stream_id not in active_streams:
        return {"error": "Invalid stream_id"}, 404

    coords = active_streams.pop(stream_id)

    def generate():
        for i, (lat, lon) in enumerate(coords):
            try:
                dist, idx = balltree.query([[lat, lon]], k=1)
                nearest_node = nodes_proj.iloc[idx[0]]
                node_id = nearest_node.name
                osm_highway = get_osm_highway_from_node(G, node_id)

                logger.info(f"[QUERY] #{i} lat={lat:.5f} lon={lon:.5f} → node={node_id}, highway={osm_highway}, dist={dist[0]:.2f}m")

                yield f"data: {json.dumps({ 'index': i, 'osm_highway': osm_highway, 'nearest_distance_m': float(dist[0]) })}\n\n"

            except Exception as e:
                logger.warning(f"[ERROR] stream index={i} lat={lat} lon={lon} → {e}")
                yield f"data: {json.dumps({ 'index': i, 'osm_highway': 'error', 'nearest_distance_m': None })}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)