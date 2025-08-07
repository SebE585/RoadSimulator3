# tests/test_enrich_from_simulated.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from core.osmnx.client import enrich_road_type_stream

CSV_PATH = "out/simulation_20250724_222714/trace.csv"  # ← adapte si besoin

def test_enrich_first_points(n=5):
    print(f"🔍 Chargement du fichier : {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if "lat" not in df.columns or "lon" not in df.columns:
        print("❌ Le fichier ne contient pas les colonnes lat/lon")
        return

    sample = df[["lat", "lon"]].head(n).copy()
    print("📍 Points à tester :")
    print(sample)

    enriched = enrich_road_type_stream(sample)

    print("\n📊 Résultats enrichis :")
    print(enriched[["lat", "lon", "osm_highway", "road_type"]])

if __name__ == "__main__":
    test_enrich_first_points()