# tests/test_osmnx/test_osmnx.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import pandas as pd
from core.osmnx.client import enrich_road_type_stream
from core.osmnx.mapping import get_edge_type_nearest

@pytest.mark.osmnx
def test_osmnx_enrichment_chain():
    """
    Teste la chaîne complète d’enrichissement via OSMnx :
    - Appel au service SSE distant,
    - Ajout de la colonne 'osm_highway',
    - Mapping vers 'road_type' avec get_edge_type_nearest,
    - Vérifie que ≥80% des road_type sont connus.
    """

    df = pd.DataFrame({
        "lat": [49.4426, 49.4421, 49.4419],
        "lon": [1.0931, 1.0938, 1.0945],
    })

    # Enrichissement via l'API locale
    df = enrich_road_type_stream(df)

    # Vérifications de base
    assert "osm_highway" in df.columns, "Colonne 'osm_highway' absente après enrichissement"
    assert df["osm_highway"].notna().any(), "Toutes les valeurs 'osm_highway' sont NaN"

    print("🚦 Tags OSM uniques avant mapping :", df["osm_highway"].unique())

    # Application du mapping harmonisé
    df["road_type"] = df["osm_highway"].apply(lambda h: get_edge_type_nearest({"highway": h}))
    df["road_type"] = df["road_type"].fillna("unknown")

    unmapped = df[df["road_type"] == "unknown"]["osm_highway"].value_counts()
    if not unmapped.empty:
        print("[⚠️ WARNING] Tags OSM non mappés dans HIGHWAY_TO_TYPE :")
        print(unmapped)

    print("🛣️ Résumé des road_type :")
    print(df["road_type"].value_counts())

    # Vérification du ratio de road_type connus
    known = (df["road_type"] != "unknown").sum()
    total = len(df)
    ratio = known / total
    print(f"[✅ TEST] Coverage road_type ≠ unknown: {known}/{total} → {ratio:.2%}")

    assert ratio >= 0.80, f"Moins de 80% des road_type sont connus ({ratio:.2%})"