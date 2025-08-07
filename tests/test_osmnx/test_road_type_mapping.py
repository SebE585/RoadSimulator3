import pandas as pd

def test_road_type_mapping_coverage():
    # 🔁 Remplace ceci par le chemin vers ton CSV simulé/enrichi
    df = pd.read_csv("data/trace_enrichie.csv")

    assert 'road_type' in df.columns, "Colonne 'road_type' absente"
    
    total = len(df)
    known = (df['road_type'] != 'unknown').sum()
    ratio = known / total

    print(f"[TEST] {known} sur {total} points ont un road_type défini ({ratio:.2%})")
    
    assert ratio >= 0.80, f"Seulement {ratio:.2%} des points ont un 'road_type' défini"