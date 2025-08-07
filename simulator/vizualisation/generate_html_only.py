import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from simulator.vizualisation.map_renderer import generate_html_map

def main():
    csv_path = "output_simulated_trajectory.csv"
    html_path = "logs/test_map.html"

    if not os.path.exists(csv_path):
        print(f"[ERROR] Fichier CSV non trouvé : {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "lat" not in df.columns or "lon" not in df.columns:
        print("[ERROR] Le DataFrame ne contient pas les colonnes 'lat' et 'lon'")
        return

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    generate_html_map(df, html_path)
    print(f"[INFO] Carte HTML générée avec succès : {html_path}")

if __name__ == "__main__":
    main()