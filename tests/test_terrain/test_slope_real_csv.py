import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os

# Ajout du chemin projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from core.terrain.slope import compute_slope_from_altitude

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.terrain.slope import compute_slope_from_altitude
from core.terrain.distance import compute_cumulative_distance
from scipy.ndimage import gaussian_filter1d

def test_slope_on_real_csv():
    # 🔁 Remplace par le chemin de ton fichier réel si différent
    csv_path = "data/trace_exemple.csv"
    assert os.path.exists(csv_path), f"Fichier introuvable : {csv_path}"

    df = pd.read_csv(csv_path)

    if 'distance_m' not in df.columns:
        df = compute_cumulative_distance(df)

    # ✅ Lissage de l'altitude
    df['altitude_smoothed'] = gaussian_filter1d(df['altitude'], sigma=3)

    # ✅ Calcul pente depuis altitude lissée
    slope = compute_slope_from_altitude(df, altitude_col='altitude_smoothed')
    slope_clipped = pd.Series(slope).clip(-30, 30)
    df['slope_percent'] = slope_clipped

    # 📊 Graphe comparatif
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(df['distance_m'], df['altitude'], label="Altitude brute", color='tab:gray')
    axs[0].plot(df['distance_m'], df['altitude_smoothed'], label="Altitude lissée", color='tab:blue')
    axs[0].legend()
    axs[0].set_ylabel("Altitude (m)")

    axs[1].plot(df['distance_m'], slope, label="Pente brute", color='tab:red', alpha=0.5)
    axs[1].set_ylabel("Pente (%)")

    axs[2].plot(df['distance_m'], slope_clipped, label="Pente lissée (clip ±30%)", color='tab:green')
    axs[2].set_ylabel("Pente (%)")
    axs[2].set_xlabel("Distance (m)")
    axs[2].legend()

    plt.suptitle("Comparaison pente brute vs lissée (±30%)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# Appel explicite quand lancé directement
if __name__ == "__main__":
    test_slope_on_real_csv()