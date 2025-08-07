import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Ajout du chemin projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.terrain.slope import compute_slope_from_altitude

def test_compute_slope_with_plot():
    # Données de test : montée constante
    df = pd.DataFrame({
        'altitude': [100, 110, 120, 130, 140],
        'distance_m': [0, 100, 200, 300, 400]
    })

    slope = compute_slope_from_altitude(df)

    # Vérification logique
    expected = np.array([np.nan, 10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(slope[1:], expected[1:], atol=1e-6)
    assert np.isnan(slope[0])

    # Graphe matplotlib
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Altitude (m)", color='tab:blue')
    ax1.plot(df['distance_m'], df['altitude'], label="Altitude", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Pente (%)", color='tab:red')
    ax2.plot(df['distance_m'], slope, label="Pente", color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("✅ Test : pente calculée depuis l’altitude")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

# Appel explicite quand lancé directement
if __name__ == "__main__":
    test_compute_slope_with_plot()