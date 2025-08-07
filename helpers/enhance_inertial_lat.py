import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from datetime import datetime, timedelta
from simulator.events import *
try:
    from simulator.events.pipeline import apply_all_events
except ImportError:
    pass
from simulator.trajectory import inject_inertial_noise
from core.route_generator import simulate_route_from_towns
from core.sinuosity import apply_sinuosity_to_df
from core.road_analysis import (
    detect_turns_with_sinuosity,
    detect_roundabouts_with_sinuosity,
    compute_acc_y_from_heading
)
from core.geo_utils import compute_heading
from helpers.roundabout_utils import compare_roundabout_detections

import matplotlib.pyplot as plt


def make_output_dir():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/simulated_{now}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def simulate_dummy_route():
    interpolated_points, geometry = simulate_route_from_towns(n_points=6)
    n_points = len(interpolated_points)
    timestamps = pd.date_range(start="2025-01-01 12:00:00", periods=n_points, freq='100L')  # 0.1s interval

    df = pd.DataFrame(interpolated_points, columns=["lat", "lon"])
    df["timestamp"] = timestamps
    df["speed"] = 40.0
    df["acc_x"] = 0.0
    df["acc_y"] = 0.0
    df["acc_z"] = 9.81
    df["event"] = np.nan

    df = inject_initial_acceleration(df)
    df = inject_final_deceleration(df)
    df = inject_inertial_noise(df)

    indexes = np.linspace(100, len(df) - 500, 5, dtype=int)
    for idx in indexes:
        df.at[idx, 'speed'] = 30.0
        df = apply_all_events(df)

    df = apply_sinuosity_to_df(df, geometry_coords=geometry["coordinates"])
    return df


def enhance_and_compare():
    print("Villes sélectionnées :")
    df = simulate_dummy_route()
    out_dir = make_output_dir()

    # Sauvegarde CSV
    csv_path = os.path.join(out_dir, "trace.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV exporté avec succès : {csv_path}")

    # Calcul heading si pas présent
    if 'heading' not in df.columns:
        headings = []
        for i in range(len(df) - 1):
            h = compute_heading(df.iloc[i]['lat'], df.iloc[i]['lon'],
                                df.iloc[i+1]['lat'], df.iloc[i+1]['lon'])
            headings.append(h)
        headings.append(headings[-1])
        df['heading'] = headings
    print("[INFO] Calcul des headings effectué.")

    # Calcul acc_y_calculated
    df = compute_acc_y_from_heading(df)
    print("[INFO] Calcul de acc_y_calculated effectué.")

    # Debug nombre de "rond-point" dans sinuosity
    if 'sinuosity' in df.columns:
        count_rond_points = (df['sinuosity'] == 'rond-point').sum()
        print(f"[DEBUG] Nombre de lignes avec 'rond-point' dans 'sinuosity' : {count_rond_points}")
    else:
        print("[WARN] Colonne 'sinuosity' absente dans le DataFrame.")

    # Détection virages et rond-points AVANT lissage
    turn_indices_raw = detect_turns_with_sinuosity(df, heading_threshold=10, acc_y_threshold=0.3)
    roundabouts_raw = detect_roundabouts_with_sinuosity(df, acc_y_threshold=0.5)

    print(f"[RESULTATS] Virages détectés AVANT lissage : {len(turn_indices_raw)}")
    print(f"[RESULTATS] Ronds-points détectés AVANT lissage : {len(roundabouts_raw)}")

    # Lissage léger sur acc_y (moyenne mobile)
    df_liss = df.copy()
    df_liss['acc_y'] = df_liss['acc_y'].rolling(window=5, center=True, min_periods=1).mean()

    # Détection virages et rond-points APRÈS lissage
    turn_indices_smooth = detect_turns_with_sinuosity(df_liss, heading_threshold=10, acc_y_threshold=0.3)
    roundabouts_smooth = detect_roundabouts_with_sinuosity(df_liss, acc_y_threshold=0.5)

    print(f"[RESULTATS] Virages détectés APRES lissage : {len(turn_indices_smooth)}")
    print(f"[RESULTATS] Ronds-points détectés APRES lissage : {len(roundabouts_smooth)}")

    # Graphique acc_y réel vs lissé
    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp'], df['acc_y'], label='acc_y réel', alpha=0.6)
    plt.plot(df_liss['timestamp'], df_liss['acc_y'], label='acc_y lissé', linestyle='--')
    plt.title("Comparaison acc_y réel vs lissé")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_y_real_vs_liss.png"))
    plt.close()
    print("[INFO] Graphique acc_y réel vs lissé sauvegardé.")

    # Histogramme delta_heading aux indices de virage détectés
    df['delta_heading'] = np.abs(np.ediff1d(df['heading'], to_begin=0))
    df['delta_heading'] = np.where(df['delta_heading'] > 180, 360 - df['delta_heading'], df['delta_heading'])

    plt.figure(figsize=(12, 4))
    plt.hist(df.loc[turn_indices_raw, 'delta_heading'], bins=30, alpha=0.7, label='Raw turns')
    plt.hist(df.loc[turn_indices_smooth, 'delta_heading'], bins=30, alpha=0.7, label='Smooth turns')
    plt.title("Histogramme delta_heading aux indices de virage détectés")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_delta_heading_turns.png"))
    plt.close()
    print("[INFO] Histogramme delta_heading aux virages sauvegardé.")

    # Histogramme acc_y absolue aux indices de virage détectés
    plt.figure(figsize=(12, 4))
    plt.hist(df.loc[turn_indices_raw, 'acc_y'].abs(), bins=30, alpha=0.7, label='Raw turns acc_y')
    plt.hist(df_liss.loc[turn_indices_smooth, 'acc_y'].abs(), bins=30, alpha=0.7, label='Smooth turns acc_y')
    plt.title("Histogramme acc_y absolue aux indices de virage détectés")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_acc_y_turns.png"))
    plt.close()
    print("[INFO] Histogramme acc_y aux virages sauvegardé.")


if __name__ == "__main__":
    enhance_and_compare()
