import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from datetime import datetime, timedelta
from core.route_generator import simulate_route_from_towns
from simulator.trajectory import inject_inertial_noise
from core.sinuosity import apply_sinuosity_to_df
from core.road_analysis import detect_turns_with_sinuosity, compute_acc_y_from_heading, validate_turns
from core.geo_utils import compute_heading

def make_output_dir():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/diagnose_turn_validation_{now}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def simulate_and_prepare_df():
    interpolated_points, geometry = simulate_route_from_towns(n_points=6)
    n_points = len(interpolated_points)
    timestamps = pd.date_range(start="2025-01-01 12:00:00", periods=n_points, freq='100ms')

    df = pd.DataFrame(interpolated_points, columns=["lat", "lon"])
    df["timestamp"] = timestamps
    df["speed"] = 40.0
    df["acc_x"] = 0.0
    df["acc_y"] = 0.0
    df["acc_z"] = 9.81
    df["event"] = np.nan

    df = inject_inertial_noise(df)
    df = apply_sinuosity_to_df(df, geometry_coords=geometry["coordinates"])

    # Calcul heading
    headings = []
    for i in range(len(df) - 1):
        h = compute_heading(df.iloc[i]['lat'], df.iloc[i]['lon'], df.iloc[i+1]['lat'], df.iloc[i+1]['lon'])
        headings.append(h)
    headings.append(headings[-1])
    df['heading'] = headings

    df = compute_acc_y_from_heading(df)
    return df

def diagnose_threshold(df, heading_threshold=10, window=5, acc_y_thresholds=None):
    if acc_y_thresholds is None:
        acc_y_thresholds = np.arange(0.05, 0.55, 0.05)

    turn_indices = detect_turns_with_sinuosity(df, window_size=window, heading_threshold=heading_threshold, acc_y_threshold=0)
    print(f"Total virages détectés (sans seuil acc_y) : {len(turn_indices)}")

    for acc_thresh in acc_y_thresholds:
        valid_turns = []
        print(f"\nSeuil acc_y pour validation : {acc_thresh:.3f}")
        for idx in turn_indices:
            window_start = max(0, idx - window)
            window_end = min(len(df), idx + window + 1)
            acc_y_mean = df.loc[window_start:window_end, 'acc_y'].abs().mean()
            if acc_y_mean >= acc_thresh:
                valid_turns.append(idx)
            if acc_y_mean >= acc_thresh and idx % 5000 == 0:  # Exemples critiques par index tous les 5000
                print(f" - Virage index {idx} : acc_y moyenne = {acc_y_mean:.3f}")
        print(f"Nombre virages validés : {len(valid_turns)} / {len(turn_indices)}")

if __name__ == "__main__":
    print("Simulation et préparation des données...")
    df = simulate_and_prepare_df()
    diagnose_threshold(df)

import matplotlib.pyplot as plt

def diagnose_threshold_with_plot(df, heading_threshold=10, window=5, acc_y_thresholds=None):
    if acc_y_thresholds is None:
        acc_y_thresholds = np.arange(0.05, 0.55, 0.05)

    turn_indices = detect_turns_with_sinuosity(df, window_size=window, heading_threshold=heading_threshold, acc_y_threshold=0)
    print(f"Total virages détectés (sans seuil acc_y) : {len(turn_indices)}")

    validated_counts = []
    for acc_thresh in acc_y_thresholds:
        valid_turns = []
        for idx in turn_indices:
            window_start = max(0, idx - window)
            window_end = min(len(df), idx + window + 1)
            acc_y_mean = df.loc[window_start:window_end, 'acc_y'].abs().mean()
            if acc_y_mean >= acc_thresh:
                valid_turns.append(idx)
        validated_counts.append(len(valid_turns))

    # Tracé
    plt.figure(figsize=(8,5))
    plt.plot(acc_y_thresholds, validated_counts, marker='o')
    plt.title("Nombre de virages validés en fonction du seuil acc_y")
    plt.xlabel("Seuil acc_y (m/s²)")
    plt.ylabel("Nombre virages validés")
    plt.grid(True)
    plt.tight_layout()
    out_dir = make_output_dir()
    plot_path = os.path.join(out_dir, "validated_turns_vs_acc_y_threshold.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Graphique sauvegardé : {plot_path}")

if __name__ == "__main__":
    print("Simulation et préparation des données...")
    df = simulate_and_prepare_df()
    diagnose_threshold_with_plot(df)

