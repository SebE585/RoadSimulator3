import os
import sys
# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from core.route_generator import simulate_route_from_towns
from core.road_analysis import detect_turns_with_sinuosity, compute_acc_y_from_heading, validate_turns
from core.geo_utils import compute_heading
from simulator.trajectory import inject_inertial_noise
from simulator.events import *
from simulator.events.pipeline import apply_all_events

def make_output_dir():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/explore_turn_thresholds_{now}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def simulate_route_and_prepare_df():
    interpolated_points, geometry = simulate_route_from_towns(n_points=6)
    n_points = len(interpolated_points)
    timestamps = [datetime(2025, 1, 1, 12, 0, 0) + timedelta(seconds=i*0.1) for i in range(n_points)]
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

    # Remplacement de l'injection d'événements par la pipeline
    df = apply_all_events(df)

    # Compute headings
    headings = []
    for i in range(len(df) - 1):
        h = compute_heading(df.iloc[i]['lat'], df.iloc[i]['lon'], df.iloc[i+1]['lat'], df.iloc[i+1]['lon'])
        headings.append(h)
    headings.append(headings[-1])
    df['heading'] = headings

    # Compute acc_y_calculated from heading
    df = compute_acc_y_from_heading(df)
    
    # Apply light smoothing on acc_y for better detection stability
    df['acc_y'] = df['acc_y'].rolling(window=7, center=True, min_periods=1).mean()

    return df

def explore_thresholds(df, out_dir):
    heading_thresholds = np.arange(5, 30, 5)  # degrees
    acc_y_thresholds = np.arange(0.1, 1.1, 0.2)  # m/s²

    results = []

    for ht in heading_thresholds:
        for at in acc_y_thresholds:
            turn_indices = detect_turns_with_sinuosity(df, heading_threshold=ht, acc_y_threshold=at)
            valid = validate_turns(df, turn_indices, threshold=at)
            results.append({
                'heading_threshold': ht,
                'acc_y_threshold': at,
                'turns_detected': len(turn_indices),
                'turns_validated': sum(valid) if isinstance(valid, (list, np.ndarray)) else int(valid)
            })

    df_res = pd.DataFrame(results)

    # Plot turns_detected heatmap
    pivot_detected = df_res.pivot(index='acc_y_threshold', columns='heading_threshold', values='turns_detected')
    plt.figure(figsize=(8,6))
    plt.title("Nombre de virages détectés")
    plt.xlabel("Seuil de variation de cap (deg)")
    plt.ylabel("Seuil acc_y (m/s²)")
    plt.imshow(pivot_detected, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Virages détectés')
    plt.xticks(ticks=np.arange(len(heading_thresholds)), labels=heading_thresholds)
    plt.yticks(ticks=np.arange(len(acc_y_thresholds)), labels=np.round(acc_y_thresholds,2))
    plt.savefig(os.path.join(out_dir, "heatmap_turns_detected.png"))
    plt.close()

    # Plot turns_validated heatmap
    pivot_validated = df_res.pivot(index='acc_y_threshold', columns='heading_threshold', values='turns_validated')
    plt.figure(figsize=(8,6))
    plt.title("Nombre de virages validés")
    plt.xlabel("Seuil de variation de cap (deg)")
    plt.ylabel("Seuil acc_y (m/s²)")
    plt.imshow(pivot_validated, origin='lower', aspect='auto', cmap='plasma')
    plt.colorbar(label='Virages validés')
    plt.xticks(ticks=np.arange(len(heading_thresholds)), labels=heading_thresholds)
    plt.yticks(ticks=np.arange(len(acc_y_thresholds)), labels=np.round(acc_y_thresholds,2))
    plt.savefig(os.path.join(out_dir, "heatmap_turns_validated.png"))
    plt.close()

    print(f"[INFO] Exploration terminée. Graphiques sauvegardés dans {out_dir}")

if __name__ == "__main__":
    out_dir = make_output_dir()
    df = simulate_route_and_prepare_df()
    explore_thresholds(df, out_dir)
