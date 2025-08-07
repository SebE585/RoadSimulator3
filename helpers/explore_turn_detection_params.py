import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import circmean

from core.route_generator import simulate_route_from_towns
from core.geo_utils import compute_heading
from core.road_analysis import (
    detect_turns_with_sinuosity,
    compute_acc_y_from_heading,
    validate_turns,
)
from simulator.trajectory import inject_inertial_noise

def make_output_dir():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/explore_turn_thresholds_{now}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

@deprecated
def circular_smooth_heading(heading_series, window=7):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    rad = np.radians(heading_series)
    smoothed = rad.copy()
    half_win = window // 2
    for i in range(len(rad)):
        start = max(0, i - half_win)
        end = min(len(rad), i + half_win + 1)
        smoothed[i] = circmean(rad[start:end], high=2*np.pi, low=0)
    return np.degrees(smoothed)

def simulate_and_prepare():
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

    # Compute heading
    headings = []
    for i in range(len(df) - 1):
        h = compute_heading(df.iloc[i]['lat'], df.iloc[i]['lon'],
                            df.iloc[i + 1]['lat'], df.iloc[i + 1]['lon'])
        headings.append(h)
    headings.append(headings[-1])
    df['heading'] = headings

    df = compute_acc_y_from_heading(df)

    return df

def plot_turns(df, turn_indices, title, save_path=None):
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['delta_heading'], label='Delta Heading')
    plt.plot(df['timestamp'], df['acc_y'].abs(), label='|acc_y|', alpha=0.6)
    # Surbrillance des virages détectés
    turn_times = df.loc[turn_indices, 'timestamp']
    turn_delta_heading = df.loc[turn_indices, 'delta_heading']
    plt.scatter(turn_times, turn_delta_heading, color='red', label='Turns detected', s=10)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def explore_thresholds():
    df = simulate_and_prepare()

    # Calcul delta_heading
    df['delta_heading'] = np.abs(np.ediff1d(df['heading'], to_begin=0))
    df['delta_heading'] = np.where(df['delta_heading'] > 180, 360 - df['delta_heading'], df['delta_heading'])

    out_dir = make_output_dir()

    heading_thresholds = np.arange(5, 30, 5)
    acc_y_thresholds = np.arange(0.1, 1.1, 0.2)

    for h_thresh in heading_thresholds:
        for a_thresh in acc_y_thresholds:
            turn_indices = detect_turns_with_sinuosity(df, heading_threshold=h_thresh, acc_y_threshold=a_thresh)
            valid_turns = validate_turns(df, turn_indices, threshold=0.15)

            print(f"Seuils heading={h_thresh}°, acc_y={a_thresh:.2f} m/s² -> Virages détectés: {len(turn_indices)}, validés: {valid_turns}")

            plot_file = os.path.join(out_dir, f"turns_h{h_thresh}_a{int(a_thresh*100)}.png")
            plot_turns(df, turn_indices, f"Turns detection - heading {h_thresh}°, acc_y {a_thresh:.2f} m/s²", save_path=plot_file)

if __name__ == "__main__":
    explore_thresholds()
