import pytest
import numpy as np
import pandas as pd

from simulator import events, detectors
from core.osrm_utils import validate_turns

HZ = 10

def make_base_df(size=2200):
    return pd.DataFrame({
        'speed': np.linspace(10, 50, size),
        'acc_x': np.zeros(size, dtype=float),
        'acc_y': np.zeros(size, dtype=float),
        'acc_z': np.full(size, events.G, dtype=float),
        'event': pd.Series([np.nan] * size, dtype=object)
    })

def test_generate_and_detect_acceleration():
    base = make_base_df(100)
    base['speed'] = 30.0
    df = events.generate_acceleration(base.copy(), max_events=1, amplitude=5.0, duration_s=2.0)

    # chercher les indices où l'accélération a été injectée
    injected_indices = df.index[df['event'] == 'acceleration'].tolist()
    assert injected_indices, "Aucune accélération injectée"

    delta_v = df['speed'].iloc[injected_indices[-1]] - df['speed'].iloc[injected_indices[0]]
    print(f"[DEBUG] Gain de vitesse : {delta_v:.2f} km/h")

    indices = detectors.detect_acceleration(df['acc_x'], df['speed'])
    print(f"[DEBUG] Indices détectés pour accélération : {indices}")

    assert delta_v >= 1.5
    assert len(indices) >= 1

def test_generate_and_detect_freinage():
    base = make_base_df(100)
    base['speed'] = 60.0
    df = events.generate_freinage(base.copy(), max_events=1, amplitude=-6.0, duration_s=2.0)

    injected_indices = df.index[df['event'] == 'freinage'].tolist()
    assert injected_indices, "Aucun freinage injecté"

    delta_v = df['speed'].iloc[injected_indices[0]] - df['speed'].iloc[injected_indices[-1]]
    print(f"[DEBUG] Perte de vitesse : {delta_v:.2f} km/h")

    indices = detectors.detect_freinage(df['acc_x'], df['speed'])
    print(f"[DEBUG] Indices détectés pour freinage : {indices}")

    assert delta_v >= 4.0
    assert len(indices) >= 1

def test_detect_stop_and_wait():
    df = make_base_df()
    stop_idx = 500
    stop_duration = HZ * 120  # 2 min
    df.loc[stop_idx:stop_idx+stop_duration, 'speed'] = 0
    df.loc[stop_idx:stop_idx+stop_duration, ['acc_x', 'acc_y']] = 0
    df.loc[stop_idx:stop_idx+stop_duration, 'acc_z'] = events.G
    df.loc[stop_idx:stop_idx+stop_duration, 'event'] = 'stop'

    wait_idx = 1500
    wait_duration = HZ * 60  # 1 min
    df.loc[wait_idx:wait_idx+wait_duration, 'speed'] = 0
    df.loc[wait_idx:wait_idx+wait_duration, 'event'] = 'wait'
    df.loc[wait_idx:wait_idx+wait_duration, 'acc_x'] = np.random.normal(0, 0.05, wait_duration)
    df.loc[wait_idx:wait_idx+wait_duration, 'acc_y'] = np.random.normal(0, 0.05, wait_duration)
    df.loc[wait_idx:wait_idx+wait_duration, 'acc_z'] = events.G + np.random.normal(0, 0.1, wait_duration)

    detected = detectors.detect_stop_and_wait_from_signals(df)

    # vérifie que la majorité des points sont bien détectés
    stop_detected = (detected[stop_idx:stop_idx+stop_duration] == 'stop').sum() / stop_duration
    wait_detected = (detected[wait_idx:wait_idx+wait_duration] == 'wait').sum() / wait_duration

    print(f"[DEBUG] % stop détecté : {stop_detected*100:.1f}%")
    print(f"[DEBUG] % wait détecté : {wait_detected*100:.1f}%")

    assert stop_detected > 0.9, "Moins de 90% des stops détectés"
    assert wait_detected > 0.9, "Moins de 90% des waits détectés"

