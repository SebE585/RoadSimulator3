import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.events import (
    generate_dos_dane,
    generate_nid_de_poule,
    generate_trottoir,
    generate_acceleration,
    generate_freinage,
    inject_inertial_noise
)
from simulator.detectors import (
    detect_dos_dane,
    detect_nid_de_poule,
    detect_trottoir,
    detect_acceleration,
    detect_freinage
)

HZ = 10

def smooth_accelerations(df, window=3):
    # Lissage simple par moyenne mobile sur acc_x, acc_y, acc_z
    df['acc_x'] = df['acc_x'].rolling(window, center=True, min_periods=1).mean()
    df['acc_y'] = df['acc_y'].rolling(window, center=True, min_periods=1).mean()
    df['acc_z'] = df['acc_z'].rolling(window, center=True, min_periods=1).mean()
    return df

def create_empty_df(n_points=1000):
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq=f'{int(1000/HZ)}ms')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'lat': np.linspace(0, 0.001, n_points),
        'lon': np.linspace(0, 0.001, n_points),
        'speed': np.zeros(n_points),
        'acc_x': np.zeros(n_points),
        'acc_y': np.zeros(n_points),
        'acc_z': np.full(n_points, 9.81),
        'event': [np.nan]*n_points
    })
    return df

# Version adoucie pour la détection d’accélération (test uniquement)
@deprecated
def detect_acceleration_test(acc_x, speed=None, threshold=0.7, min_duration=0.3, hz=10):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    window_size = int(min_duration * hz)
    for i in range(len(acc_x) - window_size):
        window = acc_x[i:i+window_size]
        if all(x > threshold for x in window):
            if speed is not None and i+window_size < len(speed):
                delta_v = speed[i+window_size] - speed[i]
                if delta_v < 3:
                    continue
            return True, i
    return False, -1

def test_event_detection_full_pipeline(event_name, generator_fn, detector_fn):
    df = create_empty_df()
    df = generator_fn(df, max_events=1)
    df = inject_inertial_noise(df)
    df = smooth_accelerations(df)

    if event_name == 'freinage':
        detected_indices = detector_fn(df['acc_x'], df['speed'])  # <-- ici passer deux arguments
        detected = len(detected_indices) > 0
        idx = detected_indices[0] if detected else -1
    elif event_name == 'acceleration':
        detected_indices = detector_fn(df['acc_x'], df['speed'])  # pareil ici
        detected = len(detected_indices) > 0
        idx = detected_indices[0] if detected else -1
    else:
        # Pour les autres, on suppose que le détecteur prend df seul
        result = detector_fn(df)
        if isinstance(result, tuple):
            detected, idx = result
        else:
            detected = result
            idx = -1

    print(f"[DEBUG] {event_name.upper()} détection : détecté={detected}, index={idx}")
    assert detected, f"❌ {event_name} non détecté autour de l'index {idx if idx != -1 else 'non disponible'}."
    print(f"✅ Test complet {event_name} : détection OK autour de l'index {idx if idx != -1 else 'non disponible'}.")


def main():
    test_event_detection_full_pipeline('dos_dane', generate_dos_dane, detect_dos_dane)
    test_event_detection_full_pipeline('nid_de_poule', generate_nid_de_poule, detect_nid_de_poule)
    test_event_detection_full_pipeline('trottoir', generate_trottoir, detect_trottoir)
    test_event_detection_full_pipeline('acceleration', generate_acceleration, detect_acceleration)
    test_event_detection_full_pipeline('freinage', generate_freinage, detect_freinage)

if __name__ == "__main__":
    main()
