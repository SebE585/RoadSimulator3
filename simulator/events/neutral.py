import numpy as np
import pandas as pd

def inject_neutral_phases(df, duration_s=2.0, speed_kmh=30.0, hz=10):
    """
    Injecte des phases de roulage neutre (vitesse constante, bruit inertiel) entre les événements.
    Ces points ont event == NaN et simulent un roulage calme sans changement brutal.
    """
    if 'event' not in df.columns:
        df['event'] = np.nan

    n_points = int(duration_s * hz)
    speed_mps = speed_kmh / 3.6

    # Repérer les index où ajouter les phases neutres (entre événements)
    event_indices = df.index[df['event'].notna() & ~df['event'].isin(['stop', 'wait'])].tolist()
    insertion_indices = []

    for i in range(1, len(event_indices)):
        prev_end = event_indices[i - 1]
        next_start = event_indices[i]
        if next_start - prev_end > n_points:
            insertion_indices.append(prev_end + 1)

    # Créer les phases et les insérer
    neutral_rows = []
    for idx in insertion_indices:
        lat = df.at[idx, 'lat']
        lon = df.at[idx, 'lon']
        timestamp = df.at[idx, 'timestamp']
        for j in range(n_points):
            neutral_rows.append({
                'timestamp': pd.to_datetime(timestamp) + pd.Timedelta(seconds=(j+1)/hz),
                'lat': lat,
                'lon': lon,
                'speed': speed_kmh,
                'acc_x': np.random.normal(0, 0.3),
                'acc_y': np.random.normal(0, 0.35),
                'acc_z': np.random.normal(9.8, 0.3),
                'gyro_x': np.random.normal(0, 0.3),
                'gyro_y': np.random.normal(0, 0.3),
                'gyro_z': np.random.normal(0, 0.3),
                'event': np.nan
            })

    # 🔁 Fallback si aucun point inséré : ajouter une phase neutre à la fin
    if len(neutral_rows) == 0 and len(df) > n_points:
        idx = len(df) - n_points - 1
        lat = df.at[idx, 'lat']
        lon = df.at[idx, 'lon']
        timestamp = df.at[idx, 'timestamp']
        for j in range(n_points):
            neutral_rows.append({
                'timestamp': pd.to_datetime(timestamp) + pd.Timedelta(seconds=(j+1)/hz),
                'lat': lat,
                'lon': lon,
                'speed': speed_kmh,
                'acc_x': np.random.normal(0, 0.3),
                'acc_y': np.random.normal(0, 0.35),
                'acc_z': np.random.normal(9.8, 0.3),
                'gyro_x': np.random.normal(0, 0.3),
                'gyro_y': np.random.normal(0, 0.3),
                'gyro_z': np.random.normal(0, 0.3),
                'event': np.nan
            })

    df_neutral = pd.DataFrame(neutral_rows)
    # Supprimer les lignes ayant des lat/lon non valides (NaN ou inf)
    df_neutral = df_neutral.replace([np.inf, -np.inf], np.nan)
    df_neutral = df_neutral.dropna(subset=['lat', 'lon'])
    df = pd.concat([df, df_neutral], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df
