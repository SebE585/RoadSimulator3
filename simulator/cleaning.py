import numpy as np
import pandas as pd

def clean_simulation_errors(df,
                            max_speed_kmh=160,
                            max_acc_m_s2=15,
                            max_speed_delta_kmh=30,
                            max_event_nan_ratio=0.3):
    """
    Nettoie et corrige les incohérences dans le DataFrame simulé.

    Args:
        df (pd.DataFrame): DataFrame simulé contenant au minimum les colonnes :
                           ['speed', 'acc_x', 'acc_y', 'acc_z', 'event', 'timestamp']
        max_speed_kmh (float): Seuil max de vitesse réaliste.
        max_acc_m_s2 (float): Seuil max d'accélération (en m/s²) toléré.
        max_speed_delta_kmh (float): Variation max de vitesse entre points consécutifs.
        max_event_nan_ratio (float): Seuil de tolérance pour NaN dans `event`.

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """

    df = df.copy()

    # --- 1. Clamp vitesse sur plage physique réaliste ---
    if 'speed' in df.columns:
        df.loc[df['speed'] < 0, 'speed'] = 0
        df.loc[df['speed'] > max_speed_kmh, 'speed'] = max_speed_kmh

    # --- 2. Clamp accélérations sur plage réaliste ---
    for axis in ['acc_x', 'acc_y', 'acc_z']:
        if axis in df.columns:
            df.loc[df[axis] < -max_acc_m_s2, axis] = -max_acc_m_s2
            df.loc[df[axis] > max_acc_m_s2, axis] = max_acc_m_s2

    # --- 3. Filtrer variations de vitesse trop brutales ---
    if 'speed' in df.columns:
        speed = df['speed'].values
        delta = np.diff(speed, prepend=speed[0])
        for i in range(1, len(speed)):
            if abs(delta[i]) > max_speed_delta_kmh:
                # Clamp delta à max_speed_delta_kmh
                speed[i] = speed[i-1] + np.sign(delta[i]) * max_speed_delta_kmh
        df['speed'] = speed

    # --- 4. Gestion NaN dans 'event' ---
    if 'event' in df.columns:
        nan_ratio = df['event'].isna().mean()
        if nan_ratio > max_event_nan_ratio:
            print(f"[WARN] Trop de NaN dans 'event' ({nan_ratio:.2%}), remplissage par 'unknown'")
            df['event'] = df['event'].fillna('unknown')
        else:
            # Optionnel : remplacer NaN par '' ou 'none'
            df['event'] = df['event'].fillna('')

    # --- 5. Supprimer lignes où timestamp est NaN ou non croissant ---
    if 'timestamp' in df.columns:
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # --- 6. Corriger NaN sur coordonnées GPS ---
    for coord in ['lat', 'lon']:
        if coord in df.columns:
            # Remplacement par interpolation linéaire
            df[coord] = df[coord].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # --- 7. Vérifier NaN résiduels, les supprimer si nécessaires ---
    if df.isna().sum().sum() > 0:
        print("[WARN] Suppression des lignes avec NaN résiduels")
        df = df.dropna()

    return df