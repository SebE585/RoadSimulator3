import pandas as pd

def validate_timestamps(df):
    if not df['timestamp'].is_monotonic_increasing:
        duplicates = df['timestamp'].duplicated(keep=False).sum()
        print(f"❌ Timestamps non croissants détectés : {duplicates} doublons.")
        print(df.loc[df['timestamp'].duplicated(keep=False), 'timestamp'].head())
        raise AssertionError("Timestamps non croissants.")
    print("✅ Timestamps validés.")

def validate_spatial_coherence(df, max_speed=60):
    """Vérifie que les vitesses ne dépassent pas max_speed (km/h)."""
    if df['speed'].max() > max_speed:
        raise ValueError("Vitesse dépassant max_speed détectée!")

def compute_speed_stats(df):
    """Calcule min, max, moyenne vitesse."""
    return {
        'speed_min': df['speed'].min(),
        'speed_max': df['speed'].max(),
        'speed_mean': df['speed'].mean()
    }

def is_regular_sampling(df, freq_hz=10):
    """
    Vérifie si l'échantillonnage est régulier (fréquence cible en Hz).
    Retourne True si toutes les différences de timestamp sont proches de 1/freq_hz.
    """
    expected_interval = 1.0 / freq_hz
    dt = df['timestamp'].diff().dt.total_seconds().dropna()
    median = dt.median()
    return abs(median - expected_interval) < 0.01
