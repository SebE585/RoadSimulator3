import pandas as pd
import numpy as np

def recompute_inertial_acceleration(df, hz=10):
    """
    Recalcule acc_x (accélération longitudinale) et acc_y (latérale) à partir de la vitesse et du heading.

    Args:
        df (pd.DataFrame): Doit contenir les colonnes 'speed' (en km/h) et 'heading' (en degrés).
        hz (int): Fréquence d’échantillonnage en Hz (par défaut 10 Hz).

    Returns:
        pd.DataFrame: Même DataFrame avec acc_x, acc_y recalculées.
    """
    df = df.copy()
    dt = 1.0 / hz

    if "speed" not in df.columns or "heading" not in df.columns:
        raise ValueError("Les colonnes 'speed' et 'heading' doivent être présentes dans le DataFrame.")

    v_ms = df["speed"].fillna(0).clip(lower=0) / 3.6  # Convertir en m/s

    # Accélération longitudinale (dérivée temporelle de la vitesse)
    acc_x = np.gradient(v_ms, dt)

    # Accélération latérale via la dérivée du heading (approx. gyroscope)
    heading_filled = df["heading"].ffill().bfill()
    heading_rad = np.radians(heading_filled)
    d_heading = np.gradient(heading_rad, dt)
    acc_y = v_ms * d_heading  # formule simplifiée pour une vitesse angulaire

    df["acc_x"] = acc_x
    df["acc_y"] = acc_y

    return df


# -----------------------------------------------------------
# Utility: Affiche des statistiques sur acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
def check_inertial_stats(df, label="Inertial Stats"):
    """
    Affiche des statistiques sommaires sur les colonnes d'accélération et gyro.

    Args:
        df (pd.DataFrame): DataFrame contenant acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
        label (str): Label pour le print.
    """
    cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    print(f"📊 {label}:")
    for col in cols:
        if col in df.columns:
            series = df[col].dropna()
            print(f" - {col}: mean={series.mean():.3f}, std={series.std():.3f}, min={series.min():.3f}, max={series.max():.3f}")
        else:
            print(f" - {col}: colonne absente")
