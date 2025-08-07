# simulator/events/gyro.py
import numpy as np
import pandas as pd
from core.gps_utils import compute_heading

def generate_gyroscope_signals(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Génère les signaux gyroscopiques (gyro_x, gyro_y, gyro_z) à partir de la trajectoire et des événements.

    - gyro_z : taux de rotation autour de l'axe vertical (cap), estimé à partir du heading.
    - gyro_x, gyro_y : bruit léger par défaut, enrichi selon les événements spécifiques.

    Args:
        df (pd.DataFrame): Doit contenir une colonne 'heading' en radians et éventuellement 'event'.
        hz (int): Fréquence d'échantillonnage (10 Hz par défaut).

    Returns:
        pd.DataFrame: avec colonnes gyro_x, gyro_y, gyro_z ajoutées ou modifiées.
    """
    df = df.copy()

    # gyro_z : taux de rotation dérivé du heading (en rad/s)
    heading = df["heading"].ffill().fillna(0).values
    delta_heading = np.unwrap(np.gradient(heading))  # évite les discontinuités ±pi
    df["gyro_z"] = delta_heading * hz

    # gyro_x/y : bruit faible par défaut
    np.random.seed(42)  # reproductibilité
    df["gyro_x"] = np.random.normal(0.01, 0.02, size=len(df))  # tangage faible
    df["gyro_y"] = np.random.normal(0.01, 0.02, size=len(df))  # roulis faible

    # Appliquer les signatures gyroscopiques des événements
    n = len(df)
    for i, row in df.iterrows():
        evt = row.get("event", None)
        if pd.isna(evt):
            continue

        # Fenêtre d'application autour de l'événement (ex : 0.5 s)
        window = 5
        i0 = max(0, i - window)
        i1 = min(n, i + window + 1)

        if evt == "dos_dane":
            n_pts = i1 - i0
            if n_pts > 0:
                df.loc[i0:i1 - 1, "gyro_x"] += np.sin(np.linspace(0, np.pi, n_pts)) * 2.0
        elif evt == "freinage":
            n_pts = i1 - i0
            if n_pts > 0:
                df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(0.5, -0.5, n_pts)
        elif evt == "acceleration":
            n_pts = i1 - i0
            if n_pts > 0:
                df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(-0.5, 0.5, n_pts)
        elif evt == "trottoir":
            n_pts = i1 - i0
            if n_pts > 0:
                df.loc[i0:i1 - 1, "gyro_y"] += np.sin(np.linspace(0, np.pi, n_pts)) * 2.0
        elif evt == "nid_de_poule":
            n_pts = i1 - i0
            if n_pts > 0:
                df.loc[i0:i1 - 1, "gyro_z"] += np.random.normal(0, 3.0, n_pts)

    df[["gyro_x", "gyro_y", "gyro_z"]] = df[["gyro_x", "gyro_y", "gyro_z"]].round(4)
    return df


# Nouvelle fonction : recalculer l'accélération inertielle à partir des vitesses GPS et heading
def recompute_inertial_acceleration(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Recalcule l'accélération inertielle (acc_x, acc_y) à partir des vitesses GPS.
    L'accélération verticale (acc_z) est laissée vide ou constante.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'lat', 'lon', 'speed'.
        hz (int): Fréquence d’échantillonnage.

    Returns:
        pd.DataFrame: avec colonnes acc_x, acc_y, acc_z recalculées.
    """
    df = df.copy()

    if "speed" not in df.columns or "heading" not in df.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'speed' et 'heading'.")

    v = df["speed"].fillna(0).values
    heading = df["heading"].fillna(0).values

    dv = np.gradient(v) * hz
    dh = np.gradient(heading) * hz

    acc_x = dv * np.cos(heading) - v * np.sin(heading) * dh
    acc_y = dv * np.sin(heading) + v * np.cos(heading) * dh

    df["acc_x"] = acc_x
    df["acc_y"] = acc_y
    # Vérifie si 'acc_z' existait initialement, sinon ne pas le forcer à 0
    if "acc_z" not in df.columns:
        df["acc_z"] = 0.0
    return df


__all__ = [
    "generate_gyroscope_signals",
    "recompute_inertial_acceleration",
]