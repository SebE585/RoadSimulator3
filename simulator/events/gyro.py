# simulator/events/gyro.py
import numpy as np
import pandas as pd

# Canonicalize common event names (supports client vocabulary)
_EVENT_ALIASES = {
    "dos_dane": "dos_d_ane",          # legacy typo → canonical
    "dos_d_ane": "dos_d_ane",
    "nid_de_poule": "nid_de_poule",
    "freinage": "freinage",
    "freinage_fort": "freinage",
    "acceleration": "acceleration",
    "acceleration_initiale": "acceleration_initiale",
    "trottoir": "trottoir",
}

def _canon(ev: str | None) -> str | None:
    if not isinstance(ev, str):
        return None
    ev = ev.strip()
    return _EVENT_ALIASES.get(ev, ev)

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

    Garantit la présence des colonnes gyro_x/gyro_y/gyro_z même si aucun événement n'est présent
    ou si l'événement est "acceleration_initiale" (retour client v1.0).
    """
    df = df.copy()

    # Ensure minimal columns exist even on empty/degenerate inputs
    if df is None or df.empty:
        df = pd.DataFrame(index=[])
        for c in ("gyro_x", "gyro_y", "gyro_z"):
            df[c] = pd.Series(dtype="float32")
        return df

    # gyro_z : taux de rotation dérivé du heading (en rad/s)
    if "heading" not in df.columns:
        # Pas de heading : colonnes gyro à 0.0 pour garantir présence
        df["gyro_x"] = 0.0
        df["gyro_y"] = 0.0
        df["gyro_z"] = 0.0
        return df

    heading = df["heading"].ffill().fillna(0).to_numpy()
    dhead = np.gradient(np.unwrap(heading)) * hz  # évite les discontinuités ±pi puis dérive
    df["gyro_z"] = dhead

    # Bruit faible par défaut (ne modifie pas l'état global du RNG)
    rng = np.random.default_rng(42)
    df["gyro_x"] = rng.normal(0.01, 0.02, size=len(df))
    df["gyro_y"] = rng.normal(0.01, 0.02, size=len(df))

    n = len(df)
    for i, row in df.iterrows():
        evt_raw = row.get("event", None)
        evt = _canon(evt_raw)
        if not evt:
            continue

        # Fenêtre d'application autour de l'événement (±0.5 s à 10 Hz → 5 pts)
        window = max(1, int(0.5 * hz))
        i0 = max(0, i - window)
        i1 = min(n, i + window + 1)
        n_pts = i1 - i0
        if n_pts <= 0:
            continue

        if evt == "dos_d_ane":
            df.loc[i0:i1 - 1, "gyro_x"] += np.sin(np.linspace(0, np.pi, n_pts)) * 2.0
        elif evt == "freinage":
            df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(0.5, -0.5, n_pts)
        elif evt == "acceleration":
            df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(-0.5, 0.5, n_pts)
        elif evt == "acceleration_initiale":
            # Cas client: s'assurer de la présence de gyro même si l'événement est initial
            # Signature douce (petit bump tangage) + légère dérive de yaw
            df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(-0.3, 0.3, n_pts)
            df.loc[i0:i1 - 1, "gyro_z"] += np.linspace(0.0, 0.05, n_pts)
        elif evt == "trottoir":
            df.loc[i0:i1 - 1, "gyro_y"] += np.sin(np.linspace(0, np.pi, n_pts)) * 2.0
        elif evt == "nid_de_poule":
            df.loc[i0:i1 - 1, "gyro_z"] += np.random.normal(0, 3.0, n_pts)

    df["gyro_x"] = df["gyro_x"].astype("float32")
    df["gyro_y"] = df["gyro_y"].astype("float32")
    df["gyro_z"] = df["gyro_z"].astype("float32")

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

    df["acc_x"] = df["acc_x"].astype("float32")
    df["acc_y"] = df["acc_y"].astype("float32")
    df["acc_z"] = df["acc_z"].astype("float32")
    return df


__all__ = [
    "generate_gyroscope_signals",
    "recompute_inertial_acceleration",
]