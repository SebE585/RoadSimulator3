import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from core.kinematics import compute_cumulative_distance, compute_total_distance




def spatial_reprojection(df, speed_target, dt=0.1):
    df = df.reset_index(drop=True).copy()

    cumdist = compute_cumulative_distance(df)
    total_dist = compute_total_distance(df)

    if isinstance(speed_target, pd.Series):
        speed_m_s = speed_target.to_numpy()
        speed_m_s = np.nan_to_num(speed_m_s, nan=np.nanmean(speed_m_s))
    elif isinstance(speed_target, (float, int)):
        speed_m_s = float(speed_target)
        speed_m_s = np.full(len(cumdist), speed_m_s)
    else:
        raise TypeError("speed_target doit être un float ou une pd.Series")

    # Ne pas interpoler lat/lon, générer timestamp régulier, interpoler autres colonnes numériques
    new_df = pd.DataFrame()
    for col in df.columns:
        if col in ['lat', 'lon']:
            # Ne pas réinterpoler les coordonnées GPS
            new_df[col] = df[col].values
            continue
        if col == 'timestamp':
            # Générer de nouveaux timestamps réguliers
            start = df['timestamp'].iloc[0]
            end = df['timestamp'].iloc[-1]
            freq_hz = int(round(1.0 / dt))
            new_timestamps = pd.date_range(start=start, end=end, freq=f"{int(1000/freq_hz)}ms")
            new_df[col] = new_timestamps[:len(df)]
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            f_col = interp1d(cumdist, df[col], bounds_error=False, fill_value='extrapolate')
            new_df[col] = f_col(cumdist)
        else:
            new_df[col] = df[col]

    from simulator.events.tracker import EventCounter
    tracker = EventCounter()
    tracker.count_from_dataframe(new_df)
    tracker.show("Après reprojection (spatial_reprojection)")
    return new_df


# core/reprojection.py

def resample_time(df: pd.DataFrame, freq_hz: int = 10) -> pd.DataFrame:
    """
    Rééchantillonne le DataFrame à une fréquence régulière (10 Hz par défaut)
    avec interpolation des colonnes numériques.
    """
    df = df.copy()

    # Assure que le timestamp est bien en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Supprimer les doublons sur 'timestamp' pour éviter erreurs lors du reindex
    if df['timestamp'].duplicated().any():
        df = df.loc[~df['timestamp'].duplicated(keep='first')]
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Doublons timestamp détectés et supprimés avant rééchantillonnage.")

    # Génère une nouvelle base temporelle régulière
    start = df['timestamp'].iloc[0]
    end = df['timestamp'].iloc[-1]
    new_timestamps = pd.date_range(start=start, end=end, freq=f"{int(1000/freq_hz)}ms")  # L = ms

    # Préserver explicitement la colonne 'event' si elle existe, pour éviter la propagation incorrecte
    if "event" in df.columns:
        original_event = df[["timestamp", "event"]].dropna().copy()

    # On interpole les colonnes numériques
    df = df.set_index('timestamp').reindex(new_timestamps)
    df.index.name = 'timestamp'

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()

    # Récupère les colonnes non numériques (remplissage par la valeur précédente), sans 'event'
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    non_event_cols = [col for col in non_numeric_cols if col != "event"]
    df[non_event_cols] = df[non_event_cols].ffill().bfill()

    # Réaffecter la colonne 'event' pour éviter la propagation de labels
    if "event" in df.columns:
        df = df.reset_index()
        df["event"] = np.nan
        for _, row in original_event.iterrows():
            ts = row["timestamp"]
            if ts in df["timestamp"].values:
                df.loc[df["timestamp"] == ts, "event"] = row["event"]
    else:
        df = df.reset_index()

    from simulator.events.tracker import EventCounter
    tracker = EventCounter()
    tracker.count_from_dataframe(df)
    tracker.show("Après reprojection (resample_time)")
    return df
