import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@deprecated
def load_meteo_data(filepath):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Charge un fichier CSV de données météo contenant :
    timestamp, lat, lon, temperature, humidity, wind_speed, precipitation
    """
    return pd.read_csv(filepath, parse_dates=['timestamp'])

@deprecated
def enrich_with_meteo(df, meteo_df):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Enrichit le DataFrame df avec les colonnes météo par interpolation temporelle et spatiale.
    Colonnes ajoutées : temperature, humidity, wind_speed, precipitation
    """
    # Initialiser les colonnes à NaN
    df['temperature'] = np.nan
    df['humidity'] = np.nan
    df['wind_speed'] = np.nan
    df['precipitation'] = np.nan

    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        lat, lon = row['lat'], row['lon']

        # Filtrer météo proche en temps (+/- 30 min)
        time_window = meteo_df[
            (meteo_df['timestamp'] >= timestamp - timedelta(minutes=30)) &
            (meteo_df['timestamp'] <= timestamp + timedelta(minutes=30))
        ]

        if time_window.empty:
            continue

        # Approximation : on prend le point météo spatialement le plus proche
        time_window['dist'] = np.sqrt((time_window['lat'] - lat)**2 + (time_window['lon'] - lon)**2)
        closest = time_window.loc[time_window['dist'].idxmin()]

        df.at[idx, 'temperature'] = closest['temperature']
        df.at[idx, 'humidity'] = closest['humidity']
        df.at[idx, 'wind_speed'] = closest['wind_speed']
        df.at[idx, 'precipitation'] = closest['precipitation']

    return df

@deprecated
def enrich_with_meteo_dummy(df):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Version fallback : remplit les colonnes météo avec des valeurs réalistes simulées.
    """
    df['temperature'] = np.random.normal(15, 5, len(df))
    df['humidity'] = np.random.uniform(40, 90, len(df))
    df['wind_speed'] = np.random.uniform(0, 15, len(df))
    df['precipitation'] = np.random.choice([0, 0.1, 0.5, 1.0], size=len(df), p=[0.8, 0.1, 0.05, 0.05])

    return df
