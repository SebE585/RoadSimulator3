"""
utils.py
Utilitaires génériques pour la validation et l'export CSV dans RoadSimulator3
"""

import pandas as pd
import numpy as np
import os
import warnings
def deprecated(func):
    """
    Décorateur pour marquer une fonction comme obsolète.
    Affiche un avertissement à l'exécution.
    """
    def wrapper(*args, **kwargs):
        warnings.warn(f"[DEPRECATED] La fonction '{func.__name__}' est obsolète.", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper


def ensure_event_column_object(df):
    """
    Assure que la colonne 'event' existe et est de type object (pour accepter des chaînes).
    """
    if 'event' not in df.columns:
        df['event'] = pd.Series(dtype='object')
    else:
        df['event'] = df['event'].astype('object')
    return df

@deprecated
def find_latest_trace(base_dir='data/simulations'):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Recherche le dernier fichier trace.csv dans le répertoire spécifié.

    Args:
        base_dir (str): Répertoire racine contenant les sous-dossiers de simulation.

    Returns:
        str or None: Chemin complet vers le dernier trace.csv, ou None si absent.
    """
    if not os.path.exists(base_dir):
        return None
    out_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not out_dirs:
        return None
    latest_dir = max(out_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    trace_path = os.path.join(base_dir, latest_dir, 'trace.csv')
    return trace_path if os.path.exists(trace_path) else None

CSV_COLUMNS = [
    'timestamp', 'lat', 'lon', 'altitude', 'speed', 
    'acc_x', 'acc_y', 'acc_z', 
    'gyro_x', 'gyro_y', 'gyro_z',
    'event', 'heading', 'sinuosity', 'curvature',
    'osm_highway', 'road_type', 'slope_percent'
]

def ensure_csv_column_order(df):
    """
    Réordonne le DataFrame selon l'ordre strict attendu pour l'export CSV.
    Vérifie la présence de toutes les colonnes nécessaires, y compris les colonnes gyroscopiques.
    Ignore les colonnes en surplus.
    """
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes avant export CSV : {missing}")
    return df[CSV_COLUMNS]


def export_csv(df, output_path):
    """
    Exporte le DataFrame au format CSV avec ordre des colonnes garanti.
    """
    df_ordered = ensure_csv_column_order(df)
    df_ordered.to_csv(output_path, index=False)
    print(f"✅ CSV exporté avec succès : {output_path}")

def load_trace(csv_path):
    """
    Charge un CSV de simulation en DataFrame avec parsing du timestamp.
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    return df

@deprecated
@deprecated
def get_log_path(log_type, timestamp, base_dir='data/simulations'):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Génère le chemin du fichier de log pour un type donné.

    Args:
        log_type (str): type de log ('errors', 'events', 'summary', etc.)
        timestamp (str): timestamp de la simulation (format YYYYMMDD_HHMMSS)
        base_dir (str): répertoire racine des logs

    Returns:
        str: chemin complet du fichier log
    """
    folder = os.path.join(base_dir, f'simulated_{timestamp}')
    os.makedirs(folder, exist_ok=True)
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("[DEPRECATED] Utilisation de get_log_path")
    return os.path.join(folder, f'{log_type}.log')

def ensure_strictly_increasing_timestamps(df, col="timestamp"):
    """
    Corrige les timestamps non strictement croissants :
    - Tri par timestamp si nécessaire.
    - Décalage léger des doublons (ajout de 1ms) pour éviter les égalités.
    """
    if not df[col].is_monotonic_increasing:
        print("🔧 Correction des timestamps non croissants...")

        # 1. Tri
        df = df.sort_values(by=col).reset_index(drop=True)

        # 2. Correction des égalités
        timestamps = df[col].values.astype('datetime64[ns]')
        diffs = np.diff(timestamps)

        # Détecte les doublons ou égalités successives
        non_increasing_idx = np.where(diffs <= np.timedelta64(0, 'ns'))[0]
        if len(non_increasing_idx) > 0:
            print(f"⚠️ {len(non_increasing_idx)} timestamps égaux ou décroissants détectés. Application de décalages...")
            for idx in non_increasing_idx:
                # Décale le timestamp suivant de +1 ms par rapport au précédent
                timestamps[idx + 1] = timestamps[idx] + np.timedelta64(1, 'ms')
            df[col] = timestamps
        else:
            print("✅ Tri suffisant, aucun doublon strict à corriger.")
    else:
        print("✅ Timestamps déjà strictement croissants.")
    return df


# Nouvelle fonction pour générer le chemin de sortie d'une simulation
def get_simulation_output_dir(timestamp, base_dir="data/simulations"):
    """
    Génère le chemin complet du répertoire de sortie pour une simulation.

    Args:
        timestamp (str): timestamp de la simulation (format YYYYMMDD_HHMMSS)
        base_dir (str): répertoire racine de sortie

    Returns:
        str: chemin du répertoire simulation complet
    """
    output_dir = os.path.join(base_dir, f"simulated_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Nouvelle fonction pour obtenir le dernier dossier de simulation
def get_latest_simulation_dir(base_dir='data/simulations'):
    """
    Retourne le chemin complet du dernier dossier de simulation dans base_dir.
    """
    if not os.path.exists(base_dir):
        return None
    out_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not out_dirs:
        return None
    latest_dir = max(out_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    return os.path.join(base_dir, latest_dir)

def save_dataframe_as_csv(df, filename):
    """
    Enregistre un DataFrame au format CSV avec colonnes ordonnées.
    """
    # Ajouter les colonnes manquantes avec NaN
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df_ordered = ensure_csv_column_order(df)
    df_ordered.to_csv(filename, index=False)
    print(f"✅ Fichier CSV sauvegardé : {filename}")
