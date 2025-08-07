"""Fonctions utilitaires pour la gestion des événements dans les DataFrames de simulation.

Contient notamment :
- `ensure_event_column_object(df)` : garantit que la colonne 'event' est bien typée `object`
  afin d’éviter les warnings pandas lors des affectations (ex : chaînes, NaN...).
"""

import pandas as pd
import numpy as np
import logging

try:
    from core.decorators import deprecated
except ImportError:
    def deprecated(func):
        return func

logger = logging.getLogger(__name__)

@deprecated
def ensure_event_column_object(df):
    """
    Assure que la colonne 'event' est bien de type 'object' (string),
    ce qui évite les warnings pandas lors de l'affectation.
    """
    if 'event' not in df.columns:
        df['event'] = pd.Series([np.nan] * len(df), dtype='object')
    elif df['event'].dtype != 'object':
        df['event'] = df['event'].astype('object')
    return df


@deprecated
def clean_invalid_events(df):
    """
    Supprime les événements invalides (non string ou NaN dans des colonnes critiques).
    Remplace également les chaînes vides ou les types inattendus par NaN dans la colonne 'event'.
    """
    if 'event' not in df.columns:
        return df

    # Remplace les valeurs non-string ou vides par NaN
    df['event'] = df['event'].apply(lambda x: x if isinstance(x, str) and x.strip() else np.nan)

    # Supprime les lignes où lat/lon/timestamp sont NaN
    df = df[df[['lat', 'lon', 'timestamp']].notna().all(axis=1)]

    return df



@deprecated
def marquer_livraisons(df, prefix="stop_", start_index=1):
    """
    Marque les points de livraison dans le DataFrame en ajoutant des événements de type 'stop_xxx'
    sur les lignes déjà marquées comme livraison (ex: colonne 'delivery' == True).

    Args:
        df (pd.DataFrame): Le DataFrame contenant les colonnes 'lat', 'lon' et éventuellement 'delivery'.
        prefix (str): Le préfixe utilisé pour les événements de livraison.
        start_index (int): L'index de départ pour la numérotation (par défaut 1).

    Returns:
        pd.DataFrame: Le DataFrame mis à jour avec les événements de livraison dans la colonne 'event'.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("marquer_livraisons attend un DataFrame en entrée, pas une liste.")

    ensure_event_column_object(df)
    if 'delivery' not in df.columns:
        return df

    delivery_indices = df.index[df['delivery'] == True].tolist()
    for i, idx in enumerate(delivery_indices, start=start_index):
        df.at[idx, 'event'] = f"{prefix}{i:03d}"

    return df