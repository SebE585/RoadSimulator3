"""
roundabouts.py

Ce module permet d'injecter des signatures inertielle réalistes associées aux virages
et aux ronds-points détectés sur une trajectoire simulée ou mesurée.

Fonctionnalités principales :
- `inject_inertial_signature_for_turns` : simule la courbure et les accélérations associées à un virage.
- `generate_inertial_signature_for_osrm_roundabouts` : injecte une signature inertielle et modifie la géométrie
  pour représenter un rond-point autour d’un centre détecté (via OSRM par exemple).

Les injections modifient directement les colonnes :
- `acc_x`, `acc_y`, `acc_z` (composantes inertielle),
- `lat`, `lon` (trajet modifié en forme de courbe ou cercle),
- `event` (ajout du label 'virage' ou 'rond_point').

Utilisation typique : enrichir une trajectoire de test pour l'analyse inertielle ou la détection automatique
de configurations routières spécifiques.

Auteurs : RoadSimulator3 team
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

HZ = 10

# Les fonctions inject_inertial_signature_for_turns
# et generate_inertial_signature_for_osrm_roundabouts seront placées ici

# Exemple :
def inject_inertial_signature_for_turns(df, turn_infos, window_size=10, hz=HZ):
    meters_per_degree_lat = 111320

    for turn in turn_infos:
        idx_center = turn['index']
        start_idx = max(0, idx_center - window_size)
        end_idx = min(len(df), idx_center + window_size + 1)

        radius_m = np.random.uniform(10, 50)
        angle_total = np.pi / 4
        delta_angle_per_point = angle_total / (end_idx - start_idx)

        angle = 0

        for idx in range(start_idx, end_idx):
            relative_pos = (idx - idx_center) / window_size
            df.at[idx, 'acc_y'] += 1.5 * np.sin(np.pi * relative_pos)
            df.at[idx, 'acc_x'] += 0.3 * np.cos(np.pi * relative_pos)
            df.at[idx, 'acc_z'] = 9.81
            df.at[idx, 'event'] = 'virage'

            d_lat = (radius_m * np.sin(angle)) / meters_per_degree_lat
            d_lon = (radius_m * (1 - np.cos(angle))) / (
                40075000 * np.cos(np.radians(df.at[idx, 'lat'])) / 360)

            df.at[idx, 'lat'] += d_lat
            df.at[idx, 'lon'] += d_lon
            angle += delta_angle_per_point

        logger.debug(f"Virage injecté autour de l'index {idx_center} avec rayon ~{radius_m:.1f}m")

    return df

def inject_inertial_signature_for_turns(df, turn_infos, window_size=10, hz=10):
    """
    Injecte une signature inertielle et modifie lat/lon pour simuler un virage.

    Args:
        df (pd.DataFrame): dataframe trajectoire
        turn_infos (list): liste de dicts avec {'index': index du centre du virage}
        window_size (int): nombre de points autour du centre à modifier
        hz (int): fréquence de simulation (par défaut 10Hz)

    Returns:
        df (pd.DataFrame): modifié avec signatures inertielle et géométrie du virage
    """
    meters_per_degree_lat = 111320  # approx conversion latitude deg -> m

    for turn in turn_infos:
        idx_center = turn['index']
        start_idx = max(0, idx_center - window_size)
        end_idx = min(len(df), idx_center + window_size + 1)

        radius_m = np.random.uniform(10, 50)  # rayon du virage en mètres
        angle_total = np.pi / 4  # virage de 45°
        delta_angle_per_point = angle_total / (end_idx - start_idx)

        angle = 0

        for idx in range(start_idx, end_idx):
            relative_pos = (idx - idx_center) / window_size

            # Inertie simulée
            df.at[idx, 'acc_y'] += 1.5 * np.sin(np.pi * relative_pos)
            df.at[idx, 'acc_x'] += 0.3 * np.cos(np.pi * relative_pos)
            df.at[idx, 'acc_z'] = 9.81
            df.at[idx, 'event'] = 'virage'

            # Décalage GPS
            d_lat = (radius_m * np.sin(angle)) / meters_per_degree_lat
            d_lon = (radius_m * (1 - np.cos(angle))) / (
                40075000 * np.cos(np.radians(df.at[idx, 'lat'])) / 360
            )

            df.at[idx, 'lat'] += d_lat
            df.at[idx, 'lon'] += d_lon

            angle += delta_angle_per_point

        print(f"[DEBUG] Virage injecté autour de l'index {idx_center} avec rayon ~{radius_m:.1f}m")

    return df


def generate_inertial_signature_for_osrm_roundabouts(df, osrm_roundabouts, window_size=20):
    """
    Injecte la signature inertielle + modifie lat/lon pour simuler un rond-point circulaire.

    Args:
        df (pd.DataFrame): trajectoire
        osrm_roundabouts (list): liste des ronds-points détectés par OSRM (lon, lat)
        window_size (int): nombre de points autour du centre du rond-point à modifier

    Returns:
        df (pd.DataFrame): modifié avec signature inertielle et géométrie simulée
    """
    meters_per_degree_lat = 111320  # approx conversion latitude deg -> m

    for r in osrm_roundabouts:
        lat_center, lon_center = r['location'][1], r['location'][0]
        distances = np.sqrt((df['lat'] - lat_center) ** 2 + (df['lon'] - lon_center) ** 2)
        idx_center = distances.idxmin()
        start_idx = max(0, idx_center - window_size)
        end_idx = min(len(df), idx_center + window_size + 1)

        radius_m = np.random.uniform(12, 20)  # rayon du rond-point simulé (en mètres)
        full_rotation = 2 * np.pi
        delta_angle = full_rotation / (end_idx - start_idx)

        current_angle = 0

        for idx in range(start_idx, end_idx):
            relative_pos = (idx - idx_center) / window_size

            # Signature inertielle
            df.at[idx, 'acc_y'] = 2.0 * np.sin(np.pi * relative_pos)
            df.at[idx, 'acc_x'] = 0.5 * np.cos(np.pi * relative_pos)
            df.at[idx, 'acc_z'] = 9.81
            df.at[idx, 'event'] = 'rond_point'

            # Décalage GPS : cercle autour du centre du rond-point
            d_lat = (radius_m * np.cos(current_angle)) / meters_per_degree_lat
            d_lon = (radius_m * np.sin(current_angle)) / (
                40075000 * np.cos(np.radians(df.at[idx, 'lat'])) / 360
            )

            df.at[idx, 'lat'] = lat_center + d_lat
            df.at[idx, 'lon'] = lon_center + d_lon

            current_angle += delta_angle
            df.at[idx, 'heading'] += np.degrees(delta_angle)

        print(f"[DEBUG] Rond-point injecté autour de l'index {idx_center} avec rayon ~{radius_m:.1f}m")

    return df
