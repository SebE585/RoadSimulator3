import unittest
import pandas as pd
import numpy as np

from core.kinematics import (
    compute_target_speed,
    interpolate_speed_towards_target,
    recalc_accelerations,
    recompute_gps_positions,
)

class TestKinematics(unittest.TestCase):

    def setUp(self):
        # Configuration exemple pour speed_profile
        self.config = {
            'speed_profile': {
                'autoroute': {'v_max': 130, 'v_min': 100, 'k': 20},
                'urbaine': {'v_max': 50, 'v_min': 30, 'k': 40},
                'chemin': {'v_max': 30, 'v_min': 15, 'k': 50},
            }
        }
        # DataFrame de test simple
        self.df = pd.DataFrame({
            'lat': [48.8566, 48.8567, 48.8568, 48.8569, 48.8570],
            'lon': [2.3522, 2.3523, 2.3524, 2.3525, 2.3526],
            'speed': [50, 52, 54, 53, 51],
            'heading': [0, 10, 20, 15, 10],
            'road_type': ['urbaine'] * 5,
            'sinuosity': [0.1, 0.15, 0.2, 0.18, 0.16],
            'event': [None] * 5
        })

    def test_compute_target_speed(self):
        v = compute_target_speed('urbaine', 0.1, self.config)
        self.assertTrue(30 <= v <= 50)  # doit être entre v_min et v_max
        self.assertAlmostEqual(v, 50 - 40 * 0.1)

    def test_interpolate_speed_towards_target_respects_events(self):
        # Ajout d’un événement freinage au milieu
        self.df.loc[2, 'event'] = 'freinage'
        self.df['target_speed'] = [45, 48, 40, 42, 43]
        new_df = interpolate_speed_towards_target(self.df, acceleration=5.0)
        # Vitesse au point avec freinage ne doit pas changer
        self.assertEqual(new_df.loc[2, 'speed'], self.df.loc[2, 'speed'])
        # Vitesse aux autres points doit tendre vers target_speed
        self.assertNotEqual(new_df.loc[1, 'speed'], self.df.loc[1, 'speed'])

    def test_recalc_accelerations_consistency(self):
        df = self.df.copy()
        df['speed'] = [0, 10, 20, 15, 5]  # km/h
        df['heading'] = [0, 0, 0, 90, 90]  # deg
        df = recalc_accelerations(df)
        # acc_x should be positive where speed increases
        self.assertTrue(df.loc[1, 'acc_x'] > 0)
        # acc_y should reflect change in heading
        self.assertNotEqual(df.loc[3, 'acc_y'], 0)

    def test_recompute_gps_positions_monotonic(self):
        df = self.df.copy()
        df['speed'] = [36, 36, 36, 36, 36]  # 36 km/h = 10 m/s constant
        df['heading'] = [0, 0, 0, 0, 0]
        df_new = recompute_gps_positions(df)
        # latitudes doivent augmenter légèrement car heading=0 (nord)
        self.assertTrue(all(np.diff(df_new['lat']) > 0))
        # longitudes doivent être presque constantes
        self.assertTrue(np.allclose(np.diff(df_new['lon']), 0, atol=1e-6))

def clean_and_recompute(df, dt=0.1):
    """
    Détecte anomalies GPS et vitesse, nettoie les points, puis
    recalcul :
      - positions GPS,
      - heading (avec lissage),
      - accélérations longitudinales et latérales.

    Args:
        df (pd.DataFrame): DataFrame simulé avec colonnes lat, lon, speed, heading, etc.
        dt (float): intervalle temporel entre points (s)

    Returns:
        pd.DataFrame nettoyé et recalculé
    """
    gps_jumps = detect_gps_jumps(df)
    speed_anomalies = detect_speed_anomalies(df)
    indices_to_clean = sorted(set(gps_jumps + speed_anomalies))

    if indices_to_clean:
        print(f"[INFO] Nettoyage de {len(indices_to_clean)} points anormaux détectés (GPS + vitesse)...")
        df = clean_anomalies(df, indices_to_clean)

        df = recompute_gps_positions(df, dt=dt)

        # Lissage du heading
        df['heading'] = smooth_heading(df['heading'])

        # Recalcul des accélérations
        df = recalc_accelerations(df, dt=dt)

    else:
        print("[INFO] Aucun point anormal détecté, pas de nettoyage nécessaire.")

    return df


if __name__ == '__main__':
    unittest.main()
