import unittest
import numpy as np
import pandas as pd
from geopy.distance import distance
from core.geo_utils import compute_heading

class TestTraceIntegrity(unittest.TestCase):
    def setUp(self):
        # Exemple minimaliste de DataFrame de trace simulée
        self.df = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01 12:00:00', periods=10, freq='100ms'),
            'lat': [48.8566, 48.8567, 48.8568, 48.8569, 48.8570, 48.8571, 48.8572, 48.8573, 48.8574, 48.8575],
            'lon': [2.3522, 2.3523, 2.3524, 2.3525, 2.3526, 2.3527, 2.3528, 2.3529, 2.3530, 2.3531],
            'speed': [36.0]*10,  # km/h constant (10 m/s)
            'heading': [0]*10,
            'acc_x': [0.0]*10,
            'acc_y': [0.0]*10,
            'event': [np.nan]*10
        })
        self.tol_time = 1e-3  # tolérance 1 ms
        self.tol_dist = 1.0   # tolérance distance en mètres
        self.tol_heading = 5  # tolérance heading en degrés
        self.tol_acc = 0.5    # tolérance accélération m/s²

    def test_uniform_timestamps(self):
        dt = self.df['timestamp'].diff().dropna().dt.total_seconds()
        self.assertAlmostEqual(dt.max() - dt.min(), 0.0, delta=self.tol_time,
                               msg="Timestamps non uniformes")

    def test_distance_speed_coherence(self):
        dt = self.df['timestamp'].diff().dropna().dt.total_seconds().values
        for i in range(1, len(self.df)):
            dist_m = distance(
                (self.df.loc[i-1,'lat'], self.df.loc[i-1,'lon']),
                (self.df.loc[i,'lat'], self.df.loc[i,'lon'])
            ).meters
            speed_m_s = self.df.loc[i,'speed'] * 1000 / 3600
            expected_dist = speed_m_s * dt[i-1]
            self.assertAlmostEqual(dist_m, expected_dist, delta=self.tol_dist,
                                   msg=f"Incohérence distance-vitesse à l'index {i}")

    def test_heading_consistency(self):
        for i in range(1, len(self.df)-1):
            expected_heading = compute_heading(
                self.df.loc[i,'lat'], self.df.loc[i,'lon'],
                self.df.loc[i+1,'lat'], self.df.loc[i+1,'lon']
            )
            diff = abs((self.df.loc[i,'heading'] - expected_heading) % 360)
            diff = min(diff, 360 - diff)
            self.assertLessEqual(diff, self.tol_heading,
                                 msg=f"Heading incohérent à l'index {i}")

    def test_acceleration_longitudinal(self):
        dt = self.df['timestamp'].diff().dropna().dt.total_seconds().values
        speed_m_s = self.df['speed'] * 1000 / 3600
        acc_x_calc = np.diff(speed_m_s) / dt
        acc_x_reported = self.df['acc_x'].iloc[1:].values
        for i in range(len(acc_x_calc)):
            self.assertAlmostEqual(acc_x_calc[i], acc_x_reported[i], delta=self.tol_acc,
                                   msg=f"acc_x incohérent à l'index {i+1}")

    # Optionnel: test acc_y lié à virages

    # Optionnel: test cohérence événementielle (présence, vitesse, accélérations)

if __name__ == '__main__':
    unittest.main()
