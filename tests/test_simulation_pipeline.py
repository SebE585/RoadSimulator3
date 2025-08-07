import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from simulator.pipeline.pipeline import SimulationPipeline

class TestSimulationPipeline(unittest.TestCase):

    def setUp(self):
        self.config = {
            'simulation': {
                'hz': 10,
                'step_m': 0.83,
                'initial_speed_kmh': 30,
                'initial_ramp_duration_s': 5,
                'smoothing_window': 3,
                'max_speed_delta_kmh': 5,
                'inertial_noise_std': 0.03,
                'dynamic_gain': True,
                'mnt_path': "data/normandy_l93.tif",
                'cities_coords': [
                    [49.2738, 1.2127],
                    [49.4431, 1.0993],
                    [49.3568, 1.2342],
                    [49.3653, 1.2361],
                    [49.3305, 1.1811],
                    [49.3364, 1.1733]
                ]
            },
            'events': {
                'acceleration': {'count': 0},
                'freinage': {'count': 0},
                'dos_dane': {'count': 0},
                'trottoir': {'count': 0},
                'nid_de_poule': {'count': 0},
                'stops': {'count': 0},
                'waits': {'count': 0}
            }
        }
        self.pipeline = SimulationPipeline(self.config)

    def test_pipeline_runs_on_minimal_dataframe(self):
        """Teste que la pipeline s'exécute sur un très petit DataFrame (3 points)"""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=3, freq='100ms'),
            'lat': [49.0, 49.00001, 49.00002],
            'lon': [1.0, 1.00001, 1.00002],
            'speed': [0.0, 10.0, 20.0]
        })
        result = self.pipeline.run(df.copy())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('acc_x', result.columns)

    def test_pipeline_on_simulated_data(self):
        """Teste la pipeline sur une trajectoire interpolée plus réaliste"""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='100ms'),
            'lat': np.linspace(49.0, 49.005, 50),
            'lon': np.linspace(1.0, 1.005, 50),
            'speed': np.linspace(0, 50, 50)
        })
        result = self.pipeline.run(df.copy())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('acc_x', result.columns)
        self.assertIn('event', result.columns)

if __name__ == '__main__':
    unittest.main()
