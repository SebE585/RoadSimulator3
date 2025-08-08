# core/gyroscope.py
import numpy as np
import pandas as pd

class GyroscopeSimulator:
    """
    Simule les signaux gyroscopiques (gyro_x, gyro_y, gyro_z) 
    pour un DataFrame de trajectoire inertielle à 10 Hz.
    """

    def __init__(self, noise_std=0.02, drift_per_sec=0.001, seed=None):
        """
        :param noise_std: Écart-type du bruit blanc (rad/s)
        :param drift_per_sec: Dérive lente simulée (rad/s par seconde)
        :param seed: Graine aléatoire pour reproductibilité
        """
        self.noise_std = noise_std
        self.drift_per_sec = drift_per_sec
        if seed is not None:
            np.random.seed(seed)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les colonnes gyro_x, gyro_y, gyro_z dans le DataFrame.

        :param df: DataFrame contenant au moins 'acc_x', 'acc_y', 'acc_z', 'timestamp'
        :return: DataFrame avec gyro_x, gyro_y, gyro_z ajoutés
        """
        n = len(df)
        if n == 0:
            df["gyro_x"] = df["gyro_y"] = df["gyro_z"] = np.nan
            return df

        # Dérive lente sur la durée
        drift = np.linspace(0, self.drift_per_sec * n / 10, n)

        # Bruit blanc
        noise_x = np.random.normal(0, self.noise_std, n)
        noise_y = np.random.normal(0, self.noise_std, n)
        noise_z = np.random.normal(0, self.noise_std, n)

        # Signature basique liée à l'accélération latérale et longitudinale
        gyro_x = drift + noise_x
        gyro_y = drift + noise_y
        gyro_z = drift + noise_z

        df["gyro_x"] = gyro_x
        df["gyro_y"] = gyro_y
        df["gyro_z"] = gyro_z
        return df

# Utilisation rapide
if __name__ == "__main__":
    # Exemple de DataFrame factice
    df_test = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="100ms"),
        "acc_x": np.random.randn(100),
        "acc_y": np.random.randn(100),
        "acc_z": np.random.randn(100),
    })

    sim = GyroscopeSimulator(noise_std=0.02, drift_per_sec=0.001, seed=42)
    df_test = sim.generate(df_test)
    print(df_test.head())