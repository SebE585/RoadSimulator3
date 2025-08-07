import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simuler un signal heading bruité (exemple sinusoidal + bruit)
np.random.seed(42)
n_points = 500
t = np.arange(n_points)
heading_raw = (np.sin(0.02 * t) * 180 / np.pi + 180) % 360  # signal sinusoidal en degrés
noise = np.random.normal(0, 10, n_points)  # bruit gaussien
heading_noisy = (heading_raw + noise) % 360

def smooth_heading(heading_series, window_size=21):
    rad = np.deg2rad(heading_series)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)
    sin_smooth = pd.Series(sin_vals).rolling(window=window_size, center=True, min_periods=1).mean()
    cos_smooth = pd.Series(cos_vals).rolling(window=window_size, center=True, min_periods=1).mean()
    smooth_rad = np.arctan2(sin_smooth, cos_smooth)
    smooth_deg = np.rad2deg(smooth_rad) % 360
    return smooth_deg

# Appliquer les lissages
heading_smooth_21 = smooth_heading(heading_noisy, 21)
heading_smooth_31 = smooth_heading(heading_noisy, 31)
heading_smooth_41 = smooth_heading(heading_noisy, 41)

plt.figure(figsize=(14,7))
plt.plot(t, heading_noisy, label='Heading brut', color='gray', alpha=0.5)
plt.plot(t, heading_smooth_21, label='Lissage fenêtre 21')
plt.plot(t, heading_smooth_31, label='Lissage fenêtre 31')
plt.plot(t, heading_smooth_41, label='Lissage fenêtre 41')
plt.xlabel('Index')
plt.ylabel('Heading (degrés)')
plt.title('Comparaison du lissage du heading selon la taille de la fenêtre')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
