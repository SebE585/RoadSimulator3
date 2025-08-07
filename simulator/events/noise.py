import numpy as np
import logging

logger = logging.getLogger(__name__)

def inject_inertial_noise(df, noise_params, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # S'assurer que les colonnes acc_x, acc_y, acc_z existent
    for axis in ['acc_x', 'acc_y', 'acc_z']:
        if axis not in df.columns:
            df[axis] = 0.0

    std_acc_x = noise_params.get('acc_std', 0.3)
    std_acc_y = noise_params.get('acc_std', 0.3)
    std_acc_z = noise_params.get('acc_std', 0.3)

    noise_x = np.random.normal(0, std_acc_x, size=len(df))
    noise_y = np.random.normal(0, std_acc_y, size=len(df))
    noise_z = np.random.normal(0, std_acc_z, size=len(df))

    # Ondulation inertielle basse fréquence
    t = np.linspace(0, len(df) / 10, len(df))  # 10 Hz
    osc_acc_x = 0.08 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz, amplitude augmentée
    osc_acc_y = 0.06 * np.cos(2 * np.pi * 0.07 * t)  # amplitude augmentée
    osc_acc_z = 0.05 * np.sin(2 * np.pi * 0.05 * t)  # amplitude augmentée
    noise_x += osc_acc_x
    noise_y += osc_acc_y
    noise_z += osc_acc_z
    # Ajout oscillations supplémentaires et marche aléatoire
    osc2_acc_x = 0.03 * np.sin(2 * np.pi * 0.4 * t)
    osc2_acc_y = 0.02 * np.cos(2 * np.pi * 0.3 * t)
    osc2_acc_z = 0.01 * np.sin(2 * np.pi * 0.15 * t)
    walk_x = np.cumsum(np.random.normal(0, std_acc_x / 50, size=len(df)))  # diviseur diminué pour effet amplifié
    walk_y = np.cumsum(np.random.normal(0, std_acc_y / 50, size=len(df)))
    walk_z = np.cumsum(np.random.normal(0, std_acc_z / 50, size=len(df)))
    noise_x += osc2_acc_x + walk_x
    noise_y += osc2_acc_y + walk_y
    noise_z += osc2_acc_z + walk_z

    mask = df['event'].isna()
    if mask.sum() < 10:
        logger.warning("⚠️ Trop peu de points sans événement — bruit inertiel injecté globalement.")
        mask[:] = True
    df.loc[mask, 'acc_x'] += noise_x[mask]
    df.loc[mask, 'acc_y'] += noise_y[mask]
    df.loc[mask, 'acc_z'] += noise_z[mask]

    # Lissage post-injection (fenêtre glissante)
    # window = 5
    # df['acc_x'] = df['acc_x'].rolling(window, center=True, min_periods=1).mean()
    # df['acc_y'] = df['acc_y'].rolling(window, center=True, min_periods=1).mean()
    # df['acc_z'] = df['acc_z'].rolling(window, center=True, min_periods=1).mean()

    std_gyro_x = noise_params.get('gyro_std', 0.15)
    std_gyro_y = noise_params.get('gyro_std', 0.15)
    std_gyro_z = noise_params.get('gyro_std', 0.15)

    t = np.linspace(0, len(df) / 10, len(df))  # en secondes (10 Hz)
    drift_x = np.cumsum(np.random.normal(0, std_gyro_x / 50, size=len(df)))  # drift lent
    drift_y = np.cumsum(np.random.normal(0, std_gyro_y / 50, size=len(df)))
    drift_z = np.cumsum(np.random.normal(0, std_gyro_z / 50, size=len(df)))
    osc_x = 0.01 * np.sin(2 * np.pi * 0.2 * t)  # oscillation 0.2 Hz
    osc_y = 0.01 * np.cos(2 * np.pi * 0.1 * t)
    osc_z = 0.01 * np.sin(2 * np.pi * 0.05 * t)
    # Ajout oscillations supplémentaires et biais dérivant
    osc2_x = 0.005 * np.cos(2 * np.pi * 0.3 * t)
    osc2_y = 0.005 * np.sin(2 * np.pi * 0.15 * t)
    osc2_z = 0.005 * np.cos(2 * np.pi * 0.07 * t)
    drift_bias = 0.001 * t  # Biais gyroscopique croissant lentement

    noise_gyro_x = drift_x + osc_x + osc2_x + drift_bias + np.random.normal(0, std_gyro_x, size=len(df))
    noise_gyro_y = drift_y + osc_y + osc2_y + drift_bias + np.random.normal(0, std_gyro_y, size=len(df))
    noise_gyro_z = drift_z + osc_z + osc2_z + drift_bias + np.random.normal(0, std_gyro_z, size=len(df))

    for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
        if axis not in df.columns:
            df[axis] = 0.0

    if mask.sum() == 0:
        logger.warning("Aucun point sans événement (event == NaN), bruit inertiel injecté sur tous les points.")
        mask[:] = True
    df.loc[mask, 'gyro_x'] += noise_gyro_x[mask.to_numpy()]
    df.loc[mask, 'gyro_y'] += noise_gyro_y[mask.to_numpy()]
    df.loc[mask, 'gyro_z'] += noise_gyro_z[mask.to_numpy()]

    logger.info("Bruit inertiel injecté sur acc_x, acc_y, acc_z.")
    logger.info("Bruit gyroscopique injecté sur gyro_x, gyro_y, gyro_z.")

    # 🛡️ Sécurité : forcer timestamps croissants si corrompus
    if not df['timestamp'].is_monotonic_increasing:
        logger.warning("⛔ inject_inertial_noise() a généré des timestamps non croissants ! Tri forcé.")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df
