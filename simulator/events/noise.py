import numpy as np
import logging

logger = logging.getLogger(__name__)


def inject_inertial_noise(df, noise_params, seed=None):
    """Injecte du bruit inertiel (acc) et gyroscopique (gyro) sur les points sans événement.

    Corrections v1.0:
      - Dtypes stabilisés en float32 pour éviter les FutureWarning pandas lors des .loc +=
      - Masques convertis via .to_numpy() pour aligner l'indexation numpy/pandas
      - RNG local via default_rng (option seed) pour ne pas polluer l'état global
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # S'assurer que les colonnes acc_x, acc_y, acc_z existent (et en float32)
    for axis in ("acc_x", "acc_y", "acc_z"):
        if axis not in df.columns:
            df[axis] = 0.0
        df[axis] = df[axis].astype("float32", copy=False)

    std_acc_x = float(noise_params.get("acc_std", 0.3))
    std_acc_y = float(noise_params.get("acc_std", 0.3))
    std_acc_z = float(noise_params.get("acc_std", 0.3))

    n = len(df)
    t = np.linspace(0, n / 10.0, n, dtype=np.float32)  # 10 Hz

    # Bruit blanc + ondulations basse fréquence
    noise_x = rng.normal(0.0, std_acc_x, size=n).astype(np.float32)
    noise_y = rng.normal(0.0, std_acc_y, size=n).astype(np.float32)
    noise_z = rng.normal(0.0, std_acc_z, size=n).astype(np.float32)

    osc_acc_x = (0.08 * np.sin(2 * np.pi * 0.1 * t)).astype(np.float32)
    osc_acc_y = (0.06 * np.cos(2 * np.pi * 0.07 * t)).astype(np.float32)
    osc_acc_z = (0.05 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)

    noise_x += osc_acc_x
    noise_y += osc_acc_y
    noise_z += osc_acc_z

    # Oscillations supplémentaires + marche aléatoire
    osc2_acc_x = (0.03 * np.sin(2 * np.pi * 0.4 * t)).astype(np.float32)
    osc2_acc_y = (0.02 * np.cos(2 * np.pi * 0.3 * t)).astype(np.float32)
    osc2_acc_z = (0.01 * np.sin(2 * np.pi * 0.15 * t)).astype(np.float32)

    walk_x = np.cumsum(rng.normal(0.0, std_acc_x / 50.0, size=n)).astype(np.float32)
    walk_y = np.cumsum(rng.normal(0.0, std_acc_y / 50.0, size=n)).astype(np.float32)
    walk_z = np.cumsum(rng.normal(0.0, std_acc_z / 50.0, size=n)).astype(np.float32)

    noise_x += (osc2_acc_x + walk_x)
    noise_y += (osc2_acc_y + walk_y)
    noise_z += (osc2_acc_z + walk_z)

    # Masque des points sans événement
    mask = df["event"].isna()
    if mask.sum() < 10:
        logger.warning("⚠️ Trop peu de points sans événement — bruit inertiel injecté globalement.")
        mask[:] = True
    idx = mask.to_numpy()

    # Appliquer en évitant les FutureWarning (cast explicite + assignation)
    df.loc[idx, "acc_x"] = (df.loc[idx, "acc_x"].astype("float32") + noise_x[idx]).astype("float32")
    df.loc[idx, "acc_y"] = (df.loc[idx, "acc_y"].astype("float32") + noise_y[idx]).astype("float32")
    df.loc[idx, "acc_z"] = (df.loc[idx, "acc_z"].astype("float32") + noise_z[idx]).astype("float32")

    # --- Bruit gyroscopique ---
    for axis in ("gyro_x", "gyro_y", "gyro_z"):
        if axis not in df.columns:
            df[axis] = 0.0
        df[axis] = df[axis].astype("float32", copy=False)

    std_gyro_x = float(noise_params.get("gyro_std", 0.15))
    std_gyro_y = float(noise_params.get("gyro_std", 0.15))
    std_gyro_z = float(noise_params.get("gyro_std", 0.15))

    # Composantes gyro: drift lent + oscillations + bruit blanc + biais dérivant
    drift_x = np.cumsum(rng.normal(0.0, std_gyro_x / 50.0, size=n)).astype(np.float32)
    drift_y = np.cumsum(rng.normal(0.0, std_gyro_y / 50.0, size=n)).astype(np.float32)
    drift_z = np.cumsum(rng.normal(0.0, std_gyro_z / 50.0, size=n)).astype(np.float32)

    osc_x = (0.01 * np.sin(2 * np.pi * 0.2 * t)).astype(np.float32)
    osc_y = (0.01 * np.cos(2 * np.pi * 0.1 * t)).astype(np.float32)
    osc_z = (0.01 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)

    osc2_x = (0.005 * np.cos(2 * np.pi * 0.3 * t)).astype(np.float32)
    osc2_y = (0.005 * np.sin(2 * np.pi * 0.15 * t)).astype(np.float32)
    osc2_z = (0.005 * np.cos(2 * np.pi * 0.07 * t)).astype(np.float32)

    drift_bias = (0.001 * t).astype(np.float32)  # biais croissant lentement

    noise_gyro_x = (drift_x + osc_x + osc2_x + drift_bias + rng.normal(0.0, std_gyro_x, size=n)).astype(np.float32)
    noise_gyro_y = (drift_y + osc_y + osc2_y + drift_bias + rng.normal(0.0, std_gyro_y, size=n)).astype(np.float32)
    noise_gyro_z = (drift_z + osc_z + osc2_z + drift_bias + rng.normal(0.0, std_gyro_z, size=n)).astype(np.float32)

    if mask.sum() == 0:
        logger.warning("Aucun point sans événement (event == NaN), bruit inertiel injecté sur tous les points.")
        mask[:] = True
    idx = mask.to_numpy()

    df.loc[idx, "gyro_x"] = (df.loc[idx, "gyro_x"].astype("float32") + noise_gyro_x[idx]).astype("float32")
    df.loc[idx, "gyro_y"] = (df.loc[idx, "gyro_y"].astype("float32") + noise_gyro_y[idx]).astype("float32")
    df.loc[idx, "gyro_z"] = (df.loc[idx, "gyro_z"].astype("float32") + noise_gyro_z[idx]).astype("float32")

    logger.info("Bruit inertiel injecté sur acc_x, acc_y, acc_z.")
    logger.info("Bruit gyroscopique injecté sur gyro_x, gyro_y, gyro_z.")

    # 🛡️ Sécurité : forcer timestamps croissants si corrompus
    if "timestamp" in df.columns and not df["timestamp"].is_monotonic_increasing:
        logger.warning("⛔ inject_inertial_noise() a généré des timestamps non croissants ! Tri forcé.")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df
