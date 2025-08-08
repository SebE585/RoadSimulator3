import numpy as np
import logging
from simulator.events.generation import generate_opening_door
from geopy.distance import geodesic
from core.decorators import deprecated

logger = logging.getLogger(__name__)

def apply_progressive_acceleration_after_stop_wait(df, hz=10, target_speed_kmh=30, duration_s=5):
    """
    Ajoute une accélération progressive après les événements 'stop' ou 'wait' pour atteindre target_speed_kmh.

    Args:
        df (pd.DataFrame): DataFrame avec colonne 'event' et 'speed'.
        hz (int): fréquence d’échantillonnage (Hz).
        target_speed_kmh (float): vitesse cible à atteindre après stop/wait (km/h).
        duration_s (float): durée sur laquelle faire la rampe d’accélération (secondes).

    Returns:
        pd.DataFrame: DataFrame avec vitesse modifiée sur la rampe.
    """
    df = df.copy()
    n_points = int(duration_s * hz)
    speed_target_m_s = target_speed_kmh / 3.6  # conversion en m/s
    accel_profile = np.linspace(0, speed_target_m_s, n_points)

    stop_wait_indices = df.index[(df['event'] == 'stop') | (df['event'] == 'wait')].tolist()

    if not stop_wait_indices:
        logger.info("[INFO] Aucun événement 'stop' ou 'wait' détecté pour appliquer accélération progressive.")
        return df

    # Ne garder que le début de chaque séquence de stop ou wait
    stop_wait_indices = [i for i in stop_wait_indices if i == 0 or df['event'].iloc[i-1] != df['event'].iloc[i]]

    for idx in stop_wait_indices:
        start_idx = idx + 1
        end_idx = min(start_idx + n_points, len(df))
        length = end_idx - start_idx
        if length <= 0:
            continue
        # Appliquer la rampe d'accélération progressive sur speed (en km/h)
        df.loc[start_idx:end_idx-1, 'speed'] = accel_profile[:length] * 3.6
        # Fixer acc_x à une valeur positive indicative (optionnel, à ajuster selon besoin)
        df.loc[start_idx:end_idx-1, 'acc_x'] = (speed_target_m_s / duration_s) * 3.6  # approx.

    return df

@deprecated
def apply_stop_wait_at_positions(df, events_df, window_m=20):
    # Ne pas injecter deux stops trop proches : espacement minimal de 800 points (≈80s à 10 Hz)
    df = df.copy()
    matched_indices = []
    min_spacing_pts = 800

    for _, row in events_df.iterrows():
        lat0, lon0, label = row["lat"], row["lon"], row["event"]

        distances = df.apply(lambda r: geodesic((lat0, lon0), (r["lat"], r["lon"])).meters, axis=1)
        idx = distances.idxmin()
        if distances[idx] <= window_m:
            if matched_indices and abs(idx - matched_indices[-1]) < min_spacing_pts:
                continue  # trop proche du précédent
            if idx in matched_indices:
                continue
            matched_indices.append(idx)
            duration_pts = 1200 if label == "stop" else 300
            df.loc[idx:idx+duration_pts, "event"] = label

            # Exemple de signature inertielle pour un stop/wait
            # (Peut être remplacé par des signatures réalistes)
            df.loc[idx:idx+duration_pts, "speed"] = 0.0
            df.loc[idx:idx+duration_pts, "acc_x"] = 0.0 if label == "stop" else -0.5
            df.loc[idx:idx+duration_pts, "acc_z"] = 9.81
            df.loc[idx:idx+duration_pts, "gyro_x"] = 0.0
            df.loc[idx:idx+duration_pts, "gyro_y"] = 0.0
            df.loc[idx:idx+duration_pts, "gyro_z"] = 0.0

            # Injection automatique d'une ouverture de porte autour du stop/wait
            if label in ["stop", "wait"]:
                config_local = {
                    "events": {
                        "ouverture_porte": {
                            "enabled": True,
                            "around_event": label,
                            "window_before_s": 1.0,
                            "window_after_s": 2.0,
                            "intensity": 1.0,
                            "probability": 1.0,
                            "duration_pts": 30,
                            "max_events": 1,
                            "max_attempts": 3
                        }
                    }
                }
                df = generate_opening_door(df, config=config_local)

    logger.info(f"[stop_wait] {len(matched_indices)} événements uniques 'stop/wait' injectés à partir de positions fournies.")
    return df