import pandas as pd

"""
Détecteurs inertiels pour les événements simulés dans RoadSimulator3.
Chaque fonction analyse un DataFrame à 10 Hz pour repérer des signatures d'événements spécifiques :
dos d'âne, freinage, accélération, trottoir, nid de poule, arrêt, attente, ouverture de porte, etc.
"""
G = 9.81
def detect_dos_dane(df, acc_z_thresh=4.0, gyro_thresh=6.0, window_pts=8, refractory=20, config=None):
    """
    Détecte les dos d'âne en analysant les pics d'accélération verticale (acc_z) 
    et les variations de vitesse angulaire autour de l'axe z (gyro_z).

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonnes acc_z, gyro_z.
        acc_z_thresh (float): Seuil de détection en m/s² pour acc_z.
        gyro_thresh (float): Variation minimale gyro_z pour valider l’événement.
        window_pts (int): Taille de la fenêtre glissante.
        refractory (int): Délai de récupération après détection.
        config (dict, optional): Configuration inertielle (remplace les seuils par défaut).

    Returns:
        Tuple[bool, List[int]]: Indicateur de détection, indices détectés.
    """
    if config is not None:
        ev_cfg = config.get("dos_dane", {})
        acc_z_thresh = ev_cfg.get("acc_z_thresh", acc_z_thresh)
        gyro_thresh = ev_cfg.get("gyro_thresh", gyro_thresh)
        window_pts = ev_cfg.get("window_pts", window_pts)
        refractory = ev_cfg.get("refractory", refractory)

    acc_z = pd.to_numeric(df["acc_z"], errors='coerce').fillna(0.0)
    gyro_z = pd.to_numeric(df["gyro_z"], errors='coerce').fillna(0.0)
    detected_indices = []
    i = 0
    while i < len(df) - window_pts:
        z = acc_z[i:i + window_pts].values
        g = gyro_z[i:i + window_pts].values

        acc_condition = (z.max() >= acc_z_thresh) and (z.min() <= -acc_z_thresh)
        gyro_condition = g.max() - g.min() >= gyro_thresh

        if acc_condition and gyro_condition:
            print(f"[DEBUG DOS D'ANE] acc_z max: {z.max():.2f}, min: {z.min():.2f}, gyro_z Δ: {g.max() - g.min():.2f}")
            print(f"[DEBUG WINDOW] acc_z: {z.tolist()}, gyro_z: {g.tolist()}")
            detected_indices.append(i)
            i += refractory
            continue
        i += 1

    if detected_indices:
        print(f"[DETECT DOS D'ANE] ✅ {len(detected_indices)} dos d’âne détectés")
        return True, detected_indices
    print("[DETECT DOS D'ANE] ❌ Aucun dos d’âne détecté (acc_z + gyro_z).")
    return False, []



def detect_nid_de_poule(df, acc_z_thresh=5.0, gyro_thresh=3.0, window_pts=8, refractory=20, config=None):
    """
    Détecte les nids de poule par analyse des variations d'accélération verticale (acc_z)
    et de la vitesse angulaire (généralement gyro_x).

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonnes acc_z, gyro_x (ou autre axe).
        acc_z_thresh (float): Seuil de détection en m/s² pour acc_z.
        gyro_thresh (float): Variation minimale gyro pour valider l’événement.
        window_pts (int): Taille de la fenêtre glissante.
        refractory (int): Délai de récupération après détection.
        config (dict, optional): Configuration inertielle (remplace les seuils par défaut, axe gyro).

    Returns:
        Tuple[bool, List[int]]: Indicateur de détection, indices détectés.
    """
    if config is not None:
        ev_cfg = config.get("nid_de_poule", {})
        acc_z_thresh = ev_cfg.get("acc_z_thresh", acc_z_thresh)
        gyro_thresh = ev_cfg.get("gyro_thresh", gyro_thresh)
        window_pts = ev_cfg.get("window_pts", window_pts)
        refractory = ev_cfg.get("refractory", refractory)

    acc_z = pd.to_numeric(df["acc_z"], errors='coerce').fillna(0.0)
    gyro_col = "gyro_x"
    if config is not None:
        gyro_col = config.get("nid_de_poule", {}).get("gyro_axis", "gyro_x")
    gyro = pd.to_numeric(df[gyro_col], errors='coerce').fillna(0.0)
    detected_indices = []
    i = 0
    while i < len(df) - window_pts:
        z = acc_z[i:i + window_pts].values
        g = gyro[i:i + window_pts].values
        acc_condition = (z.min() <= -acc_z_thresh) and (z.max() >= acc_z_thresh)
        gyro_condition = g.max() - g.min() >= gyro_thresh
        if acc_condition and gyro_condition:
            detected_indices.append(i)
            i += refractory
            continue
        i += 1
    if detected_indices:
        print(f"[DETECT NID DE POULE] ✅ {len(detected_indices)} nids de poule détectés")
    else:
        print("[DETECT NID DE POULE] ❌ Aucun nid de poule détecté")
    return bool(detected_indices), detected_indices

def detect_trottoir(df, acc_z_thresh=6.0, gyro_thresh=2.0, window_pts=6, refractory=20, config=None):
    """
    Détecte les chocs de trottoir via les pics d'accélération verticale (acc_z)
    et les variations de vitesse angulaire sur plusieurs axes.

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonnes acc_z et gyro_x/y/z.
        acc_z_thresh (float): Seuil de détection en m/s² pour acc_z.
        gyro_thresh (float): Variation minimale gyro pour valider l’événement.
        window_pts (int): Taille de la fenêtre glissante.
        refractory (int): Délai de récupération après détection.
        config (dict, optional): Configuration inertielle (remplace les seuils/axes par défaut).

    Returns:
        Tuple[bool, List[int]]: Indicateur de détection, indices détectés.
    """
    if config is not None:
        ev_cfg = config.get("trottoir", {})
        acc_z_thresh = ev_cfg.get("acc_z_thresh", acc_z_thresh)
        gyro_thresh = ev_cfg.get("gyro_thresh", gyro_thresh)
        window_pts = ev_cfg.get("window_pts", window_pts)
        refractory = ev_cfg.get("refractory", refractory)
        gyro_axes = ev_cfg.get("gyro_axes_used", ["x", "y", "z"])
    else:
        gyro_axes = ["x", "y", "z"]

    acc_z = pd.to_numeric(df["acc_z"], errors='coerce').fillna(0.0)
    gyro_data = {
        axis: pd.to_numeric(df[f"gyro_{axis}"], errors='coerce').fillna(0.0)
        for axis in gyro_axes
    }

    detected_indices = []
    i = 0
    while i < len(df) - window_pts:
        z = acc_z[i:i + window_pts].values
        acc_condition = z.max() >= acc_z_thresh
        gyro_variations = [gyro[i:i + window_pts].max() - gyro[i:i + window_pts].min() for gyro in gyro_data.values()]
        gyro_condition = max(gyro_variations) >= gyro_thresh

        if acc_condition and gyro_condition:
            detected_indices.append(i)
            i += refractory
            continue
        i += 1
    if detected_indices:
        print(f"[DETECT TROTTOIR] ✅ {len(detected_indices)} chocs trottoir détectés")
    else:
        print("[DETECT TROTTOIR] ❌ Aucun choc trottoir détecté")
    return bool(detected_indices), detected_indices

def detect_freinage(df, acc_x_thresh=-3.0, window_pts=6, refractory=20, config=None):
    """
    Détecte les épisodes de freinage en repérant une accélération longitudinale (acc_x) négative prolongée.

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonne acc_x.
        acc_x_thresh (float): Seuil de détection (accélération négative) en m/s².
        window_pts (int): Taille de la fenêtre glissante.
        refractory (int): Délai de récupération après détection.
        config (dict, optional): Configuration inertielle (remplace les seuils par défaut).

    Returns:
        Tuple[bool, List[int]]: Indicateur de détection, indices détectés.
    """
    if config is not None:
        ev_cfg = config.get("freinage", {})
        acc_x_thresh = ev_cfg.get("acc_x_thresh", acc_x_thresh)
        window_pts = ev_cfg.get("window_pts", window_pts)
        refractory = ev_cfg.get("refractory", refractory)

    acc_x = pd.to_numeric(df["acc_x"], errors='coerce').fillna(0.0)
    detected_indices = []
    i = 0
    while i < len(df) - window_pts:
        x = acc_x[i:i + window_pts].values
        if (x < acc_x_thresh).all():
            detected_indices.append(i)
            i += refractory
            continue
        i += 1
    if detected_indices:
        print(f"[DETECT FREINAGE] ✅ {len(detected_indices)} freinages détectés")
    else:
        print("[DETECT FREINAGE] ❌ Aucun freinage détecté")
    return bool(detected_indices), detected_indices

def detect_acceleration(df, acc_x_thresh=2.5, window_pts=6, refractory=20, config=None):
    """
    Détecte les accélérations franches via une accélération longitudinale (acc_x) positive prolongée.

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonne acc_x.
        acc_x_thresh (float): Seuil de détection (accélération positive) en m/s².
        window_pts (int): Taille de la fenêtre glissante.
        refractory (int): Délai de récupération après détection.
        config (dict, optional): Configuration inertielle (remplace les seuils par défaut).

    Returns:
        Tuple[bool, List[int]]: Indicateur de détection, indices détectés.
    """
    if config is not None:
        ev_cfg = config.get("acceleration", {})
        acc_x_thresh = ev_cfg.get("acc_x_thresh", acc_x_thresh)
        window_pts = ev_cfg.get("window_pts", window_pts)
        refractory = ev_cfg.get("refractory", refractory)

    acc_x = pd.to_numeric(df["acc_x"], errors='coerce').fillna(0.0)
    detected_indices = []
    i = 0
    while i < len(df) - window_pts:
        x = acc_x[i:i + window_pts].values
        if (x > acc_x_thresh).all():
            detected_indices.append(i)
            i += refractory
            continue
        i += 1
    if detected_indices:
        print(f"[DETECT ACCELERATION] ✅ {len(detected_indices)} accélérations détectées")
    else:
        print("[DETECT ACCELERATION] ❌ Aucune accélération détectée")
    return bool(detected_indices), detected_indices


# Détection globale de tous les événements
def detect_all_events(df, config=None):
    summary = {}
    if "event" in df.columns:
        for ev in ["dos_dane", "nid_de_poule", "trottoir", "freinage", "acceleration", "stop", "wait", "ouverture", "initial_acceleration", "final_deceleration"]:
            summary[ev] = df["event"].eq(ev).any()
        return summary

    from simulator.s import (
        detect_dos_dane, detect_nid_de_poule, detect_trottoir,
        detect_freinage, detect_acceleration,
        detect_stop, detect_wait, detect_ouverture_porte,
        detect_initial_acceleration, detect_final_deceleration
    )
    summary["dos_dane"] = detect_dos_dane(df, config=config)[0]
    summary["nid_de_poule"] = detect_nid_de_poule(df, config=config)[0]
    summary["trottoir"] = detect_trottoir(df, config=config)[0]
    summary["freinage"] = detect_freinage(df, config=config)[0]
    summary["acceleration"] = detect_acceleration(df, config=config)[0]
    summary["stop"] = detect_stop(df, config=config)[0]
    summary["wait"] = detect_wait(df, config=config)[0]
    summary["ouverture"] = detect_ouverture_porte(df, config=config)[0]
    summary["initial_acceleration"] = detect_initial_acceleration(df, config=config)[0]
    summary["final_deceleration"] = detect_final_deceleration(df, config=config)[0]
    return summary


# Détection de l'arrêt prolongé (stop)
def detect_stop(df, duration_threshold_s=120, hz=10, config=None):
    """
    Détecte les arrêts prolongés (moteur coupé) en analysant les périodes où la vitesse est nulle.

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonne 'speed'.
        duration_threshold_s (float): Durée minimale de l’arrêt en secondes.
        hz (int): Fréquence d’échantillonnage (Hz).
        config (dict, optional): Paramètres personnalisés (clé 'stop').

    Returns:
        Tuple[bool, List[int]]: Détection binaire, indices de début d’arrêt détectés.
    """
    if config is not None:
        ev_cfg = config.get("stop", {})
        duration_threshold_s = ev_cfg.get("min_duration_s", duration_threshold_s)
    if "speed" not in df.columns:
        return False, []
    is_stopped = df["speed"].fillna(0.0) < 0.1
    min_points = duration_threshold_s * hz
    detected = []
    i = 0
    while i < len(df):
        if is_stopped[i]:
            start = i
            while i < len(df) and is_stopped[i]:
                i += 1
            if i - start >= min_points:
                detected.append(start)
        else:
            i += 1
    if detected:
        print(f"[DETECT STOP] ✅ {len(detected)} arrêts prolongés détectés")
    else:
        print("[DETECT STOP] ❌ Aucun arrêt prolongé détecté")
    return bool(detected), detected

# Détection de l'attente (wait)
def detect_wait(df, min_duration_s=30, max_duration_s=120, hz=10, config=None):
    """
    Détecte les phases d'attente moteur allumé (30s < durée < 2min) en analysant les périodes à vitesse nulle.

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonne 'speed'.
        min_duration_s (float): Seuil bas de durée en secondes.
        max_duration_s (float): Seuil haut de durée en secondes.
        hz (int): Fréquence d’échantillonnage.
        config (dict, optional): Paramètres personnalisés (clé 'wait').

    Returns:
        Tuple[bool, List[int]]: Détection binaire, indices de début d’attente détectés.
    """
    if config is not None:
        ev_cfg = config.get("wait", {})
        min_duration_s = ev_cfg.get("min_duration_s", min_duration_s)
        max_duration_s = ev_cfg.get("max_duration_s", max_duration_s)
    if "speed" not in df.columns:
        return False, []
    is_stopped = df["speed"].fillna(0.0) < 0.1
    min_pts, max_pts = min_duration_s * hz, max_duration_s * hz
    detected = []
    i = 0
    while i < len(df):
        if is_stopped[i]:
            start = i
            while i < len(df) and is_stopped[i]:
                i += 1
            dur = i - start
            if min_pts <= dur < max_pts:
                detected.append(start)
        else:
            i += 1
    if detected:
        print(f"[DETECT WAIT] ✅ {len(detected)} attentes détectées")
    else:
        print("[DETECT WAIT] ❌ Aucune attente détectée")
    return bool(detected), detected

# Détection de l'ouverture de porte (ouverture)
def detect_ouverture_porte(df, gyro_z_thresh=50.0, refractory=20, window_pts=6, config=None):
    """
    Détecte les ouvertures de porte par forte variation gyroscopique sur l'axe Z.

    Args:
        df (pd.DataFrame): Données simulées à 10 Hz avec colonne 'gyro_z'.
        gyro_z_thresh (float): Seuil de variation pour détection.
        refractory (int): Fenêtre d'attente après détection.
        window_pts (int): Taille de la fenêtre d’analyse.
        config (dict, optional): Paramètres personnalisés (clé 'ouverture').

    Returns:
        Tuple[bool, List[int]]: Détection binaire, indices détectés.
    """
    if config is not None:
        ev_cfg = config.get("ouverture", {})
        gyro_z_thresh = ev_cfg.get("gyro_z_thresh", gyro_z_thresh)
        window_pts = ev_cfg.get("window_pts", window_pts)
        refractory = ev_cfg.get("refractory", refractory)
    gyro_z = pd.to_numeric(df["gyro_z"], errors='coerce').fillna(0.0)
    detected = []
    i = 0
    while i < len(df) - window_pts:
        g = gyro_z[i:i + window_pts].values
        if g.max() - g.min() >= gyro_z_thresh:
            detected.append(i)
            i += refractory
            continue
        i += 1
    if detected:
        print(f"[DETECT OUVERTURE PORTE] ✅ {len(detected)} ouvertures de porte détectées")
    else:
        print("[DETECT OUVERTURE PORTE] ❌ Aucune ouverture de porte détectée")
    return bool(detected), detected


# Détection de l'accélération initiale
def detect_initial_acceleration(df, v_min_kmh=3.0, v_max_kmh=50.0, acc_x_min=1.0, hz=10, config=None):
    """
    Détecte l’accélération initiale dans les premières secondes du trajet.

    Args:
        df (pd.DataFrame): Données avec colonnes 'speed' et 'acc_x'.
        v_min_kmh (float): Vitesse minimale en km/h.
        v_max_kmh (float): Vitesse maximale en km/h.
        acc_x_min (float): Seuil minimal d’accélération longitudinale.
        hz (int): Fréquence d’échantillonnage.
        config (dict, optional): Paramètres personnalisés.

    Returns:
        Tuple[bool, List[int]]: Détection binaire, index du début.
    """
    if config is not None:
        ev_cfg = config.get("initial_acceleration", {})
        v_min_kmh = ev_cfg.get("v_min_kmh", v_min_kmh)
        v_max_kmh = ev_cfg.get("v_max_kmh", v_max_kmh)
        acc_x_min = ev_cfg.get("acc_x_min", acc_x_min)

    speed = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    acc_x = pd.to_numeric(df["acc_x"], errors="coerce").fillna(0.0)
    detected = []

    for i in range(min(300, len(df) - 5)):  # premières 30 secondes
        if v_min_kmh < speed[i] < v_max_kmh and acc_x[i] > acc_x_min:
            detected.append(i)
            break

    if detected:
        print(f"[DETECT INITIAL ACCEL] ✅ Accélération initiale détectée à {detected[0]}")
    else:
        print("[DETECT INITIAL ACCEL] ❌ Aucune accélération initiale détectée")
    return bool(detected), detected

# Détection de la décélération finale
def detect_final_deceleration(df, v_min_kmh=3.0, v_max_kmh=50.0, acc_x_max=-1.0, hz=10, config=None):
    """
    Détecte la décélération finale dans les dernières secondes du trajet.

    Args:
        df (pd.DataFrame): Données avec colonnes 'speed' et 'acc_x'.
        v_min_kmh (float): Vitesse minimale en km/h.
        v_max_kmh (float): Vitesse maximale en km/h.
        acc_x_max (float): Seuil maximal d’accélération longitudinale (négatif).
        hz (int): Fréquence d’échantillonnage.
        config (dict, optional): Paramètres personnalisés.

    Returns:
        Tuple[bool, List[int]]: Détection binaire, index du début.
    """
    if config is not None:
        ev_cfg = config.get("final_deceleration", {})
        v_min_kmh = ev_cfg.get("v_min_kmh", v_min_kmh)
        v_max_kmh = ev_cfg.get("v_max_kmh", v_max_kmh)
        acc_x_max = ev_cfg.get("acc_x_max", acc_x_max)

    speed = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    acc_x = pd.to_numeric(df["acc_x"], errors="coerce").fillna(0.0)
    detected = []

    for i in range(len(df) - 1, max(0, len(df) - 300), -1):  # dernières 30 secondes
        if v_min_kmh < speed[i] < v_max_kmh and acc_x[i] < acc_x_max:
            detected.append(i)
            break

    if detected:
        print(f"[DETECT FINAL DECEL] ✅ Décélération finale détectée à {detected[0]}")
    else:
        print("[DETECT FINAL DECEL] ❌ Aucune décélération finale détectée")
    return bool(detected), detected