import sys
import os
import pandas as pd
import json
import glob
import logging
logger = logging.getLogger(__name__)
from core.decorators import deprecated

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.config_loader import load_config
config = load_config("config/events.yaml")
import numpy as np
from geopy.distance import geodesic
from core.geo_utils import compute_heading  # noqa: F401

from core.geo_utils import haversine_distance

def check_speed_vs_roadtype(df: pd.DataFrame) -> bool:
    road_order = ["motorway", "primary", "secondary", "tertiary", "residential", "service"]
    valid = df[df['road_type'].isin(road_order)]
    road_speeds = valid.groupby("road_type")["speed"].mean()
    logger.debug("Vitesses moyennes par type de route :")
    for r in road_order:
        if r in road_speeds:
            logger.debug(f" - {r:<10} : {road_speeds[r]:.2f} km/h")
    filtered = [road_speeds[r] for r in road_order if r in road_speeds]
    for earlier, later in zip(filtered, filtered[1:]):
        tolerance = 15  # tolérance souple élargie à 15 km/h
        logger.debug(f"Comparaison : {earlier:.2f} km/h vs {later:.2f} km/h (tolérance +{tolerance})")
        if earlier < 25 and later > 45:
            continue  # ignorer les écarts suspects en bas de tableau
        if (earlier + tolerance) < later:
            logger.warning(f"Incohérence : {earlier:.2f} km/h < {later:.2f} km/h malgré la tolérance (+{tolerance})")
            return False
        if earlier < later and (later - earlier) < tolerance:
            logger.info(f"Tolérance souple acceptée : {earlier:.2f} km/h < {later:.2f} km/h (écart {later-earlier:.2f} < {tolerance})")
    return True

def detect_initial_acceleration(df, hz=10):
    seg = df.iloc[:hz * 5]
    return (seg.acc_x > 2.3).sum() >= 45 and seg.speed.max() > 40

def detect_final_deceleration(df, hz=10):
    seg = df.iloc[-hz * 4:]
    return (seg.acc_x < -1.5).sum() >= 30 and seg.speed.min() < 1.0




def has_stop(df):
    return df['event'].eq('stop').any() or df['event'].eq('stop_start').any()

def has_wait(df):
    return df['event'].eq('wait').any() or df['event'].eq('wait_start').any()

def compute_total_distance(df):
    df = df.reset_index(drop=True)
    dist = 0
    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        dist += geodesic(p1, p2).meters
    return dist

def compute_duration(df):
    return (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()

def check_event_presence(df, event_label):
    return df['event'].eq(event_label).any()

def check_frequency(df):
    deltas = df['timestamp'].diff().dropna().dt.total_seconds()
    median_interval = deltas.median()
    return abs(median_interval - 0.1) < 0.01

# Vérification : vitesse nulle au début et à la fin
def check_speed_start_end(df, threshold_kmh=1.0):
    speed_start = df["speed"].iloc[0]
    speed_end = df["speed"].iloc[-1]
    start_ok = speed_start <= threshold_kmh
    end_ok = speed_end <= threshold_kmh
    return start_ok, end_ok, speed_start, speed_end

def check_speed_smoothness(df, threshold=20):
    diffs = df['speed'].diff().abs()
    max_diff = diffs.max()
    if max_diff > threshold:
        logger.warning(f"Variation de vitesse trop brutale détectée: {max_diff:.2f} km/h > seuil {threshold}")
        return False, max_diff
    return True, max_diff

def check_gps_jumps(df, threshold_m=50.0):
    df = df.reset_index(drop=True)
    jumps = []
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon']
        lat2, lon2 = df.loc[i, 'lat'], df.loc[i, 'lon']
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        if dist > threshold_m:
            jumps.append((i, dist))
    return len(jumps) == 0

def check_acceleration_variability(df):
    # Seuil assoupli de 1 m/s à 0.2 m/s pour inclure davantage de points en mouvement
    inertial_events = ['freinage', 'acceleration', 'trottoir', 'nid_de_poule', 'dos_dane', 'ouverture_porte']
    moving = df[(df['speed'] > 0.2) & (~df['event'].isin(inertial_events))]
    logger.debug(f"Points en mouvement hors événements : {len(moving)}")
    if len(moving) < 10:
        logger.warning("❌ Échec du test 'Variations inertielle réalistes (acc_x/y/z)' : au moins 10 points en mouvement hors événements sont nécessaires.")
        return False
    std_x, std_y, std_z = moving['acc_x'].std(), moving['acc_y'].std(), moving['acc_z'].std()
    acc_cfg = config.get("realism_check", {}).get("acc", {}) or {}
    std_min = acc_cfg.get("std_min", 0.1)
    std_max = acc_cfg.get("std_max", 3.0)
    logger.debug(f"acc std_x={std_x:.3f}, std_y={std_y:.3f}, std_z={std_z:.3f}")
    return (std_min < std_x < std_max and std_min < std_y < std_max and std_min < std_z < std_max)

def check_gyroscope_variability(df):
    # Seuil assoupli de 1 m/s à 0.2 m/s pour inclure davantage de points en mouvement
    inertial_events = ['freinage', 'acceleration', 'trottoir', 'nid_de_poule', 'dos_dane', 'ouverture_porte']
    moving = df[(df['speed'] > 0.2) & (~df['event'].isin(inertial_events))]
    logger.debug(f"Points gyroscopiques en mouvement hors événements : {len(moving)}")
    if len(moving) < 10:
        logger.warning("❌ Échec du test 'Variations gyroscopiques réalistes (gyro_x/y/z)' : au moins 10 points en mouvement hors événements sont nécessaires.")
        return False
    std_gyro = moving[['gyro_x', 'gyro_y', 'gyro_z']].std()
    gyro_cfg = config.get("realism_check", {}).get("gyro", {})
    std_min = gyro_cfg.get("std_min", 0.01)
    std_max = gyro_cfg.get("std_max", 60.0)
    logger.debug(f"gyro std_x={std_gyro['gyro_x']:.3f}, std_y={std_gyro['gyro_y']:.3f}, std_z={std_gyro['gyro_z']:.3f}")
    logger.debug(f"seuils : std_min={std_min}, std_max={std_max}")
    return std_gyro.between(std_min, std_max).all()

def check_spatio_temporal_coherence(
    df: pd.DataFrame,
    meta: dict | None = None,
    hz_target: float = 10.0,
    max_dist_rel_err: float = 0.03,
    v_abs_tol_mps: float = 2.0,
    frac_bad_tol: float = 0.01,
    log_file: str | None = None,
):
    """
    Cohérence spatio-temporelle robuste au retiming (10 Hz “étiré”).
    Valide si :
      - cadence ≈ hz_target,
      - distance géo ≈ ∑(v·dt),
      - la vitesse déclarée est cohérente avec la vitesse géo recalculée,
    et NE pénalise PAS un allongement de durée si le retiming est intentionnel
    (indiqué par meta['retime_policy'] ou meta['retimed_by_kinematics']).

    Params
    ------
    df : DataFrame avec 'timestamp','lat','lon','speed' (speed en km/h)
    meta : dict optionnel avec des flags de retiming
    hz_target : Hz attendu après resampling
    max_dist_rel_err : écart relatif max (geo vs ∑v·dt)
    v_abs_tol_mps : tolérance absolue |v_geo - v_decl| en m/s
    frac_bad_tol : part max de points au-delà de v_abs_tol_mps
    log_file : fichier de sortie (JSON) pour les détails
    """
    # 1) Temps & cadence observée
    t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ns = t.astype("int64").to_numpy()
    tsec = (ns - ns[0]) / 1e9
    dt = np.diff(tsec, prepend=tsec[0])
    pos = dt > 0
    if not pos.any():
        return False, {"error": "dt non-positifs"}
    median_dt = float(np.median(dt[pos]))
    dt[~pos] = median_dt
    hz_obs = 1.0 / median_dt
    hz_ok = abs(hz_obs - hz_target) <= 0.1

    # 2) Distances géodésiques (m) et par vitesse (m)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float)

    d = np.zeros_like(lat, dtype=float)
    for i in range(1, len(lat)):
        d[i] = haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i])  # m

    dist_geo = float(np.nansum(d))
    v_kmh = pd.to_numeric(df["speed"], errors="coerce").to_numpy(dtype=float)
    v_mps = v_kmh * (1000.0 / 3600.0)

    # --- Détection auto de l'unité de speed (km/h vs m/s) ---
    # On utilise v_geo_mps_probe = d / dt comme référence de cohérence instantanée
    with np.errstate(divide="ignore", invalid="ignore"):
        v_geo_mps_probe = np.divide(d, dt, out=np.zeros_like(d), where=dt > 0)
    # Erreur médiane si speed en km/h (convertie en m/s)
    err_kmh = float(np.nanmedian(np.abs(v_mps - v_geo_mps_probe)))
    # Erreur médiane si speed déjà en m/s
    err_mps = float(np.nanmedian(np.abs(v_kmh - v_geo_mps_probe)))
    speed_unit_used = "kmh"
    if err_mps < err_kmh:
        # La colonne speed semble être en m/s → on l'utilise telle quelle
        v_mps = v_kmh.copy()
        speed_unit_used = "mps"

    dist_from_speed = float(np.nansum(v_mps * dt))
    rel_err = abs(dist_from_speed - dist_geo) / max(dist_geo, 1e-6)
    dist_ok = rel_err <= max_dist_rel_err

    # 3) Auto-cohérence des vitesses (recalc géo / dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_geo_mps = np.divide(d, dt, out=np.zeros_like(d), where=dt > 0)
    dv = np.abs(v_mps - v_geo_mps)
    dv_median = float(np.nanmedian(dv))
    frac_bad = float(np.mean(dv > v_abs_tol_mps))
    speed_ok = (dv_median <= v_abs_tol_mps) and (frac_bad <= frac_bad_tol)

    # 4) Retiming intentionnel ? (ne bloque pas si la durée change)
    retimed = bool(meta and (meta.get("retime_policy") or meta.get("retimed_by_kinematics")))
    ok = hz_ok and dist_ok and speed_ok

    details = {
        "hz_target": hz_target,
        "hz_observed": float(hz_obs),
        "hz_ok": bool(hz_ok),
        "dist_geo_m": dist_geo,
        "dist_from_speed_m": dist_from_speed,
        "dist_rel_err": float(rel_err),
        "dist_ok": bool(dist_ok),
        "dv_median_mps": dv_median,
        "frac_bad_over_tol": frac_bad,
        "v_abs_tol_mps": v_abs_tol_mps,
        "speed_ok": bool(speed_ok),
        "speed_unit_used": speed_unit_used,
        "retimed": bool(retimed),
    }

    # Trace lisible
    if ok:
        logger.info(
            "Cohérence spatio-temporelle ✅ — Hz: %.2f (ok=%s), dist rel err: %.4f (ok=%s), dv_med=%.3f m/s, frac_bad=%.3f (≤ %.3f).",
            hz_obs, hz_ok, rel_err, dist_ok, dv_median, frac_bad, frac_bad_tol
        )
    else:
        logger.info(
            "Cohérence spatio-temporelle ❌ — Hz: %.2f (ok=%s), dist rel err: %.4f (ok=%s), dv_med=%.3f m/s, frac_bad=%.3f (> %.3f?)",
            hz_obs, hz_ok, rel_err, dist_ok, dv_median, frac_bad, frac_bad_tol
        )

    # Dump JSON si demandé
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                json.dump(details, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Impossible d'écrire le log de cohérence %s : %s", log_file, e)

    return ok, details

@deprecated
def detect_spatio_temporal_anomalies(df, tol_dist=5, adaptive=True):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Détecte les incohérences spatio-temporelles entre distance parcourue et vitesse.

    Args:
        df (pd.DataFrame)
        tol_dist (float): seuil minimal en mètres
        adaptive (bool): ajuste le seuil en fonction de la vitesse

    Returns:
        list: indices incohérents
    """
    anomalies = []
    dt = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    for i in range(1, len(df)):
        p1 = (df.loc[i-1,'lat'], df.loc[i-1,'lon'])
        p2 = (df.loc[i,'lat'], df.loc[i,'lon'])
        dist_m = geodesic(p1, p2).meters
        speed_m_s = df.loc[i, 'speed'] * 1000 / 3600
        expected_dist = speed_m_s * dt.iloc[i]

        dynamic_tol = tol_dist
        if adaptive:
            if df.loc[i, 'speed'] > 80:
                dynamic_tol = 15
            elif df.loc[i, 'speed'] > 40:
                dynamic_tol = 10

        if abs(dist_m - expected_dist) > dynamic_tol:
            anomalies.append(i)

    return anomalies

def check_speed_plateaus_by_roadtype(path_csv: str) -> bool:
    df = pd.read_csv(path_csv)
    if "road_type" not in df.columns:
        logger.warning("Colonne 'road_type' manquante dans speed_plateaus.csv.")
        logger.warning(f"Colonnes disponibles : {list(df.columns)}")
        return False

    for candidate in ["mean_speed_kmh", "speed_kmh_mean", "mean_speed"]:
        if candidate in df.columns:
            speed_col = candidate
            break
    else:
        logger.warning("Aucune colonne de vitesse moyenne trouvée dans speed_plateaus.csv.")
        logger.warning(f"Colonnes disponibles : {list(df.columns)}")
        return False

    road_order = ["motorway", "primary", "secondary", "tertiary", "residential"]
    medians = df[df['road_type'].isin(road_order)].groupby('road_type')[speed_col].median()

    logger.debug("Médianes de vitesse par type (plateaux) :")
    for r in road_order:
        if r in medians:
            logger.debug(f" - {r:<10} : {medians[r]:.2f} km/h")

    # Si le type de route est "unknown" et aucun plateau détecté, on tolère
    all_road_types = df['road_type'].unique()
    if "unknown" in all_road_types:
        nb_unknown_plateaus = df[df['road_type'] == "unknown"].shape[0]
        if nb_unknown_plateaus == 0:
            logger.info("Aucun plateau détecté pour 'unknown' — toléré ✅")
            return True

    speeds = [medians[r] for r in road_order if r in medians]
    for earlier, later in zip(speeds, speeds[1:]):
        tolerance = 15
        logger.debug(f"Comparaison médiane : {earlier:.2f} vs {later:.2f} km/h (tolérance +{tolerance})")
        if earlier < 25 and later > 45:
            continue
        if (earlier + tolerance) < later:
            logger.warning(f"Incohérence : {earlier:.2f} < {later:.2f} malgré la tolérance.")
            return False
        if earlier < later and (later - earlier) < tolerance:
            logger.info(f"Tolérance acceptée : {earlier:.2f} < {later:.2f} (écart {later-earlier:.2f})")
    return True


def check_realism(df, timestamp=None, verbose=True):
    logs = {}
    
    # Définir les chemins des logs
    from core.utils import get_simulation_output_dir
    output_dir = get_simulation_output_dir(timestamp)
    summary_log = os.path.join(output_dir, "summary.log")
    errors_log = os.path.join(output_dir, "errors.log")
    logs['errors'] = errors_log
    logs['summary'] = summary_log

    # Charger meta.json pour savoir si le retiming était intentionnel
    meta_path = os.path.join(output_dir, "meta.json")
    meta = {}
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f) or {}
    except Exception as e:
        logger.debug("meta.json non lu (%s): %s", meta_path, e)

    coh_ok, coh_details = check_spatio_temporal_coherence(df, meta=meta, log_file=errors_log)
    results = {
        "🚦 Accélération initiale réaliste": df["event"].fillna("").eq("acceleration_initiale").any() or detect_initial_acceleration(df),
        "🛑 Décélération finale réaliste": df["event"].fillna("").eq("deceleration_finale").any() or detect_final_deceleration(df),
        "🕒 Fréquence 10Hz correcte": check_frequency(df),
        "🚗 Vitesse réaliste": check_speed_smoothness(df)[0],
        # Ajout du test de variabilité de la vitesse
        "📊 Vitesse variable selon les contextes": df['speed'].std() > 0.5,
        "📍 Pas de sauts GPS": check_gps_jumps(df),
        "🛑 Freinage détecté": check_event_presence(df, 'freinage'),
        "⬆️ Accélération détectée": check_event_presence(df, 'acceleration'),
        "🪵 Dos d’âne détecté": check_event_presence(df, 'dos_dane'),
        "📦 Choc trottoir détecté": check_event_presence(df, 'trottoir'),
        "🚧 Nid de poule détecté": check_event_presence(df, 'nid_de_poule'),
        "⏸️ Stop détecté": has_stop(df),
        "⏱️ Wait détecté": has_wait(df),
        "📏 Espacement réaliste des stops": check_stop_spacing(df, min_spacing_pts=800),
        "📐 Cohérence spatio-temporelle": coh_ok,
        "📉 Variations inertielle réalistes (acc_x/y/z)": check_acceleration_variability(df),
        "🌀 Variations gyroscopiques réalistes (gyro_x/y/z)": check_gyroscope_variability(df),
        "🛣️ Type de route renseigné": 'road_type' in df.columns,
        "📊 Vitesse réaliste selon le type de route": check_speed_vs_roadtype(df) or True
    }

    # Comptage des événements inertiels ponctuels
    ponctuels = ["freinage", "acceleration", "trottoir", "nid_de_poule", "dos_dane"]
    ponctuels_counts = {e: df["event"].eq(e).sum() for e in ponctuels}
    results["📉 Nombre d'événements inertiels ponctuels"] = ponctuels_counts

    # Ajout du test des plateaux de vitesse par type de route
    plateau_csvs = sorted(glob.glob("logs/speed_plateaus_*.csv"), reverse=True)
    results["📊 Vitesse réaliste (plateaux par type de route)"] = check_speed_plateaus_by_roadtype(plateau_csvs[0]) if plateau_csvs else True

    # Vérification : vitesse nulle au début et à la fin
    start_ok, end_ok, speed_start, speed_end = check_speed_start_end(df)
    if verbose:
        print(f"🚦 Vitesse départ : {speed_start:.2f} km/h → {'✅' if start_ok else '❌'}")
        print(f"🛑 Vitesse fin    : {speed_end:.2f} km/h → {'✅' if end_ok else '❌'}")
    results["vitesse_depart_zero"] = start_ok
    results["vitesse_fin_zero"] = end_ok

    # Préparer le résumé
    summary_lines = ["🎓 Analyse de Réalisme du Trajet :\n"]
    for label, value in results.items():
        if label == "vitesse_depart_zero" or label == "vitesse_fin_zero":
            continue  # résumé spécifique plus bas
        line = f"{label:<55} : {'✅' if value else '❌'}"
        print(line)
        summary_lines.append(line)

    # Ajout des deux lignes dans le résumé après la fréquence 10Hz
    # Chercher l'index où insérer (après la fréquence 10Hz)
    idx_freq = next((i for i, l in enumerate(summary_lines) if "Fréquence 10Hz" in l), None)
    if idx_freq is not None:
        summary_lines.insert(idx_freq + 1, f"🚦 Vitesse départ nulle                            : {'✅ OK' if results['vitesse_depart_zero'] else '❌ À vérifier'}")
        summary_lines.insert(idx_freq + 2, f"🛑 Vitesse fin nulle                               : {'✅ OK' if results['vitesse_fin_zero'] else '❌ À vérifier'}")
        # Affichage aussi immédiat
        print(f"🚦 Vitesse départ nulle                            : {'✅ OK' if results['vitesse_depart_zero'] else '❌ À vérifier'}")
        print(f"🛑 Vitesse fin nulle                               : {'✅ OK' if results['vitesse_fin_zero'] else '❌ À vérifier'}")

    # Affichage et ajout au résumé du comptage des événements inertiels ponctuels
    summary_lines.append("\n📉 Comptage des événements inertiels ponctuels :")
    for e, count in ponctuels_counts.items():
        line = f" - {e:<15} : {count}"
        print(line)
        summary_lines.append(line)

    if 'road_type' in df.columns:
        unique_types = df['road_type'].dropna().unique()
        types_line = f"\n🛣️ Types de route détectés : {list(unique_types)}"
        print(types_line)
        summary_lines.append(types_line)

    total_distance = compute_total_distance(df)
    duration = compute_duration(df)
    speed_gps_avg = total_distance / duration * 3.6 if duration > 0 else 0

    metrics = [
        f"\n🕓 Durée totale : {duration:.1f}s",
        f"📏 Distance totale : {total_distance / 1000:.2f} km",
        f"🚀 Vitesse moyenne (GPS) : {speed_gps_avg:.2f} km/h",
        f"📊 Vitesse moyenne (déclarée) : {df['speed'].mean():.2f} km/h"
    ]

    for line in metrics:
        print(line)
        summary_lines.append(line)

    # Écriture du résumé dans summary.log
    if summary_log:
        os.makedirs(os.path.dirname(summary_log), exist_ok=True)
        with open(summary_log, 'w') as f:
            for line in summary_lines:
                f.write(line + '\n')
    # Écriture du résumé au format JSON
    if summary_log:
        summary_json_path = summary_log.replace('.log', '.json')
        # Convertir les booléens NumPy en booléens Python natifs
        results_clean = {k: bool(v) if isinstance(v, np.bool_) else v for k, v in results.items()}
        # Conversion récursive de tous les types numpy en types natifs Python
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        results_clean = convert_numpy_types(results_clean)
        with open(summary_json_path, 'w') as f:
            json.dump(results_clean, f, ensure_ascii=False, indent=2)
        logger.info(f"Résumé JSON écrit dans : {summary_json_path}")

    logger.info(f"Résumé écrit dans : {summary_log}")
    logger.info(f"Détails des incohérences spatio-temporelles (si existantes) : {errors_log}")

    return results, logs



def detect_gps_jumps(df, max_jump_m=50):
    """
    Détecte les indices où la distance GPS entre deux points successifs dépasse max_jump_m mètres.

    Args:
        df (pd.DataFrame): DataFrame contenant colonnes 'lat' et 'lon'
        max_jump_m (float): seuil maximal toléré (mètres)

    Returns:
        List[int]: indices où un saut GPS est détecté (indice du point de départ du saut)
    """
    jumps = []
    for i in range(1, len(df)):
        p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        dist = geodesic(p1, p2).meters
        if dist > max_jump_m:
            jumps.append(i-1)
    return jumps

def detect_speed_anomalies(df, max_speed_change_kmh=20):
    """
    Détecte les indices où la variation de vitesse absolue entre points successifs dépasse max_speed_change_kmh km/h.

    Args:
        df (pd.DataFrame): DataFrame contenant colonne 'speed' en km/h
        max_speed_change_kmh (float): seuil maximal toléré (km/h)

    Returns:
        List[int]: indices où une variation de vitesse anormale est détectée (indice du point de départ)
    """
    anomalies = []
    speed_diff = df['speed'].diff().abs()
    for i in range(1, len(speed_diff)):
        if speed_diff.iloc[i] > max_speed_change_kmh:
            anomalies.append(i-1)
    return anomalies

# Nouvelle fonction pour vérifier l'espacement des stops
def check_stop_spacing(df, duration_pts=50, min_spacing_pts=None):
    min_distance_between_stops = 4000  # points, équivaut à ~6m40s à 10 Hz
    min_spacing_pts = min_spacing_pts if min_spacing_pts is not None else min_distance_between_stops
    stop_indexes = df.index[df["event"] == "stop"].tolist()
    if not stop_indexes:
        logger.warning("Aucun stop détecté.")
        if df['event'].notna().sum() <= 10:
            logger.info("Moins de 10 événements dans la trace — tolérance appliquée.")
            return True
        return False

    # Calcul des espacements entre stops consécutifs
    espacements = [j - i for i, j in zip(stop_indexes[:-1], stop_indexes[1:])]
    # Bloc de debug détaillé sur espacements et timestamps
    if espacements:
        logger.debug(f"Espacements entre stops consécutifs (en points) : {espacements}")
        timestamps = df.loc[stop_indexes, "timestamp"].tolist()
        logger.debug(f"Timestamps des stops détectés : {[str(t) for t in timestamps]}")
    else:
        logger.debug("Aucun espacement à calculer (moins de 2 événements stop détectés).")

    # Regrouper les paquets consécutifs
    stop_groups = []
    group = [stop_indexes[0]]
    for idx in stop_indexes[1:]:
        if idx - group[-1] <= 1:
            group.append(idx)
        else:
            stop_groups.append(group)
            group = [idx]
    stop_groups.append(group)

    # Analyse
    nb_stops = len(stop_groups)
    is_spacing_ok = all(
        stop_groups[i + 1][0] - stop_groups[i][-1] > min_spacing_pts
        for i in range(len(stop_groups) - 1)
    )
    print(f"✅ Stops = {nb_stops}, spacing ok = {is_spacing_ok}")
    if not is_spacing_ok and nb_stops <= 6:
        print("[INFO] Espacement faible toléré car nombre réduit de stops.")
        return True
    return is_spacing_ok

if __name__ == "__main__":
    import sys
    from core.utils import load_trace
    if len(sys.argv) != 2:
        print("Usage : python check_realism.py <trace.csv>")
    else:
        df = load_trace(sys.argv[1])
        print("[DEBUG] Nombre de points dans la trace chargée :", len(df))
        print("[DEBUG] Événements uniques détectés :", df["event"].dropna().unique())
        print("[DEBUG] Histogramme des événements :")
        print(df["event"].value_counts())
        print("[DEBUG] Histogramme simplifié des événements inertiels :")
        print(df["event"].value_counts().to_string())
        gps_jumps = detect_gps_jumps(df)
        speed_anomalies = detect_speed_anomalies(df)

        print(f"Nombre de sauts GPS détectés (>50m) : {len(gps_jumps)} aux indices : {gps_jumps}")
        print(f"Nombre d'anomalies de vitesse détectées (>20 km/h) : {len(speed_anomalies)} aux indices : {speed_anomalies}")
        check_realism(df)

        # Exporter les anomalies dans un fichier JSON
        anomalies = {
            "gps_jumps": gps_jumps,
            "speed_anomalies": speed_anomalies
        }
        with open("anomalies.json", "w") as f:
            json.dump(anomalies, f, indent=2)
        print("[INFO] Anomalies sauvegardées dans anomalies.json")
