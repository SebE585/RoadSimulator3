import os
import sys
from datetime import datetime
import logging

import numpy as np
import pandas as pd

# Ajout du répertoire parent pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Script principal de simulation RoadSimulator3.

Ce script exécute une simulation complète de trajet incluant :
- la génération de la trajectoire via OSRM,
- l’injection d’événements inertiels et contextuels,
- la projection inertielle,
- l’injection de bruit réaliste (accéléromètre / gyroscope),
- la validation spatio-temporelle,
- l’export CSV + visualisation.

Usage :
    python runner/run_simulation.py
"""

# Ajout du tracker d'événements
from simulator.events.tracker import EventCounter


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from core import reprojection, validation
import core.kinematics as kinematics
from core.config_loader import load_full_config
from core.utils import (
    ensure_strictly_increasing_timestamps,
    get_simulation_output_dir,
)
from core.kinematics_speed import simulate_variable_speed
from core.osmnx.client import enrich_road_type_stream
from core.osrm.simulate import simulate_route_via_osrm

from simulator.events.noise import inject_inertial_noise
from simulator.events.neutral import inject_neutral_phases
from simulator.events.opening import inject_opening_for_deliveries
from simulator.pipeline_utils import inject_all_events
from simulator.events.utils import clean_invalid_events

from check.check_realism import check_realism
from runner.generate_outputs_from_csv import generate_all_outputs_from_csv


# ============================================
# Pipeline principal de simulation RoadSimulator3
# Étapes : génération de la route, simulation inertielle, reprojection, visualisation
# ============================================
def run_simulation(input_csv=None, speed_target_kmh=30):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 🔧 1. Chargement de la configuration
    full_config = load_full_config()
    full_config["simulation"]["target_speed_kmh"] = speed_target_kmh

    cities_coords = full_config["simulation"].get("cities_coords", [])
    distance_km = full_config["simulation"].get("distance_km", 100)
    nb_stops = full_config["simulation"].get("nb_stops", 15)

    # 🛑 2. Marquage des points de livraison
    from simulator.events.utils import marquer_livraisons
    coords_df = pd.DataFrame(cities_coords, columns=["lat", "lon"])
    coords_df = marquer_livraisons(coords_df, prefix="stop_", start_index=1)
    # Ajout du marquage d'événements "stop"/"wait" alternés
    coords_df["event"] = ["stop" if i % 2 == 0 else "wait" for i in range(len(coords_df))]
    coords = coords_df[["lat", "lon"]].values.tolist()

    # 🗺️ 3. Génération de la trajectoire via OSRM
    df = simulate_route_via_osrm(cities_coords=coords, hz=10)
    print(f"[DEBUG] Points retournés par simulate_route_via_osrm : {len(df)}")

    # 🚧 4. Enrichissement du contexte : type de route
    # Removed map matching here as per instructions

    # Application des événements "stop"/"wait" sur les positions correspondantes
    from simulator.events.stop_wait import apply_stop_wait_at_positions
    print("[DEBUG] Events à injecter :", coords_df[["lat", "lon", "event"]])
    df = apply_stop_wait_at_positions(df, coords_df[["lat", "lon", "event"]], window_m=100)
    print(f"[DEBUG] Points après apply_stop_wait_at_positions : {len(df)}")

    # 🚦 5. Simulation vitesse variable
    df = simulate_variable_speed(df, full_config)
    print(f"[DEBUG] Points après simulate_variable_speed : {len(df)}")

    df = enrich_road_type_stream(df)

    # 📊 Tracker pour suivi du nombre d'événements avant initial_acceleration
    tracker_pre_initial = EventCounter()
    tracker_pre_initial.count_from_dataframe(df)
    tracker_pre_initial.show(label="Avant inject_initial_acceleration")

    from simulator.events.initial_final import inject_initial_acceleration
    df = inject_initial_acceleration(df, speed_target_kmh, duration=5.0)
    print(f"[DEBUG] Points après inject_initial_acceleration : {len(df)}")

    tracker_post_initial = EventCounter()
    tracker_post_initial.count_from_dataframe(df)
    tracker_post_initial.show(label="Après inject_initial_acceleration")

    # 🛡️ 6. Préparation colonne `event` si absente
    if 'event' not in df.columns:
        df['event'] = np.nan

    # 🚧 7. Injection des événements inertiels
    if "events_injected" not in df.columns:
        df["events_injected"] = False

    if not df["events_injected"].any():
        event_tracker = EventCounter()
        df = inject_all_events(df, full_config)
        event_tracker.count_from_dataframe(df)
        event_tracker.show("Après injection")
        duplicated_events = df[df.duplicated(subset=["timestamp", "event"], keep=False) & df["event"].notna()]
        if not duplicated_events.empty:
            print(f"[WARN] {len(duplicated_events)} doublons d'événements détectés juste après injection.")
            print(duplicated_events[["timestamp", "event"]].head())
        df["events_injected"] = True
    else:
        logger.warning("⚠️ Événements déjà injectés, saut de l'étape.")

    # 🚪 8. Injection ouverture de porte
    df = inject_opening_for_deliveries(df)

    # 🆕 9. Injection des phases neutres
    # if df['event'].fillna('').str.contains('wait').sum() < 100:
    #     df = inject_neutral_phases(df)

    from simulator.events.initial_final import inject_final_deceleration
    df = inject_final_deceleration(df, speed_target_kmh, duration=5.0)

    tracker_post_final = EventCounter()
    tracker_post_final.count_from_dataframe(df)
    tracker_post_final.show(label="Après inject_final_deceleration")

    # 🧹 10. Nettoyage des événements invalides
    df = clean_invalid_events(df)

    # 🧹 11. Nettoyage des NaN / inf / données invalides
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[['lat', 'lon', 'timestamp']].notna().all(axis=1)]
    df = df[np.isfinite(df['lat']) & np.isfinite(df['lon'])]
    if df[['lat', 'lon', 'speed']].notna().all(axis=1).sum() < len(df):
        df = df[df[['lat', 'lon', 'speed']].notna().all(axis=1)]
    df = df.reset_index(drop=True)
    print(f"[DEBUG] Points après nettoyage : {len(df)}")

    for col in ['lat', 'lon', 'speed']:
        if df[col].isna().any():
            raise ValueError(f"Valeurs NaN détectées dans {col} avant reprojection")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Valeurs non finies détectées dans {col} avant reprojection")

    # 📐 12. Reprojection inertielle complète
    print(f"[DEBUG] Points avant reprojection : {len(df)}")
    df = reprojection.spatial_reprojection(df, speed_target=speed_target_kmh)

    tracker_post_reproj = EventCounter()
    tracker_post_reproj.count_from_dataframe(df)
    tracker_post_reproj.show(label="Après spatial_reprojection")

    df = kinematics.calculate_heading(df)
    df = df[df['heading'].notna() & df['target_speed'].notna()].reset_index(drop=True)
    df = kinematics.calculate_linear_acceleration(df, freq_hz=10)
    df = kinematics.calculate_angular_velocity(df, freq_hz=10)
    df = reprojection.resample_time(df, freq_hz=10)

    tracker_post_resample = EventCounter()
    tracker_post_resample.count_from_dataframe(df)
    tracker_post_resample.show(label="Après resample_time")

    # 🧮 13. Sécurisation des colonnes acc / gyro
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        if col not in df.columns:
            df[col] = 0.0
    if 'event' not in df.columns:
        df['event'] = np.nan

    # 📊 Suivi final des événements
    final_tracker = EventCounter()
    final_tracker.count_from_dataframe(df)
    final_tracker.show(label="Final après reprojection")

    # 🔉 14. Injection bruit inertiel
    noise_params = full_config.get("simulation", {})
    noise_params_extracted = {
        'acc_std': noise_params.get('inertial_noise_std', 0.05),
        'gyro_std': noise_params.get('gyro_std', 0.01),
        'acc_bias': noise_params.get('acc_bias', 0.02),
        'gyro_bias': noise_params.get('gyro_bias', 0.005)
    }
    df = inject_inertial_noise(df, noise_params_extracted)
    df = ensure_strictly_increasing_timestamps(df)

    # ✅ 15. Contrôles de cohérence (timestamps, spatial)
    validation.validate_timestamps(df)
    validation.validate_spatial_coherence(df, max_speed=130)
    _ = validation.compute_speed_stats(df)

    # 🧪 16. Contrôle de réalisme
    realism_results, realism_logs = check_realism(df, timestamp=timestamp)
    logger.info("\n=== Résumé du contrôle de réalisme ===")
    for label, passed in realism_results.items():
        logger.info(f"{label:40} : {'✅ OK' if passed else '❌ À vérifier'}")
    logger.info(f"\n[INFO] Logs détaillés : {realism_logs['summary']} / {realism_logs['errors']}")

    # Supprimer le label 'stop' du premier point s'il est en tête de trajectoire
    first_stop_idx = df[df["event"] == "stop"].index.min()
    if pd.notna(first_stop_idx):
        print(f"[INFO] Suppression du premier 'stop' à l’index {first_stop_idx}, considéré comme point de départ.")
        df.at[first_stop_idx, "event"] = float('nan')

    # 💾 17. Export CSV et visualisations
    output_dir = get_simulation_output_dir(timestamp)
    logger.info(f"📁 Dossier complet : {output_dir}")
    print(f"[DEBUG] Shape finale avant export : {df.shape}")
    print(f"[DEBUG] Extrait lat/lon :\n{df[['lat', 'lon']].head()}")
    df.to_csv(os.path.join(output_dir, "output_simulated_trajectory.csv"), index=False)

    # 🔁 Copie sous le nom standardisé trace.csv
    trace_path = os.path.join(output_dir, "trace.csv")
    df.to_csv(trace_path, index=False)
    print(f"✅ Fichier CSV standardisé : {trace_path}")

    # 🔗 Création du lien symbolique vers la dernière trace
    symlink_path = os.path.join("data", "simulations", "last_trace.csv")
    try:
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(trace_path, symlink_path)
        print(f"🔗 Lien symbolique créé : {symlink_path} → {trace_path}")
    except OSError as e:
        print(f"[WARN] Impossible de créer le lien symbolique : {e}")

    generate_all_outputs_from_csv(df, output_dir=output_dir, timestamp=timestamp)


if __name__ == "__main__":
    run_simulation(input_csv=None, speed_target_kmh=40.0)