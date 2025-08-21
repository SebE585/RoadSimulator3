import logging
import pandas as pd
import numpy as np

# Fonctions utilitaires pipeline
from simulator.pipeline_utils import apply_postprocessing, inject_all_events

# Gestion des cinématiques et vitesses (majorité dans kinematics_speed)
from core.kinematics_speed import (
    recompute_speed,  
    cap_global_speed_delta,
    apply_target_speed_by_road_type,
    smooth_target_speed,
    interpolate_target_speed_progressively,
    adjust_speed_progressively,
    cap_speed_to_target,
)

# Import spécifique pour fonctions bien définies dans core.kinematics
from core.kinematics import (
    recompute_inertial_acceleration,
    compute_distance,
    resample_trajectory_to_10hz,
    enrich_inertial_coupling,
    compute_kinematic_metrics,
    check_inertial_stats,
)

# API données géographiques
from core.osmnx.client import enrich_road_type_stream
from core.terrain.client import enrich_terrain_via_api

# Injection bruit et gyroscope
from simulator.events.noise import inject_inertial_noise
from simulator.events.gyro import generate_gyroscope_signals

# Événements inertiels
from simulator.events.initial_final import inject_final_deceleration, inject_initial_acceleration

# Détection événements
from simulator.detectors import detect_all_events


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def assert_dataframe_integrity(df, step_label=""):
    issues = []
    if "speed" in df.columns:
        if np.allclose(df["speed"], 0):
            issues.append("⚠️ Toutes les vitesses sont nulles.")
        if df["speed"].isnull().any():
            issues.append("❌ NaNs détectés dans speed.")
    if "acc_x" in df.columns and df["acc_x"].std() < 1e-3:
        issues.append("⚠️ acc_x quasi constant.")
    if "acc_y" in df.columns and df["acc_y"].std() < 1e-3:
        issues.append("⚠️ acc_y quasi constant.")
    if issues:
        print(f"\n[CHECK] Problèmes détectés après étape '{step_label}':")
        for issue in issues:
            print(issue)
    else:
        print(f"[CHECK] ✅ Intégrité OK après '{step_label}'.")


class SimulationPipeline:
    def __init__(self, config: dict):
        self.full_config = config
        self.config = config["simulation"]

    def run(self, df):
        logger.info("🎯 Lancement du pipeline de simulation...")

        hz = self.config["hz"]

        # Assurer la présence d'une colonne speed numérique (avant tout calcul)
        if "speed" not in df.columns:
            df["speed"] = 0.0
        else:
            try:
                df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
            except Exception:
                df["speed"] = 0.0

        # Étape 2 : typer les routes via API OSMnx
        df = enrich_road_type_stream(df)
        assert_dataframe_integrity(df, "enrich_road_type_stream")

        force_target = self.config.get("force_target_speed", False)

        if "target_speed" in df.columns and not force_target:
            logger.warning("⚠️ La colonne 'target_speed' existe déjà — écrasement évité.")
        else:
            if force_target:
                df = apply_target_speed_by_road_type(df, speed_by_type=self.config.get("target_speed_by_road_type"))
                df = interpolate_target_speed_progressively(df, alpha=0.1, config=self.full_config)
            else:
                df = smooth_target_speed(df, window=9, config=self.full_config)
                df = interpolate_target_speed_progressively(df, alpha=0.1, config=self.full_config)

        assert_dataframe_integrity(df, "target_speed_by_road_type")

        # Assurer la présence de 'heading' avant calculs de vitesse et inertie
        if "heading" not in df.columns:
            from core.postprocessing import fill_heading
            df = fill_heading(df)

        # Calcul et ajustements vitesse
        df = recompute_speed(df, iterations=15, alpha=0.25, config=self.full_config)
        df = adjust_speed_progressively(df, config=self.full_config)
        df = cap_speed_to_target(df, alpha=0.2)
        assert_dataframe_integrity(df, "recompute_speed (après target_speed)")

        # --- SPEED RESCUE: rebuild speed from geometry if still NaN or all zeros ---
        if df["speed"].isnull().any() or np.allclose(df["speed"], 0):
            try:
                dt = 1.0 / float(hz if hz else 10)

                if {"x", "y"}.issubset(df.columns):
                    dx = pd.to_numeric(df["x"], errors="coerce").diff()
                    dy = pd.to_numeric(df["y"], errors="coerce").diff()
                    dist = np.sqrt(dx * dx + dy * dy)
                elif {"lat", "lon"}.issubset(df.columns):
                    lat1 = np.radians(pd.to_numeric(df["lat"].shift(), errors="coerce").to_numpy(dtype=float))
                    lon1 = np.radians(pd.to_numeric(df["lon"].shift(), errors="coerce").to_numpy(dtype=float))
                    lat2 = np.radians(pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float))
                    lon2 = np.radians(pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float))
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
                    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                    dist = 6371000.0 * c  # meters
                    dist = pd.Series(np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0), index=df.index)
                else:
                    dist = pd.Series(0.0, index=df.index)

                # to km/h with a light median smoothing
                speed_rescue = (pd.to_numeric(dist, errors="coerce").fillna(0.0) / dt) * 3.6
                speed_rescue = speed_rescue.rolling(9, center=True, min_periods=1).median()

                vmax_series = df.get("target_speed", pd.Series(self.config.get("target_speed_kmh", 30), index=df.index))
                vmax_series = pd.to_numeric(vmax_series, errors="coerce").fillna(self.config.get("target_speed_kmh", 30))
                speed_rescue = np.minimum(speed_rescue, vmax_series)

                mask_invalid = df["speed"].isnull() | np.isclose(df["speed"], 0)
                df.loc[mask_invalid, "speed"] = speed_rescue.loc[mask_invalid]

                # Clamp brutal per-step jumps (km/h per step)
                df = cap_global_speed_delta(df, max_delta_kmh_per_step=3.0)
                assert_dataframe_integrity(df, "recompute_speed rescue")
            except Exception as e:
                logger.warning("[SPEED-RESCUE] Impossible de reconstruire la vitesse: %s", e)

        # Calcul métriques cinématiques
        df = compute_kinematic_metrics(df, hz=hz)
        df = recompute_inertial_acceleration(df, hz=hz)
        print(df["acc_x"].describe())
        assert_dataframe_integrity(df, "recompute_inertial_acceleration")
        check_inertial_stats(df, label="📊 Inertie avant enrichissement")

        # Injection du bruit inertiel
        if "event" not in df.columns:
            df["event"] = np.nan
        for col in ["gyro_x", "gyro_y", "gyro_z"]:
            if col not in df.columns:
                df[col] = 0.0
        std = self.config.get("inertial_noise_std", 0.03)  # valeur par défaut 0.03 si absente
        noise_params = {
            "acc_std": std,
            "gyro_std": 0.15,
            "acc_bias": 0.0,
            "gyro_bias": 0.0,
        }
        df = inject_inertial_noise(df, noise_params)
        assert_dataframe_integrity(df, "inject_inertial_noise")
        check_inertial_stats(df, label="📊 Inertie après enrichissement")

        # Simulation gyroscope (unified)
        df = generate_gyroscope_signals(df, hz=hz)

        # Injection événements inertiels
        v_max_kmh = self.config.get("target_speed_kmh", 30)
        init_dur = self.full_config.get("injection", {}).get("acceleration", {}).get("duration_s", 5.0)
        final_dur = self.full_config.get("injection", {}).get("freinage", {}).get("duration_s", 5.0)

        df = inject_initial_acceleration(df, v_max_kmh, duration=init_dur)
        df = inject_all_events(df, self.full_config)
        assert_dataframe_integrity(df, "inject_all_events")
        df = inject_final_deceleration(df, v_max_kmh, duration=final_dur)

        # Post-traitement inertiel
        df = apply_postprocessing(df, hz=hz, config=self.config)
        assert_dataframe_integrity(df, "apply_postprocessing")
        df = recompute_inertial_acceleration(df, hz=hz)

        # Altimétrie
        df = compute_distance(df)
        df = enrich_terrain_via_api(df)

        # Détection automatique des événements
        detection_summary = detect_all_events(df)
        for k, v in detection_summary.items():
            logger.info(f"  {k:20} : {'✅' if v else '❌'}")

        return df