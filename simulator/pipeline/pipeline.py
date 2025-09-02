import logging
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
from simulator.events.gyro import simulate_gyroscope_from_heading, inject_gyroscope_from_events

# Événements inertiels
from simulator.events.generation import apply_final_deceleration, apply_initial_acceleration

# Détection événements
from simulator.s import detect_all_events

# v1.0 enrichers & schema enforcement (guarded imports)
try:
    from enrichments.delivery_markers import apply_delivery_markers
except Exception:
    apply_delivery_markers = None

try:
    from enrichments.event_category_mapper import project_event_categories
except Exception:
    project_event_categories = None

try:
    from core.exporters import enforce_schema_order
except Exception:
    enforce_schema_order = None


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

        # Calcul métriques cinématiques
        df = compute_kinematic_metrics(df, hz=hz)
        df = recompute_inertial_acceleration(df, hz=hz)
        print(df["acc_x"].describe())
        assert_dataframe_integrity(df, "recompute_inertial_acceleration")
        check_inertial_stats(df, label="📊 Inertie avant enrichissement")

        # v1.0 — Colonnes minimales assurées
        if "event" not in df.columns:
            df["event"] = np.nan
        for col in ["gyro_x", "gyro_y", "gyro_z"]:
            if col not in df.columns:
                df[col] = 0.0

        # v1.0 — Marqueurs début/fin de livraison (in_delivery / delivery_state)
        try:
            if apply_delivery_markers is not None:
                df = apply_delivery_markers(df, config=self.full_config)
        except Exception:
            logger.debug("apply_delivery_markers skipped", exc_info=True)

        # v1.0 — Projection des catégories d'événements (event_infra/behavior/context)
        try:
            if project_event_categories is not None:
                df = project_event_categories(df, config=self.full_config)
        except Exception:
            logger.debug("project_event_categories skipped", exc_info=True)

        # Injection événements inertiels (avant gyroscope dépendant des events)
        df = apply_initial_acceleration(df, self.full_config)
        df = inject_all_events(df, self.full_config)
        assert_dataframe_integrity(df, "inject_all_events")
        df = apply_final_deceleration(df, self.full_config)

        # Simulation gyroscope (assure présence même si pas d'utilisation)
        df = simulate_gyroscope_from_heading(df)
        df = inject_gyroscope_from_events(df)

        # Injection du bruit inertiel (acc/gyro) — après events & gyro de base
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

        # v1.0 — Enforcement de l'ordre canonique des colonnes avant retour
        try:
            if enforce_schema_order is not None:
                df = enforce_schema_order(df, self.full_config)
        except Exception:
            logger.debug("enforce_schema_order skipped", exc_info=True)

        return df