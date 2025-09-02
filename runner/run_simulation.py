import os
from datetime import datetime
import logging

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from simulator.events.gyro import generate_gyroscope_signals


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
from simulator.events.opening import inject_opening_for_deliveries
from simulator.pipeline_utils import inject_all_events
from simulator.events.utils import clean_invalid_events

from check.check_realism import check_realism


from runner.generate_outputs_from_csv import generate_all_outputs_from_csv

# NEW v1.0 dataset enrichers & schema enforcement
try:
    from enrichments.delivery_markers import apply_delivery_markers
except Exception:
    apply_delivery_markers = None

try:
    from enrichments.event_category_mapper import project_event_categories
except Exception:
    project_event_categories = None

try:
    from enrichments.altitude_enricher import enrich_altitude
except Exception:
    enrich_altitude = None

try:
    from core.exporters import enforce_schema_order
except Exception:
    enforce_schema_order = None

# --- Helpers for realism enforcement ---

def clamp_speed_changes(df: pd.DataFrame, hz: int = 10, a_max_mps2: float = 2.0) -> pd.DataFrame:
    """Clamp per-tick speed variations to a physical acceleration limit.
    At 10 Hz and 2.0 m/s², dv_max ≈ 0.72 km/h per tick.
    """
    if "speed" not in df.columns or df.empty:
        return df
    dv_max_kmh = (a_max_mps2 * 3.6) / max(hz, 1)
    v = df["speed"].to_numpy(copy=True)
    for i in range(1, len(v)):
        dv = v[i] - v[i - 1]
        if dv > dv_max_kmh:
            v[i] = v[i - 1] + dv_max_kmh
        elif dv < -dv_max_kmh:
            v[i] = v[i - 1] - dv_max_kmh
    df["speed"] = v
    return df



def enforce_min_stop_spacing(df: pd.DataFrame, min_spacing_s: float = 60.0, hz: int = 10, labels=("stop",)) -> pd.DataFrame:
    """Ensure a minimum spacing (seconds) between successive *block starts* of given labels.
    Keeps the first block and clears later blocks starting too close.
    """
    if "event" not in df.columns or df.empty:
        return df
    min_gap_pts = int(min_spacing_s * hz)
    event = df["event"].astype("object")
    is_label = event.isin(labels)
    starts = is_label & (~is_label.shift(1, fill_value=False))
    start_idxs = df.index[starts].to_list()
    last_kept = None
    to_clear = []
    for idx in start_idxs:
        if last_kept is None or (idx - last_kept) >= min_gap_pts:
            last_kept = idx
        else:
            j = idx
            while j < len(df) and df.at[j, "event"] in labels:
                to_clear.append(j)
                j += 1
    if to_clear:
        df.loc[to_clear, "event"] = np.nan
    return df


# --- NEW: Ensure the trajectory ends at a full stop ---
def ensure_final_full_stop(df: pd.DataFrame, window_s: float = 5.0, hz: int = 10) -> pd.DataFrame:
    """Force the trajectory to end at a full stop.

    Over the last `window_s` seconds, linearly ramp the speed down to 0 and
    tag points as 'stop' when the event is empty. Also set a small negative
    longitudinal acceleration to keep kinematic consistency.
    """
    if df is None or df.empty or "speed" not in df.columns:
        return df
    n = len(df)
    window = max(1, min(int(window_s * hz), n))
    start = n - window

    # Guard against missing data
    try:
        v_start = float(df.iloc[start]["speed"]) if not pd.isna(df.iloc[start]["speed"]) else 0.0
    except Exception:
        v_start = 0.0

    # Linear ramp from current speed to 0
    try:
        df.loc[start:n-1, "speed"] = np.linspace(v_start, 0.0, window)
    except Exception:
        # Fallback per-row assignment
        for k, v in enumerate(np.linspace(v_start, 0.0, window)):
            df.at[start + k, "speed"] = v

    # Apply a modest braking profile (about -1.5 m/s^2)
    if "acc_x" in df.columns:
        df.loc[start:n-1, "acc_x"] = -1.5
    for col in ("acc_y", "acc_z"):
        if col in df.columns:
            df.loc[start:n-1, col] = 0.0

    # Label the tail as 'stop' only where event is NaN
    if "event" in df.columns:
        df["event"] = df["event"].astype("object")
        mask = df.loc[start:n-1, "event"].isna()
        if mask.any():
            df.loc[start:n-1, "event"] = df.loc[start:n-1, "event"].where(~mask, "stop")

    return df

# Optionnel : gestion centralisée des sorties via RS3DS (si disponible)
try:
    from core.rs3df import RS3DS  # Classe centralisant I/O, schéma et conventions d'export
except Exception:
    RS3DS = None


# ============================================
# Pipeline principal de simulation RoadSimulator3
# Étapes : génération de la route, simulation inertielle, reprojection, visualisation
# ============================================


def run_simulation(input_csv=None, speed_target_kmh=30, use_rs3ds: bool = True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 🔧 1. Chargement de la configuration
    full_config = load_full_config()
    full_config["simulation"]["target_speed_kmh"] = speed_target_kmh
    # NEW: ensure schema path is available for exporters
    full_config.setdefault("schema_path", "config/dataset_schema.yaml")

    from core.plugins.loader import load_plugins
    from core.exporters_schema_merge import merge_schema

    # Charger les plugins une seule fois et préparer le schéma fusionné.
    enrichers, runners = load_plugins()

    # Collecter les fragments de schéma fournis par les plugins
    fragments = []
    for p in enrichers:
        try:
            fragments += p.provides_schema_fragments() or []
        except Exception:
            logger.debug("Plugin sans fragments de schéma ou erreur dans provides_schema_fragments(): %s", getattr(p, "__class__", type(p)))

    # Fusionner le schéma core + plugins et conserver le chemin pour l'export final
    import tempfile, yaml
    merged_schema = merge_schema(full_config.get("schema_path", "config/dataset_schema.yaml"), fragments)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(merged_schema, tmp)
        merged_schema_path = tmp.name
    logger.info("[PLUGINS] %d enricher(s) chargés — schéma fusionné prêt: %s", len(enrichers), merged_schema_path)

    # 🎛️ Initialisation RS3DS (si présent)
    dataset = None
    if use_rs3ds and RS3DS is not None:
        try:
            dataset = RS3DS(timestamp=timestamp, config=full_config)
            logger.info("[RS3DS] Initialisé avec dossier de sortie %s", getattr(dataset, "output_dir", "<inconnu>"))
        except Exception as e:
            logger.warning("[RS3DS] Échec d'initialisation (%s). Retour au mode classique.", e)
            dataset = None

    # Dossier de sortie (RS3DS prioritaire si dispo)
    if dataset is not None and hasattr(dataset, "output_dir"):
        output_dir = dataset.output_dir
    else:
        output_dir = get_simulation_output_dir(timestamp)
        logger.info("📁 Dossier complet : %s", output_dir)

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
    logger.debug("Points retournés par simulate_route_via_osrm : %d", len(df))

    # 🚧 4. Enrichissement du contexte : type de route
    # Removed map matching here as per instructions

    # Application des événements "stop"/"wait" sur les positions correspondantes
    from simulator.events.stop_wait import apply_stop_wait_at_positions
    logger.debug("Events à injecter :\n%s", coords_df[["lat", "lon", "event"]])
    df = apply_stop_wait_at_positions(df, coords_df[["lat", "lon", "event"]], window_m=100)
    logger.debug("Points après apply_stop_wait_at_positions : %d", len(df))

    # 🚦 5. Simulation vitesse variable
    df = simulate_variable_speed(df, full_config)
    logger.debug("Points après simulate_variable_speed : %d", len(df))

    df = enrich_road_type_stream(df)

    # 📊 Tracker pour suivi du nombre d'événements avant initial_acceleration
    tracker_pre_initial = EventCounter()
    tracker_pre_initial.count_from_dataframe(df)
    tracker_pre_initial.show(label="Avant inject_initial_acceleration")

    from simulator.events.initial_final import inject_initial_acceleration
    df = inject_initial_acceleration(df, speed_target_kmh, duration=5.0)
    logger.debug("Points après inject_initial_acceleration : %d", len(df))

    tracker_post_initial = EventCounter()
    tracker_post_initial.count_from_dataframe(df)
    tracker_post_initial.show(label="Après inject_initial_acceleration")

    # 🛡️ 6. Préparation colonne `event` si absente
    if 'event' not in df.columns:
        df['event'] = pd.Series(index=df.index, dtype='object')
    else:
        df['event'] = df['event'].astype('object')

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
            logger.warning("%d doublons d'événements détectés juste après injection.", len(duplicated_events))
            logger.debug("Aperçu doublons:\n%s", duplicated_events[["timestamp", "event"]].head())
        df["events_injected"] = True
    else:
        logger.warning("⚠️ Événements déjà injectés, saut de l'étape.")

    # 🚪 8. Injection ouverture de porte
    try:
        df = inject_opening_for_deliveries(df)
    except Exception as e:
        logger.debug("inject_opening_for_deliveries skipped (%s)", e)

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
    # Consolidate fragmented stop/wait blocks and enforce spacing realism
    try:
        from simulator.events.stops_and_waits import merge_contiguous_stop_wait
        df = merge_contiguous_stop_wait(df, max_gap_pts=1)
    except Exception:
        logger.debug("merge_contiguous_stop_wait not available; skipping consolidation.")
    # Enforce a minimum spacing between stop blocks only if there are many starts
    try:
        event_obj = df["event"].astype("object")
        is_stop = event_obj == "stop"
        starts_mask = is_stop & (~is_stop.shift(1, fill_value=False))
        n_starts = int(starts_mask.sum())
        if n_starts > 10:  # threshold to avoid over-pruning when only a few stops exist
            df = enforce_min_stop_spacing(df, min_spacing_s=60.0, hz=10, labels=("stop",))
        else:
            logger.debug("Skip enforce_min_stop_spacing: only %d stop starts", n_starts)
    except Exception as e:
        logger.debug("Unable to assess stop starts count (%s); keeping spacing step skipped.", e)

    # 🧹 11. Nettoyage des NaN / inf / données invalides
    df = df.replace([np.inf, -np.inf], np.nan)
    # Pandas >= 2.1: make the cast explicit to avoid FutureWarning on replace
    df = df.infer_objects(copy=False)
    df = df[df[['lat', 'lon', 'timestamp']].notna().all(axis=1)]
    df = df[np.isfinite(df['lat']) & np.isfinite(df['lon'])]
    if df[['lat', 'lon', 'speed']].notna().all(axis=1).sum() < len(df):
        df = df[df[['lat', 'lon', 'speed']].notna().all(axis=1)]
    df = df.reset_index(drop=True)
    logger.debug("Points après nettoyage : %d", len(df))

    for col in ['lat', 'lon', 'speed']:
        if df[col].isna().any():
            raise ValueError(f"Valeurs NaN détectées dans {col} avant reprojection")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Valeurs non finies détectées dans {col} avant reprojection")

    # 📐 12. Reprojection inertielle complète
    logger.debug("Points avant reprojection : %d", len(df))
    df = reprojection.spatial_reprojection(df, speed_target=speed_target_kmh)
    # Physically clamp per-tick speed changes to avoid unrealistically sharp jumps
    df = clamp_speed_changes(df, hz=10, a_max_mps2=2.0)

    tracker_post_reproj = EventCounter()
    tracker_post_reproj.count_from_dataframe(df)
    tracker_post_reproj.show(label="Après spatial_reprojection")

    # --- Ensure `target_speed` exists after reprojection ---
    if ("target_speed" not in df.columns) or df["target_speed"].isna().all():
        try:
            # Prefer library smoothing if available
            from core.kinematics_speed import smooth_target_speed
            df = smooth_target_speed(df, window=9, config=full_config)
        except Exception:
            # Fallback: use a light rolling mean of current speed
            if "speed" in df.columns:
                ts = df["speed"].rolling(9, center=True, min_periods=1).mean()
                df["target_speed"] = ts.bfill().ffill()
            else:
                # Last resort: constant profile from config
                df["target_speed"] = float(full_config["simulation"].get("target_speed_kmh", 30.0))

    df = kinematics.calculate_heading(df)
    df = df[df['heading'].notna() & df['target_speed'].notna()].reset_index(drop=True)
    df = kinematics.calculate_linear_acceleration(df, freq_hz=10)
    df = kinematics.calculate_angular_velocity(df, freq_hz=10)
    df = reprojection.resample_time(df, freq_hz=10)
    # Ensure the trajectory finishes at a complete stop for realism checks
    df = ensure_final_full_stop(df, window_s=5.0, hz=10)

    tracker_post_resample = EventCounter()
    tracker_post_resample.count_from_dataframe(df)
    tracker_post_resample.show(label="Après resample_time")

    # NEW v1.0 — Delivery markers (start/end buttons → in_delivery, delivery_state)
    try:
        if apply_delivery_markers is not None:
            df = apply_delivery_markers(df, config=full_config)
        else:
            logger.debug("apply_delivery_markers not available; skipping delivery markers step.")
    except Exception as e:
        logger.debug("apply_delivery_markers skipped (%s)", e)

    # NEW v1.0 — Event → category projection (event_infra / event_behavior / event_context)
    try:
        if project_event_categories is not None:
            df = project_event_categories(df, config=full_config)
        else:
            logger.debug("project_event_categories not available; skipping event category projection.")
    except Exception as e:
        logger.debug("project_event_categories skipped (%s)", e)

    # NEW v1.0 — Altitude enrichment (altitude_m)
    try:
        if enrich_altitude is not None:
            before_cols = set(df.columns)
            df = enrich_altitude(df, config=full_config)
            after_cols = set(df.columns)
            newly_added = sorted(list(after_cols - before_cols))
            n_ok = int(df.get("altitude_m", pd.Series([pd.NA]*len(df))).notna().sum())
            logger.info("[ALTITUDE] Colonnes ajoutées: %s | points valorisés altitude_m: %d/%d", newly_added, n_ok, len(df))
        else:
            logger.warning("[ALTITUDE] enrich_altitude indisponible — étape ignorée.")
    except Exception as e:
        logger.warning("[ALTITUDE] enrich_altitude a échoué (%s). On continue sans altitude.", e)

    # Sécurité: garantir la présence de 'altitude_m' (même vide) pour le schéma v1.0
    if "altitude_m" not in df.columns:
        if "altitude" in df.columns:
            df["altitude_m"] = df["altitude"].astype("float32")
        else:
            df["altitude_m"] = pd.NA

    # 🧮 13. Sécurisation des colonnes acc / gyro (ensure gyro_x/y/z always present)
    df = generate_gyroscope_signals(df)

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

    # --- Ensure `road_type` exists for realism checks ---
    try:
        if "road_type" not in df.columns:
            logger.warning("`road_type` manquant avant check_realism — fallback en 'unknown'.")
            df["road_type"] = "unknown"
        else:
            # Normalize and fill gaps
            df["road_type"] = df["road_type"].astype("object").fillna("unknown")
    except Exception as e:
        logger.debug("Normalization of road_type failed (%s); forcing 'unknown'", e)
        df["road_type"] = "unknown"

    # 🧪 16. Contrôle de réalisme
    realism_results, realism_logs = check_realism(df, timestamp=timestamp)
    logger.info("\n=== Résumé du contrôle de réalisme ===")
    for label, passed in realism_results.items():
        logger.info(f"{label:40} : {'✅ OK' if passed else '❌ À vérifier'}")
    logger.info(f"\n[INFO] Logs détaillés : {realism_logs['summary']} / {realism_logs['errors']}")

    # Supprimer le label 'stop' du premier point s'il est en tête de trajectoire
    first_stop_idx = df[df["event"] == "stop"].index.min()
    if pd.notna(first_stop_idx):
        logger.info("Suppression du premier 'stop' à l’index %s (point de départ)", first_stop_idx)
        df["event"] = df["event"].astype("object")
        df.at[first_stop_idx, "event"] = float('nan')


    # --- Application des enrichers plugins (après pipeline interne, avant ordre de colonnes/export) ---
    try:
        for enr in enrichers:
            name = getattr(enr, "__class__", type(enr)).__name__
            n_before = len(df)
            df = enr.apply(df, config=full_config)
            logger.info("[PLUGIN] %s appliqué (lignes: %d → %d)", name, n_before, len(df))
    except Exception as e:
        logger.warning("[PLUGINS] Application des enrichers: échec partiel (%s). Poursuite du pipeline.", e)


    # NEW v1.0 — Enforce canonical column order before any export
    try:
        if enforce_schema_order is not None:
            schema_path = locals().get("merged_schema_path", full_config.get("schema_path", "config/dataset_schema.yaml"))
            # Prefer the new kwargs API; gracefully fallback to legacy cfg signature
            try:
                df = enforce_schema_order(
                    df,
                    schema_path=schema_path,
                    drop_extras=True,
                    ensure_altitude_alias=True,
                )
            except TypeError:
                cfg = {
                    "schema_path": schema_path,
                    "drop_extras": True,
                    "ensure_altitude_alias": True,
                }
                df = enforce_schema_order(df, cfg)
        else:
            logger.debug("enforce_schema_order not available; skipping schema order enforcement.")
    except Exception as e:
        logger.debug("enforce_schema_order skipped (%s)", e)

    # 💾 17. Export CSV et visualisations
    logger.debug("Shape finale avant export : %s", df.shape)
    logger.debug("Extrait lat/lon :\n%s", df[['lat', 'lon']].head())

    # Export principal (copie complète de la trajectoire)
    df.to_csv(os.path.join(output_dir, "output_simulated_trajectory.csv"), index=False)

    # 🔁 Export standardisé via RS3DS si disponible, sinon fallback local
    if dataset is not None and hasattr(dataset, "export_standard_trace"):
        try:
            trace_path, link_path = dataset.export_standard_trace(df, filename="trace.csv", create_symlink=True)
            logger.info("[RS3DS] export_standard_trace OK → %s", trace_path)
        except Exception as e:
            logger.warning("[RS3DS] export_standard_trace a échoué (%s). Fallback local.", e)
            trace_path = os.path.join(output_dir, "trace.csv")
            df.to_csv(trace_path, index=False)
            # symlink local
            symlink_path = os.path.join("data", "simulations", "last_trace.csv")
            try:
                if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                    os.remove(symlink_path)
                os.symlink(trace_path, symlink_path)
                logger.info("Lien symbolique créé : %s → %s", symlink_path, trace_path)
            except OSError as e2:
                logger.warning("Impossible de créer le lien symbolique : %s", e2)
    else:
        trace_path = os.path.join(output_dir, "trace.csv")
        df.to_csv(trace_path, index=False)
        logger.info("Fichier CSV standardisé : %s", trace_path)
        # 🔗 Lien symbolique local
        symlink_path = os.path.join("data", "simulations", "last_trace.csv")
        try:
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(trace_path, symlink_path)
            logger.info("Lien symbolique créé : %s → %s", symlink_path, trace_path)
        except OSError as e:
            logger.warning("Impossible de créer le lien symbolique : %s", e)

    generate_all_outputs_from_csv(df, output_dir=output_dir, timestamp=timestamp)


if __name__ == "__main__":
    # Permet l'exécution en tant que module : python -m runner.run_simulation
    run_simulation(input_csv=None, speed_target_kmh=40.0, use_rs3ds=True)