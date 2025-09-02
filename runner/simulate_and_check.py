import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from core.config_loader import load_full_config
from core.osrm.simulate import simulate_route_via_osrm
from core.utils import ensure_csv_column_order
from core.reports import generate_reports
from check.check_realism import check_realism
from simulator.pipeline.pipeline import SimulationPipeline

from simulator.cleaning import clean_simulation_errors

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

# ↓↓↓ Supprime les logs de debug liés aux polices matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ↓↓↓ Logger principal de l'application
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def simulate_and_enrich(csv_path: str = None, outdir: str = None) -> pd.DataFrame:
    """
    Simule ou charge un trajet, puis applique le pipeline d'enrichissement inertiel/contextuel.

    Args:
        csv_path (str, optional): Chemin vers un fichier CSV existant à charger.
                                  Si None, génère un nouveau trajet via OSRM.
        outdir (str, optional): Répertoire de sortie pour enregistrer les graphiques.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les données simulées.
    """
    config = load_full_config()
    # Ensure schema path is available for canonical ordering
    config.setdefault("schema_path", "config/dataset_schema.yaml")
    pipeline = SimulationPipeline(config)

    if csv_path and os.path.exists(csv_path):
        logger.info(f"📂 Chargement du fichier CSV existant : {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        logger.info("🛰️ Simulation d’un nouveau trajet via OSRM...")
        df = simulate_route_via_osrm(
            cities_coords=config["simulation"]["cities_coords"],
            hz=config["simulation"]["hz"],
            step_m=config["simulation"]["step_m"]
        )
        logger.info(f"✅ Trajet interpolé initial : {len(df)} points générés.")

    df = pipeline.run(df)

    from core.kinematics_speed import (
        adjust_speed_progressively,
        interpolate_target_speed_progressively,
        cap_speed_to_target
    )

    from simulator.events.gyro import generate_gyroscope_signals

    # Étapes complémentaires post-pipeline
    df = adjust_speed_progressively(df)
    df = interpolate_target_speed_progressively(
        df,
        alpha=0.1,
        force=config["simulation"].get("force_target_speed", False)
    )
    df = cap_speed_to_target(df, alpha=0.2)
    df = generate_gyroscope_signals(df)

    # v1.0 — Delivery markers & event category projection
    try:
        if apply_delivery_markers is not None:
            df = apply_delivery_markers(df, config=config)
    except Exception:
        logger.debug("apply_delivery_markers skipped", exc_info=True)

    try:
        if project_event_categories is not None:
            df = project_event_categories(df, config=config)
    except Exception:
        logger.debug("project_event_categories skipped", exc_info=True)

    # v1.0 — Enforce canonical column order (keeps `event` for backward-compat)
    try:
        if enforce_schema_order is not None:
            df = enforce_schema_order(df, config)
    except Exception:
        logger.debug("enforce_schema_order skipped", exc_info=True)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plot_path = os.path.join(outdir, "trace_speed_vs_index.png")
        df["speed"].plot(title="Vitesse vs index").figure.savefig(plot_path)
        plt.close()

    logger.info(f"🚀 Vitesse maximale : {df['speed'].max():.2f} km/h")
    return df


def main():
    """
    Point d’entrée de la simulation : simulation ou lecture CSV, enrichissement,
    export CSV, vérification de réalisme, génération des rapports.
    """
    parser = argparse.ArgumentParser(description="Simule et enrichit une trajectoire véhicule")
    parser.add_argument('--csv', type=str, default=None,
                        help="Chemin vers un fichier CSV à charger au lieu de simuler")
    parser.add_argument('--outdir', type=str, default=None,
                        help="Répertoire de sortie pour les fichiers générés (défaut : data/simulations/)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f'data/simulations/simulated_{timestamp}'
    os.makedirs(outdir, exist_ok=True)

    df = simulate_and_enrich(csv_path=args.csv, outdir=outdir)

    # Nettoyage des erreurs critiques détectées
    df = clean_simulation_errors(df)

    # Optional sanity: log presence of v1.0 columns
    for col in ("in_delivery", "delivery_state", "event_infra", "event_behavior", "event_context"):
        if col in df.columns:
            logger.info(f"[v1.0] Colonne présente : {col}")

    df = ensure_csv_column_order(df)
    csv_path = os.path.join(outdir, 'trace.csv')
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"[DEBUG] ✅ Fichier CSV écrit à : {os.path.abspath(csv_path)}")
    except Exception as e:
        logger.error(f"[❌] Échec d'écriture du fichier CSV : {e}")
        raise

    logger.info(f"📤 CSV exporté : {csv_path}")

    # Vérification de la cohérence inertielle et structurelle
    check_realism(df, timestamp=timestamp)

    # Résumé des indicateurs clés
    logger.info("\n--- ✅ Résumé final ---")
    logger.info(f"🔢 Points générés : {len(df)}")
    if 'distance_m' in df.columns:
        logger.info(f"📏 Distance estimée : {df['distance_m'].sum() / 1000:.2f} km")
    if 'speed' in df.columns:
        logger.info(f"🚗 Vitesse moyenne : {df['speed'].mean():.2f} km/h")
    logger.info(f"📁 Dossier complet : {outdir}")
    logger.info(f"🚀 Vitesse maximale : {df['speed'].max():.2f} km/h")

    generate_reports(df, outdir)


if __name__ == "__main__":
    main()
