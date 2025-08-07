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
from core.kinematics_speed import apply_target_speed_by_road_type
from simulator.cleaning import clean_simulation_errors

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

    from simulator.events.gyro import recompute_inertial_acceleration

    # Étapes complémentaires post-pipeline
    df = adjust_speed_progressively(df)
    df = recompute_inertial_acceleration(df, hz=config["simulation"]["hz"])
    df = interpolate_target_speed_progressively(
        df,
        alpha=0.1,
        force=config["simulation"].get("force_target_speed", False)
    )
    df = recompute_inertial_acceleration(df, hz=config["simulation"]["hz"])
    df = cap_speed_to_target(df, alpha=0.2)
    df = recompute_inertial_acceleration(df, hz=config["simulation"]["hz"])

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
