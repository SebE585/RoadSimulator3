# runner/simulate.py
import os
import argparse
import logging
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from core.config_loader import load_full_config
from core.utils import get_simulation_output_dir, ensure_strictly_increasing_timestamps
from core.osrm.simulate import simulate_route_via_osrm
from simulator.pipeline.pipeline import SimulationPipeline
from check.check_realism import check_realism
from runner.generate_outputs_from_csv import generate_all_outputs_from_csv

# Optionnel : I/O centralisée si dispo
try:
    from core.rs3df import RS3DS
except Exception:
    RS3DS = None

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def export_trace(df: pd.DataFrame, output_dir: str, timestamp: str, full_config: dict):
    """Export standard (RS3DS si dispo sinon local) + lien symbolique + viz."""
    # Export principal pour inspection
    df.to_csv(os.path.join(output_dir, "output_simulated_trajectory.csv"), index=False)

    # Export standardisé
    if RS3DS is not None:
        try:
            dataset = RS3DS(timestamp=timestamp, config=full_config)
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
        symlink_path = os.path.join("data", "simulations", "last_trace.csv")
        try:
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(trace_path, symlink_path)
            logger.info("Lien symbolique créé : %s → %s", symlink_path, trace_path)
        except OSError as e:
            logger.warning("Impossible de créer le lien symbolique : %s", e)

    # Génère les sorties (html/graphs)
    generate_all_outputs_from_csv(df, output_dir=output_dir, timestamp=timestamp)


def main():
    parser = argparse.ArgumentParser(description="RoadSimulator3 – lanceur pipeline unique")
    parser.add_argument("--speed-kmh", type=float, default=40.0, help="Vitesse cible par défaut (km/h)")
    parser.add_argument("--no-rs3ds", action="store_true", help="Désactiver l’export RS3DS")
    parser.add_argument("--hz", type=int, default=None, help="Fréquence d’échantillonnage (override config)")
    args = parser.parse_args()

    # 1) Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_config = load_full_config()
    full_config["simulation"]["target_speed_kmh"] = float(args.speed_kmh)
    if args.hz is not None:
        full_config["simulation"]["hz"] = int(args.hz)

    # 2) Dossier de sortie
    if RS3DS is not None and not args.no_rs3ds:
        try:
            dataset = RS3DS(timestamp=timestamp, config=full_config)
            output_dir = dataset.output_dir
            logger.info("[RS3DS] Initialisé → %s", output_dir)
        except Exception as e:
            logger.warning("[RS3DS] Échec init (%s). Fallback local.", e)
            output_dir = get_simulation_output_dir(timestamp)
            logger.info("📁 Dossier de sortie : %s", output_dir)
    else:
        output_dir = get_simulation_output_dir(timestamp)
        logger.info("📁 Dossier de sortie : %s", output_dir)

    # 3) Points / route d’entrée (sources dans la conf)
    sim_conf = full_config["simulation"]
    cities_coords = sim_conf.get("cities_coords", [])
    hz = sim_conf.get("hz", 10)

    # 4) Génération de la trajectoire OSRM (entrée du pipeline)
    df = simulate_route_via_osrm(cities_coords=cities_coords, hz=hz)
    logger.info("[OSRM] %d points bruts", len(df))

    # 5) Pipeline unique (tous traitements dans simulator/pipeline/pipeline.py)
    pipe = SimulationPipeline(full_config)
    df = pipe.run(df)

    # 6) Sécuriser les timestamps
    df = ensure_strictly_increasing_timestamps(df)

    # 7) Contrôle de réalisme (simulate_and_check réutilisé via check_realism)
    realism_results, realism_logs = check_realism(df, timestamp=timestamp)
    logger.info("\n=== Résumé du contrôle de réalisme ===")
    for label, ok in realism_results.items():
        logger.info(f"{label:40} : {'✅ OK' if ok else '❌ À vérifier'}")
    logger.info(f"[INFO] Logs détaillés : {realism_logs['summary']} / {realism_logs['errors']}")

    # 8) Exports & visualisations
    export_trace(df, output_dir=output_dir, timestamp=timestamp, full_config=full_config)


if __name__ == "__main__":
    main()