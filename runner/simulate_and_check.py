# runner/simulate_and_check.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import argparse
import logging
from datetime import datetime

import pandas as pd

from core.config_loader import load_full_config
from core.osrm.simulate import simulate_route_via_osrm
from core.utils import ensure_csv_column_order
from core.reports import generate_reports
from check.check_realism import check_realism
from simulator.pipeline.pipeline import SimulationPipeline, PipelineOptions
from simulator.events.utils import marquer_livraisons

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Simule et enrichit une trajectoire véhicule")
    parser.add_argument('--csv', type=str, default=None,
                        help="Fichier CSV existant (sinon simulation OSRM)")
    parser.add_argument('--outdir', type=str, default=None,
                        help="Répertoire de sortie (défaut : data/simulations/...)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f"data/simulations/simulated_{ts}"
    os.makedirs(outdir, exist_ok=True)

    cfg = load_full_config()

    if args.csv and os.path.exists(args.csv):
        log.info("📂 Chargement CSV: %s", args.csv)
        df = pd.read_csv(args.csv)
        scenario_stops = None
    else:
        log.info("🛰️ Simulation via OSRM…")
        cities = cfg["simulation"]["cities_coords"]
        hz = int(cfg["simulation"].get("hz", 10))
        df = simulate_route_via_osrm(cities_coords=cities, hz=hz)

        # Préparer le scénario stop/wait alterne depuis les positions (mêmes points que run_simulation)
        coords_df = pd.DataFrame(cities, columns=["lat", "lon"])
        coords_df = marquer_livraisons(coords_df, prefix="stop_", start_index=1)
        coords_df["event"] = ["stop" if i % 2 == 0 else "wait" for i in range(len(coords_df))]
        scenario_stops = coords_df

    pipe = SimulationPipeline(cfg, options=PipelineOptions())
    df = pipe.run(df, scenario_stops_df=scenario_stops)

    # Export & checks
    df = ensure_csv_column_order(df)
    csv_path = os.path.join(outdir, "trace.csv")
    df.to_csv(csv_path, index=False)
    log.info("📤 CSV exporté : %s", os.path.abspath(csv_path))

    check_realism(df, timestamp=ts)
    generate_reports(df, outdir)

    log.info("✅ Rapports générés dans %s", outdir)


if __name__ == "__main__":
    main()
