# runner/run_simulation.py
# -*- coding: utf-8 -*-
"""
Runner simple : OSRM → SimulationPipeline → exports + checks.
"""

from __future__ import annotations
import os
import logging
from datetime import datetime

import pandas as pd

from core.config_loader import load_full_config
from core.osrm.simulate import simulate_route_via_osrm
from core.utils import get_simulation_output_dir, ensure_csv_column_order
from core.reports import generate_reports
from check.check_realism import check_realism

from simulator.pipeline.pipeline import SimulationPipeline

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def _build_scenario_stops_df(cities_coords: list[list[float]] | list[tuple[float,float]]):
    """Marqueur simple stop/wait alterné sur les points 'villes'."""
    if not cities_coords:
        return None
    df = pd.DataFrame(cities_coords, columns=["lat", "lon"])
    df["event"] = ["stop" if i % 2 == 0 else "wait" for i in range(len(df))]
    return df


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_full_config()
    outdir = get_simulation_output_dir(ts)

    # 1) Génération OSRM
    coords = cfg.get("simulation", {}).get("cities_coords", [])
    hz = int(cfg.get("simulation", {}).get("hz", 10))
    df = simulate_route_via_osrm(cities_coords=coords, hz=hz)
    scenario_stops = _build_scenario_stops_df(coords)

    # 2) Pipeline
    pipe = SimulationPipeline(cfg)
    df = pipe.run(df, scenario_stops_df=scenario_stops)

    # 3) Exports
    os.makedirs(outdir, exist_ok=True)
    df = ensure_csv_column_order(df)
    csv_path = os.path.join(outdir, "trace.csv")
    df.to_csv(csv_path, index=False)
    log.info("CSV écrit : %s", csv_path)

    # 4) Rapports & réalisme
    generate_reports(df, outdir)
    check_realism(df, timestamp=ts)

    log.info("Dossier complet : %s", outdir)


if __name__ == "__main__":
    main()
