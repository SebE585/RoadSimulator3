# runner/run_fleet.py
# -*- coding: utf-8 -*-
"""
Run multi-vehicles with the same SimulationPipeline as simulate_and_check.
Usage:
  python -m runner.run_fleet --config config/fleet.yaml --profile parcels --vehicles VL-01,VL-02 --hz 10 --max-km 120
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import yaml
import numpy as np
import pandas as pd

from core.config_loader import load_full_config
from core.osrm.simulate import simulate_route_via_osrm
from core.utils import ensure_csv_column_order
from core.reports import generate_reports
from check.check_realism import check_realism
from simulator.pipeline.pipeline import SimulationPipeline, PipelineOptions
from simulator.events.utils import marquer_livraisons

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


PROFILE_DEFAULTS = {
    "parcels": {"stops": 70},
    "meubles": {"stops": 20},
}


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_route_coords(depot: Tuple[float, float], waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [tuple(depot)] + [tuple(p) for p in waypoints] + [tuple(depot)]


def _random_points_in_bbox(bbox: List[float], n: int, seed: int | None = None) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    lat_min, lon_min, lat_max, lon_max = bbox
    lats = rng.uniform(lat_min, lat_max, size=n)
    lons = rng.uniform(lon_min, lon_max, size=n)
    return list(zip(lats.tolist(), lons.tolist()))


def _select_vehicles(vehicles_cfg: List[Dict[str, Any]], names: List[str] | None) -> List[Dict[str, Any]]:
    if not names:
        return vehicles_cfg
    name_set = {n.strip() for n in names if n and n.strip()}
    return [v for v in vehicles_cfg if v.get("name") in name_set]


def _make_scenario_stops(cities_coords: List[Tuple[float, float]]) -> pd.DataFrame:
    coords_df = pd.DataFrame(cities_coords, columns=["lat", "lon"])
    coords_df = marquer_livraisons(coords_df, prefix="stop_", start_index=1)
    coords_df["event"] = ["stop" if i % 2 == 0 else "wait" for i in range(len(coords_df))]
    return coords_df


def main():
    ap = argparse.ArgumentParser(description="Simuler des tournées multi-véhicules avec le pipeline commun")
    ap.add_argument("--config", default="config/fleet.yaml", help="YAML flotte (depot, vehicles...)")
    ap.add_argument("--profile", choices=["parcels", "meubles"], default="parcels", help="Profil de tournée")
    ap.add_argument("--hz", type=int, default=None, help="Override Hz pipeline (sinon cfg.simulation.hz)")
    ap.add_argument("--count", type=int, default=None, help="Nombre de stops waypoints à générer (sinon profil)")
    ap.add_argument("--vehicle", action="append", default=[], help="Nom de véhicule à inclure (répétable)")
    ap.add_argument("--vehicles", default=None, help="Liste CSV alternative")
    ap.add_argument("--outdir", default=None, help="Dossier racine de sortie (timestampé si absent)")
    ap.add_argument("--max-km", type=float, default=None, help="Couper l’itinéraire si distance_km estimée > max-km (accélère)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        ap.error(f"Config introuvable: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        fleet_cfg = yaml.safe_load(f)

    fleet_name = fleet_cfg.get("fleet_name", "fleet")
    vehicles_cfg = fleet_cfg.get("vehicles", [])
    if not vehicles_cfg:
        ap.error("Aucun véhicule dans la config.")

    selected_names: List[str] = list(args.vehicle) if args.vehicle else []
    if args.vehicles:
        selected_names += [x.strip() for x in args.vehicles.split(",")]
    selected = _select_vehicles(vehicles_cfg, selected_names if selected_names else None)
    if not selected:
        ap.error("Aucun véhicule ne correspond à la sélection.")

    ts = _now_ts()
    root_out = Path(args.outdir) if args.outdir else Path(f"data/fleets/{fleet_name}_{args.profile}_{ts}")
    root_out.mkdir(parents=True, exist_ok=True)

    # Config globale + pipeline options communes à la flotte
    full_config = load_full_config()
    if args.hz is not None:
        full_config.setdefault("simulation", {})["hz"] = int(args.hz)

    pipe = SimulationPipeline(full_config, options=PipelineOptions(
        hz=int(full_config["simulation"].get("hz", 10)),
        enrich_road_type_before_speed=True,
        enrich_road_type_after_speed=False,
    ))

    prof_def = PROFILE_DEFAULTS.get(args.profile, PROFILE_DEFAULTS["parcels"])
    n_points = int(args.count or prof_def["stops"])

    # Dépôt commun
    depot = tuple(fleet_cfg["depot"]["coord"])

    rows = []
    for v in selected:
        try:
            # Waypoints
            if v.get("waypoints"):
                waypoints = [tuple(x) for x in v["waypoints"]]
                if len(waypoints) > n_points:
                    waypoints = waypoints[:n_points]
                elif len(waypoints) < n_points:
                    k = n_points - len(waypoints)
                    waypoints = waypoints + waypoints[:k]
            else:
                gen = v.get("generate", {})
                rawbbox = None
                if "bbox" in gen and gen["bbox"]:
                    raw = gen["bbox"][0] if isinstance(gen["bbox"][0], list) else gen["bbox"]
                    rawbbox = [float(x) for x in raw]
                if rawbbox is None:
                    raise ValueError(f"{v['name']}: aucun waypoints ni generate.bbox fourni.")
                waypoints = _random_points_in_bbox(rawbbox, n_points, seed=v.get("seed"))

            coords = _build_route_coords(depot, waypoints)

            # Simulation OSRM de la route brute
            hz = int(full_config["simulation"].get("hz", 10))
            df_raw = simulate_route_via_osrm(cities_coords=coords, hz=hz)

            # Faculatif: couper si max-km demandé (accélère les cas extrêmes)
            if args.max_km is not None and "distance_m" in df_raw.columns:
                cum_km = df_raw["distance_m"].fillna(0).cumsum() / 1000.0
                last_idx = int((cum_km <= float(args.max_km)).sum())
                if last_idx > 10:
                    df_raw = df_raw.iloc[:last_idx].reset_index(drop=True)

            # Scénario stop/wait (alternance) depuis waypoints pour ce véhicule
            scenario_df = _make_scenario_stops(coords)

            # Pipeline commun
            df = pipe.run(df_raw, scenario_stops_df=scenario_df)

            # Exports
            veh_out = root_out / v["name"]
            veh_out.mkdir(parents=True, exist_ok=True)
            df = ensure_csv_column_order(df)
            csv_path = veh_out / "trace.csv"
            df.to_csv(csv_path, index=False)

            # Checks & rapports (individuels véhicule)
            check_realism(df, timestamp=_now_ts())
            generate_reports(df, str(veh_out))

            rows.append({
                "vehicle": v["name"],
                "zone": v.get("zone", ""),
                "profile": args.profile,
                "hz": hz,
                "csv": str(csv_path),
                "points": int(len(df)),
            })
            log.info("✅ %s : %s", v["name"], csv_path)
        except Exception as e:
            log.exception("❌ %s failed: %s", v.get("name", "?"), e)
            rows.append({
                "vehicle": v.get("name", "?"),
                "zone": v.get("zone", ""),
                "profile": args.profile,
                "hz": full_config["simulation"].get("hz"),
                "csv": None,
                "error": str(e),
            })

    # Résumé flotte
    summary = pd.DataFrame(rows)
    summary_path = root_out / "fleet_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Résumé flotte → %s", summary_path)


if __name__ == "__main__":
    main()
