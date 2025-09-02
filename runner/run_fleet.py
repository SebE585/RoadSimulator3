# runner/run_fleet.py
# -*- coding: utf-8 -*-
"""
Lance N tournées (1 par véhicule) définies dans config/fleet.yaml,
en réutilisant le pipeline complet de run_simulation (RS3DS + enrichissements v1.0).

Usage exemples:
  # 1 véhicule en profil "parcels"
  python -m runner.run_fleet --profile parcels --vehicle VL-01

  # plusieurs véhicules en profil "meubles"
  python -m runner.run_fleet --profile meubles --vehicles VL-01,VL-02

  # forcer la fréquence et activer altitude "demo"
  python -m runner.run_fleet --hz 20 --with-altitude --vehicles VL-01,VL-02

  # choisir un dossier racine pour les sorties de flotte
  python -m runner.run_fleet --profile parcels --vehicles VL-01,VL-02 \
      --output-root data/simulations
"""

from __future__ import annotations

import os
import argparse
import logging
from datetime import datetime
from typing import List

from runner.run_simulation import run_simulation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _normalize_vehicle_list(single: str | None, multi: str | None) -> List[str]:
    vehicles: List[str] = []
    if single:
        vehicles = [single]
    elif multi:
        vehicles = [v.strip() for v in multi.split(",") if v.strip()]
    return vehicles


def run_fleet(
    profile: str,
    vehicles: List[str],
    hz: int = 10,
    count: int | None = None,
    output_root: str | None = None,
    with_altitude: bool = False,
    delivery_buttons: bool | None = None,
    event_categories: bool | None = None,
    target_speed_kmh: float | None = None,
) -> None:
    """Orchestre une simulation par véhicule.

    Notes d'intégration v1.0:
      - Les enrichissements (in_delivery, delivery_state, event_* catégories, altitude_m,
        gyro systématique) sont déjà gérés dans run_simulation / pipeline.
      - Ici on ne fait que passer des *intentions* via variables d'environnement,
        afin que `load_full_config()` et/ou `RS3DS` puissent les lire.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not vehicles:
        raise SystemExit("Aucun véhicule spécifié (use --vehicle ou --vehicles).")

    # Prépare un répertoire racine pour la flotte
    if output_root is None:
        output_root = os.path.join("data", "simulations")
    fleet_root = os.path.join(output_root, f"fleet_{timestamp}")
    os.makedirs(fleet_root, exist_ok=True)

    # Env overrides (laisse la liberté au config_loader/RS3DS de les consommer)
    os.environ.setdefault("RS3_SCHEMA_PATH", "config/dataset_schema.yaml")
    os.environ["RS3_SIM_HZ"] = str(int(hz))
    if target_speed_kmh is not None:
        os.environ["RS3_TARGET_SPEED_KMH"] = str(float(target_speed_kmh))

    # Altitude provider
    if with_altitude:
        # 'demo' donne une courbe douce; change en 'srtm' ou 'ign' si provider dispo
        os.environ["RS3_ALT_PROVIDER"] = os.environ.get("RS3_ALT_PROVIDER", "demo")
    else:
        os.environ.setdefault("RS3_ALT_PROVIDER", "none")

    # Delivery buttons on/off
    if delivery_buttons is not None:
        os.environ["RS3_DELIVERY_BUTTONS"] = "1" if delivery_buttons else "0"

    # Event categories projection on/off
    if event_categories is not None:
        os.environ["RS3_EVENT_CATEGORIES"] = "1" if event_categories else "0"

    logger.info("=== Fleet simulation (profile=%s) vehicles=%s ===", profile, vehicles)

    for vehicle in vehicles:
        # Chaque véhicule possède son sous-dossier
        vehicle_out = os.path.join(fleet_root, vehicle)
        os.makedirs(vehicle_out, exist_ok=True)

        # Indices d'override pour run_simulation/RS3DS (si supportés)
        os.environ["RS3_OUTPUT_DIR"] = vehicle_out
        os.environ["RS3_VEHICLE_ID"] = vehicle
        os.environ["RS3_VEHICLE_PROFILE"] = profile

        logger.info("▶️ Simulation véhicule %s (profil %s) → %s", vehicle, profile, vehicle_out)
        try:
            run_simulation(
                input_csv=None,
                speed_target_kmh=float(os.environ.get("RS3_TARGET_SPEED_KMH", 30 if target_speed_kmh is None else target_speed_kmh)),
                use_rs3ds=True,
            )
        except Exception as e:
            logger.error("Simulation %s échouée (%s)", vehicle, e, exc_info=True)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lancer une flotte de véhicules (v1.0)")
    p.add_argument("--profile", required=True, help="Profil de tournée (parcels, meubles, ...)")
    p.add_argument("--vehicle", help="Identifiant véhicule (ex: VL-01)")
    p.add_argument("--vehicles", help="Liste véhicules séparés par virgules")
    p.add_argument("--hz", type=int, default=10, help="Fréquence en Hz (défaut: 10)")
    p.add_argument("--count", type=int, help="Forcer le nombre de points simulés (si supporté)")
    p.add_argument("--output-root", default=None, help="Racine des sorties (défaut: data/simulations)")

    # Toggles v1.0
    p.add_argument("--with-altitude", action="store_true", help="Activer l'enrichissement altitude (provider=demo par défaut)")
    p.add_argument("--delivery-buttons", dest="delivery_buttons", action="store_true", help="Forcer les marqueurs début/fin de livraison")
    p.add_argument("--no-delivery-buttons", dest="delivery_buttons", action="store_false", help="Désactiver les marqueurs début/fin de livraison")
    p.set_defaults(delivery_buttons=None)  # None → laisse la config décider

    p.add_argument("--event-categories", dest="event_categories", action="store_true", help="Projeter les colonnes par catégories d'événements")
    p.add_argument("--no-event-categories", dest="event_categories", action="store_false", help="Ne pas projeter les catégories d'événements")
    p.set_defaults(event_categories=None)

    p.add_argument("--target-speed-kmh", type=float, default=None, help="Vitesse cible moyenne (km/h)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    vehicles = _normalize_vehicle_list(args.vehicle, args.vehicles)

    run_fleet(
        profile=args.profile,
        vehicles=vehicles,
        hz=args.hz,
        count=args.count,
        output_root=args.output_root,
        with_altitude=bool(args.with_altitude),
        delivery_buttons=args.delivery_buttons,
        event_categories=args.event_categories,
        target_speed_kmh=args.target_speed_kmh,
    )


if __name__ == "__main__":
    main()