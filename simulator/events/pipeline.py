# ✅ simulator/events/pipeline.py

import logging
import numpy as np
from simulator.events.generation import (
    generate_acceleration,
    generate_freinage,
    generate_dos_dane,
    generate_trottoir,
    generate_nid_de_poule
)
from simulator.events.stops_and_waits import generate_stops, generate_waits

logger = logging.getLogger(__name__)

def apply_all_events(df, config: dict = None):
    """
    Injecte tous les événements inertiels et arrêts définis dans le bloc `events` de la config.
    """
    if "event" not in df.columns:
        df["event"] = np.nan  # 🔧 Assure que la colonne existe

    if config is None:
        logger.warning("⚠️ Aucun paramètre config fourni à apply_all_events. Aucun événement injecté.")
        return df

    events_cfg = config.get("events", {})
    if not events_cfg:
        logger.warning("⚠️ Bloc `events` manquant ou vide dans la config. Aucun événement injecté.")
        return df

    for event_name, params in events_cfg.items():
        count = params.get("count", 0)
        if count <= 0:
            continue

        logger.info(f"[EVENT] Injection de {count} événement(s) de type '{event_name}'...")

        try:
            if event_name == "acceleration":
                df = generate_acceleration(df, n=count)
            elif event_name == "freinage":
                df = generate_freinage(df, n=count)
            elif event_name == "dos_dane":
                df = generate_dos_dane(df, n=count)
            elif event_name == "trottoir":
                df = generate_trottoir(df, n=count)
            elif event_name == "nid_de_poule":
                df = generate_nid_de_poule(df, n=count)
            elif event_name == "stops":
                df = generate_stops(df, n=count)
            elif event_name == "waits":
                df = generate_waits(df, n=count)
            else:
                logger.warning(f"❓ Type d’événement inconnu : {event_name}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l’injection de '{event_name}': {e}")

    return df
