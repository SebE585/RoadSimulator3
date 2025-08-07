"""Point d'entrée principal du module `simulator.events`.

Expose une fonction unifiée `inject_inertial_events(df)` pour injecter l'ensemble des événements inertiels
simulés dans un DataFrame de trajectoire.

Événements pris en charge :
- Accélération (`acceleration`)
- Freinage (`freinage`)
- Dos d’âne (`dos_dane`)
- Choc trottoir (`trottoir`)
- Nid de poule (`nid_de_poule`)
- Arrêt prolongé (`stop`)
- Attente moteur tournant (`wait`)

Les fonctions appelées proviennent des sous-modules `generation.py` et `stops_and_waits.py`.
Chaque injection est traçable via les logs configurés.
"""

import logging

from simulator.events.generation import (
    generate_acceleration,
    generate_freinage,
    generate_dos_dane,
    generate_trottoir,
    generate_nid_de_poule,
)
from simulator.events.stops_and_waits import (
    generate_stops,
    generate_waits,
)

logger = logging.getLogger(__name__)

def inject_inertial_events(df, max_events=5):
    """
    Injecte les événements inertiels (accélération, freinage, dos d'âne, trottoir, nid de poule, stop, wait).

    Args:
        df (pd.DataFrame): dataframe de la trajectoire
        max_events (int): nombre maximum d'événements par type

    Returns:
        df (pd.DataFrame): dataframe enrichi des événements inertiels
    """
    logger.info(f"Injection des événements inertiels avec max_events={max_events} par type.")

    event_generators = [
        ("acceleration", generate_acceleration),
        ("freinage", generate_freinage),
        ("dos_dane", generate_dos_dane),
        ("trottoir", generate_trottoir),
        ("nid_de_poule", generate_nid_de_poule),
        ("stop", generate_stops),
        ("wait", generate_waits),
    ]

    for event_name, generator in event_generators:
        before_count = df['event'].notna().sum()
        df = generator(df, max_events=max_events)
        after_count = df['event'].notna().sum()
        injected = after_count - before_count

        if injected == 0:
            logger.warning(f"Aucun événement '{event_name}' injecté.")
        else:
            logger.info(f"{injected} événements '{event_name}' injectés.")

    return df
