"""Injection centralisée des événements inertiels simulés dans un DataFrame de trajectoire.

Ce module contient une fonction unique `inject_inertial_events()` qui applique successivement
tous les générateurs d’événements inertiels définis dans le module :
- Accélération (acceleration)
- Freinage (freinage)
- Dos d’âne (dos_dane)
- Choc trottoir (trottoir)
- Nid de poule (nid_de_poule)
- Arrêt prolongé (stop)
- Attente moteur tournant (wait)

Chaque générateur respecte les contraintes du fichier `config/events.yaml`.

Logs de progression et de succès/échec sont générés automatiquement pour assurer la traçabilité.
"""

import logging
from simulator.events.initial_final import inject_initial_acceleration, inject_final_deceleration
from simulator.events.generation import generate_acceleration  # etc. (à compléter)
from simulator.events.stops_and_waits import generate_stops, generate_waits
from simulator.events.utils import ensure_event_column_object

logger = logging.getLogger(__name__)


def inject_events_on_route(df, hz=10, max_events_per_type=5):

    print("[INJECTION] Application de l'accélération initiale...")
    df = inject_initial_acceleration(df)

    print("[INJECTION] Injection d'accélérations vives...")
    df = generate_acceleration(df, hz=hz, max_events=max_events_per_type)

    print("[INJECTION] Injection de freinages brusques...")
    df = generate_freinage(df, hz=hz, max_events=max_events_per_type)

    print("[INJECTION] Injection de dos d'âne...")
    df = generate_dos_dane(df, max_events=max_events_per_type)

    print("[INJECTION] Injection de chocs trottoir...")
    df = generate_trottoir(df, max_events=max_events_per_type)

    print("[INJECTION] Injection de nids de poule...")
    df = generate_nid_de_poule(df, max_events=max_events_per_type)

    print("[INJECTION] Application de la décélération finale...")
    df = inject_final_deceleration(df)

    print("[INJECTION] Tous les événements injectés avec succès.")
    return df
