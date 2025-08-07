# core/osmnx/mapping.py

"""
Module de correspondance des tags OSM 'highway' vers des types de route standardisés
utilisés dans les simulations RoadSimulator3.
"""

HIGHWAY_TO_TYPE = {
    # Routes principales
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "motorway",
    "trunk_link": "motorway",

    # Routes secondaires
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",

    # Résidentiel et local
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "residential",
    "road": "residential",  # Tag fourre-tout non spécialisé

    # Voies de desserte et service
    "service": "service",
    "track": "service",
    "path": "service",
    "footway": "service",
    "cycleway": "service",
    "pedestrian": "service",
    "steps": "service",

    # Zones spécifiques
    "construction": "other",
    "bus_guideway": "other",
    "escape": "other",
    "raceway": "other",
    "rest_area": "other",

    # Spécial
    "bridleway": "service",
    "corridor": "service",
    "platform": "service",
    "proposed": "other",
    "busway": "service",
    "elevator": "service",
    "service_area": "service",
}


def get_edge_type_nearest(tags: dict) -> str:
    """
    Extrait un type de route harmonisé à partir des tags d’une arête OSM.

    Cette fonction vise à normaliser la classification des routes
    pour les analyses et visualisations. Elle s’appuie sur une
    correspondance définie dans le dictionnaire HIGHWAY_TO_TYPE.

    Args:
        tags (dict): Dictionnaire OSM de tags, typiquement reçu depuis
                     le serveur d’enrichissement (ex: {'highway': 'residential'}).

    Returns:
        str: Type de route harmonisé (ex: 'primary', 'residential', 'other', 'unknown').
    """
    if not isinstance(tags, dict):
        return "unknown"

    highway = tags.get("highway")
    if not highway or not isinstance(highway, str):
        return "unknown"

    return HIGHWAY_TO_TYPE.get(highway, "other")
