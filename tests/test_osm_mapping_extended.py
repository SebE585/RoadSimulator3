import pytest
import pandas as pd

def map_osm_highway_to_category(osm_type):
    if not osm_type:
        return 'inconnu'
    if isinstance(osm_type, list):
        osm_type = osm_type[0]

    mapping = {
        'motorway': 'autoroute',
        'motorway_link': 'autoroute',
        'trunk': 'nationale',
        'trunk_link': 'nationale',
        'primary': 'nationale',
        'primary_link': 'nationale',
        'secondary': 'départementale',
        'secondary_link': 'départementale',
        'tertiary': 'urbaine',
        'tertiary_link': 'urbaine',
        'residential': 'urbaine',
        'living_street': 'urbaine',
        'service': 'urbaine',
        'track': 'chemin',
        'path': 'chemin',
        'cycleway': 'chemin',
        'footway': 'chemin',
        'bridleway': 'chemin',
        'rest_area': 'aire_service',
        'services': 'aire_service',
        'unclassified': 'autre'
    }
    return mapping.get(osm_type, 'inconnu')

@pytest.fixture
def autoroute_data():
    # Simule un petit trajet autoroute (tags OSM)
    data = {
        'highway': ['motorway', 'motorway_link', ['motorway', 'service'], 'trunk']
    }
    return pd.DataFrame(data)

@pytest.fixture
def ville_data():
    # Simule un petit trajet centre-ville Rouen (tags OSM)
    data = {
        'highway': ['residential', 'living_street', 'service', ['residential', 'footway'], 'tertiary']
    }
    return pd.DataFrame(data)

def test_mapping_autoroute(autoroute_data):
    categories = autoroute_data['highway'].apply(map_osm_highway_to_category)
    expected = ['autoroute', 'autoroute', 'autoroute', 'nationale']
    assert list(categories) == expected, "Mapping autoroute incorrect"

def test_mapping_ville(ville_data):
    categories = ville_data['highway'].apply(map_osm_highway_to_category)
    expected = ['urbaine', 'urbaine', 'urbaine', 'urbaine', 'urbaine']
    assert list(categories) == expected, "Mapping ville incorrect"

def test_diversity_between_autoroute_and_ville(autoroute_data, ville_data):
    all_data = pd.concat([autoroute_data, ville_data], ignore_index=True)
    categories = all_data['highway'].apply(map_osm_highway_to_category)
    unique_categories = set(categories)
    assert 'autoroute' in unique_categories
    assert 'urbaine' in unique_categories
    assert 'départementale' not in unique_categories  # Pas dans ce jeu de test
