import pytest
import pandas as pd

# Ta fonction de mapping (à copier ici ou importer)
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
def sample_edges():
    # Simule un DataFrame comme celui extrait par Pyrosm
    data = {
        'highway': [
            'motorway', 'primary', 'secondary', 'tertiary', 'residential', 
            ['primary', 'secondary'], None, 'service', 'cycleway', 'unclassified'
        ]
    }
    return pd.DataFrame(data)

def test_highway_types_diversity(sample_edges):
    # Vérifier qu'il y a plus d'un type de highway
    unique_types = set()
    for val in sample_edges['highway']:
        if isinstance(val, list):
            unique_types.add(val[0])
        else:
            unique_types.add(val)
    unique_types.discard(None)
    assert len(unique_types) > 1, "Le dataset doit contenir plusieurs types highway différents"

def test_mapping_correctness(sample_edges):
    categories = sample_edges['highway'].apply(map_osm_highway_to_category)
    expected_categories = [
        'autoroute', 'nationale', 'départementale', 'urbaine', 'urbaine',
        'nationale', 'inconnu', 'urbaine', 'chemin', 'autre'
    ]
    assert list(categories) == expected_categories, "Le mapping des types highway est incorrect"

def test_no_unknowns_for_known_types(sample_edges):
    for hw in sample_edges['highway']:
        category = map_osm_highway_to_category(hw)
        # Sauf si hw est None
        if hw is not None:
            assert category != 'inconnu', f"Type {hw} devrait être reconnu"

