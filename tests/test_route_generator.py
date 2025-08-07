import pytest
from core import route_generator

def test_simulate_route_from_towns():
    points, geometry = route_generator.simulate_route_from_towns(n_points=5)
    assert isinstance(points, list)
    assert len(points) > 0
    # Accepter dict avec 'coordinates' clé (GeoJSON)
    assert isinstance(geometry, (dict, list, str))
    assert 'coordinates' in geometry
    assert isinstance(geometry['coordinates'], list)