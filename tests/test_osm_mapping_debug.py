import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from core.osmnx_utils import map_osm_highway_to_category, get_road_types_batch

def test_map_osm_highway_to_category():
    # Test mapping simple
    assert map_osm_highway_to_category(None) == 'inconnu'
    assert map_osm_highway_to_category('motorway') == 'autoroute'
    assert map_osm_highway_to_category('primary') == 'nationale'
    assert map_osm_highway_to_category('residential') == 'urbaine'
    assert map_osm_highway_to_category('cycleway') == 'chemin'
    assert map_osm_highway_to_category('unknown_type') == 'inconnu'
    # Test list input
    assert map_osm_highway_to_category(['motorway', 'primary']) == 'autoroute'


def test_get_road_types_batch(monkeypatch):
    # Mock response from requests.post for the batch API
    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return [
                {'road_type': 'motorway', 'maxspeed': '130'},
                {'road_type': 'residential', 'maxspeed': '50'},
                {'road_type': None, 'maxspeed': None}
            ]

    def mock_post(*args, **kwargs):
        return MockResponse()

    # Patch requests.post to use the mock_post
    import core.osmnx_utils as osm_utils
    import requests
    monkeypatch.setattr(requests, 'post', mock_post)

    coords_list = [[1.0, 49.0], [2.0, 48.0], [3.0, 47.0]]
    results = osm_utils.get_road_types_batch(coords_list)

    assert len(results) == 3
    assert results[0][0] == 'motorway'
    assert results[0][1] == 'autoroute'
    assert results[0][2] == '130'
    assert results[1][0] == 'residential'
    assert results[1][1] == 'urbaine'
    assert results[1][2] == '50'
    assert results[2][0] == 'inconnu'
    assert results[2][1] == 'inconnu'
    assert results[2][2] is None
