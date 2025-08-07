import pytest
import requests


@pytest.fixture
def base_urls():
    return {
        "osrm": "http://localhost:5003",
        "osmnx": "http://localhost:5002",
        "srtm": "http://localhost:5004"
    }


def test_osrm_server(base_urls):
    url = f"{base_urls['osrm']}/route/v1/driving/1.1,49.4;1.2,49.5"
    params = {
        "overview": "false",
        "alternatives": "false",
        "steps": "false"
    }
    response = requests.get(url, params=params)
    assert response.status_code == 200, "OSRM API did not return 200 OK"
    data = response.json()
    assert 'routes' in data, "'routes' not in OSRM response"


import requests

def test_osmnx_service(base_urls):
    url = f"{base_urls['osmnx']}/nearest_road_batch"
    payload = [
        {"lat": 49.4, "lon": 1.1},
        {"lat": 49.41, "lon": 1.12},
        {"lat": 49.39, "lon": 1.09}
    ]
    response = requests.post(url, json=payload)
    assert response.status_code == 200, "OSMNX API did not return 200 OK on /nearest_road_batch"
    
    data = response.json()
    assert isinstance(data, list), "Response is not a list"
    assert len(data) == len(payload), "Response list length does not match input"
    for item in data:
        assert 'nearest_node' in item, "Missing 'nearest_node' in response item"
        assert 'road_type' in item, "Missing 'road_type' in response item"




def test_srtm_service(base_urls):
    url = f"{base_urls['srtm']}/enrich_terrain"
    payload = [
        {"lat": 49.4, "lon": 1.1},
        {"lat": 49.5, "lon": 1.2},
        {"lat": 49.6, "lon": 1.3}
    ]
    response = requests.post(url, json=payload)
    assert response.status_code == 200, "SRTM API did not return 200 OK"
    data = response.json()
    assert isinstance(data, list) and len(data) == 3, "SRTM response has invalid format"
    for point in data:
        assert 'altitude' in point, "Missing 'altitude' in SRTM response"
        assert 'slope_percent' in point, "Missing 'slope_percent' in SRTM response"
