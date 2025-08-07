import pytest
import pandas as pd
from unittest.mock import patch

import sys
import os

# Ajoute la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.osrm.client import get_route_from_coords
from core.osrm.simulate import simulate_route_via_osrm
from core.osrm.routing import get_osrm_turns, validate_turns

# === Données simulées ===
MOCK_OSRM_JSON = {
    "routes": [{
        "geometry": {
            "coordinates": [
                [1.0, 49.0],
                [1.1, 49.1],
                [1.2, 49.2]
            ]
        }
    }]
}


# === Tests de détection de virages ===

def test_get_osrm_turns():
    coords = [
        (49.0, 1.0),
        (49.0005, 1.0005),
        (49.001, 1.001),
        (49.0015, 1.0015),
        (49.002, 1.002)
    ]
    turns = get_osrm_turns(coords, angle_threshold=20)
    assert isinstance(turns, list)
    for t in turns:
        assert 'location' in t
        assert 'delta_heading' in t


def test_validate_turns():
    df = pd.DataFrame({
        'lat': [49.0] * 10,
        'lon': [1.0 + i * 0.0001 for i in range(10)],
        'acc_y': [0.9] * 10
    })
    turn_indices = [4, 5]
    assert validate_turns(df, turn_indices, threshold=0.8)


# === Tests mockés OSRM ===

@patch("core.osrm.client.requests.get")
def test_get_route_from_coords_mocked(mock_get):
    mock_get.return_value.json.return_value = MOCK_OSRM_JSON
    coords = [(49.0, 1.0), (49.2, 1.2)]
    geometry, decoded = get_route_from_coords(coords)
    assert len(decoded) == 3
    assert isinstance(decoded[0], tuple)


@patch("core.osrm.client.requests.get")
def test_simulate_route_via_osrm_mocked(mock_get):
    mock_get.return_value.json.return_value = MOCK_OSRM_JSON
    df = simulate_route_via_osrm([(49.0, 1.0), (49.2, 1.2)], hz=1, step_m=1000)
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "timestamp" in df.columns
    assert len(df) >= 3
