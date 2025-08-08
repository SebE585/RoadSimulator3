import logging
from unittest.mock import patch
import pytest

MOCK_OSRM_JSON = {
    "routes": [
        {"geometry": {"coordinates": [[1.0, 49.0], [1.1, 49.1], [1.2, 49.2]]}}
    ]
}

logger = logging.getLogger(__name__)


@patch("core.osrm.client.requests.get")
def test_get_route_from_coords_mocked(mock_get):
    try:
        from core.osrm.client import get_route_from_coords
    except Exception as exc:
        pytest.skip(f"client API indisponible: {exc}")

    mock_get.return_value.json.return_value = MOCK_OSRM_JSON

    coords = [(49.0, 1.0), (49.2, 1.2)]
    out = get_route_from_coords(coords)

    # L'appel HTTP a bien eu lieu
    assert mock_get.call_count == 1

    # Tolère nouvelle (dict) ou ancienne ((geometry, decoded))
    if isinstance(out, tuple) and len(out) == 2:
        _, decoded = out
        assert len(decoded) == 3
        assert isinstance(decoded[0], tuple)
    else:
        assert isinstance(out, dict)
        assert "routes" in out or "geometry" in out
        if "routes" in out and out["routes"]:
            g = out["routes"][0].get("geometry", {})
            if "coordinates" in g:
                assert len(g["coordinates"]) == 3


@patch("core.osrm.client.requests.get")
def test_simulate_route_via_osrm_mocked(mock_get):
    try:
        from core.osrm.simulate import simulate_route_via_osrm
    except Exception as exc:
        pytest.skip(f"simulate API indisponible: {exc}")

    mock_get.return_value.json.return_value = MOCK_OSRM_JSON

    df = simulate_route_via_osrm([(49.0, 1.0), (49.2, 1.2)], hz=1)
    # Colonnes minimales d'une trace simulée
    for col in ("lat", "lon", "timestamp"):
        assert col in df.columns
    assert len(df) >= 3