# helpers/test_osrm_roundabouts.py
import logging
import requests
import pytest

logger = logging.getLogger(__name__)

OSRM_URL = (
    "http://localhost:5001/route/v1/driving/"
    "1.2361,49.3653;1.1906,49.3553;1.1733,49.3364;1.2342,49.3568"
    "?steps=true&annotations=nodes"
)

def _osrm_alive() -> bool:
    try:
        r = requests.get("http://localhost:5001/health", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False

@pytest.mark.network
@pytest.mark.osrm
def test_osrm_roundabouts_endpoint_smoke():
    if not _osrm_alive():
        pytest.skip("OSRM backend indisponible sur localhost:5001 — test ignoré")
    resp = requests.get(OSRM_URL, timeout=2)
    assert resp.status_code == 200