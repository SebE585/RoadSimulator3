from core.geo_utils import compute_heading

def test_compute_heading():
    heading = compute_heading(49.0, 1.0, 49.001, 1.001)
    assert 0 <= heading <= 360