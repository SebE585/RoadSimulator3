import logging
import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

@pytest.fixture()
def base_df():
    n = 300
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms"),
        "lat": np.linspace(49.0, 49.01, n),
        "lon": np.linspace(1.0, 1.01, n),
        "speed": np.full(n, 10.0),
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": np.full(n, 9.81),
        "gyro_x": np.zeros(n),
        "gyro_y": np.zeros(n),
        "gyro_z": np.zeros(n),
        "road_type": "residential",
        "event": pd.Series([np.nan]*n, dtype="object"),
    })
    return df

@pytest.mark.parametrize("evt,gen,det", [
    ("dos_dane",   "generate_dos_dane",   "detect_dos_dane"),
    ("trottoir",   "generate_trottoir",   "detect_trottoir"),
    ("nid_de_poule","generate_nid_de_poule","detect_nid_de_poule"),
])
def test_inject_and_detect_single_event(base_df, evt, gen, det):
    try:
        import simulator.events.generation as G
        import simulator.detectors as D
    except Exception as exc:
        pytest.skip(f"APIs events indisponibles: {exc}")

    gen_fn = getattr(G, gen, None)
    det_fn = getattr(D, det, None)
    if not callable(gen_fn) or not callable(det_fn):
        pytest.skip(f"Fonctions manquantes: {gen} / {det}")

    df = gen_fn(base_df.copy(), max_events=1)
    res = det_fn(df)
    if isinstance(res, tuple):
        detected = bool(res[0])
    else:
        detected = bool(res)
    assert detected, f"{evt} non détecté après injection"
    assert (df["event"] == evt).any()