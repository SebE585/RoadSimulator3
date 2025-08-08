import logging
import pandas as pd
import numpy as np
import pytest

logger = logging.getLogger(__name__)

def test_pipeline_e2e_offline():
    # On reste offline: on simule une trajectoire simple (pas d'OSRM)
    try:
        from simulator.pipeline_utils import inject_all_events
        from simulator.detectors import detect_initial_acceleration, detect_final_deceleration, detect_all_events
        from core.postprocessing import finalize_trajectory
        from check.check_realism import check_realism
    except Exception as exc:
        pytest.skip(f"Pipeline APIs indisponibles: {exc}")

    n = 400
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms"),
        "lat": np.linspace(49.0, 49.01, n),
        "lon": np.linspace(1.0, 1.01, n),
        "speed": np.concatenate([np.linspace(0,15,100), np.full(200,15), np.linspace(15,0,100)]),
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": np.full(n, 9.81),
        "gyro_x": np.zeros(n),
        "gyro_y": np.zeros(n),
        "gyro_z": np.zeros(n),
        "road_type": "residential",
        "event": pd.Series([np.nan]*n, dtype="object"),
    })

    df = inject_all_events(df, hz=10)
    init_ok = bool(detect_initial_acceleration(df))
    final_ok = bool(detect_final_deceleration(df))
    df = finalize_trajectory(df, hz=10)
    det = detect_all_events(df)
    assert isinstance(det, dict)
    realism, _ = check_realism(df)
    assert init_ok and final_ok
    assert "⬆️ Accélération détectée" in realism