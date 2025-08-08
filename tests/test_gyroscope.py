import logging
import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

def test_generate_gyroscope_signals():
    try:
        from core.gyroscope import generate_gyroscope_signals
    except Exception as exc:
        pytest.skip(f"generate_gyroscope_signals indisponible: {exc}")

    n = 200
    df = pd.DataFrame({
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": np.full(n, 9.81),
        "speed": np.linspace(0, 20, n),
    })
    out = generate_gyroscope_signals(df, hz=10)
    for col in ("gyro_x","gyro_y","gyro_z"):
        assert col in out.columns
        assert np.isfinite(out[col]).all()