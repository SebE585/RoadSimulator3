import logging
import pandas as pd
import numpy as np
import pytest

logger = logging.getLogger(__name__)

def _fake_df(n=20):
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms"),
        "lat": np.linspace(49.0, 49.001, n),
        "lon": np.linspace(1.0, 1.001, n),
        "speed": np.linspace(0, 15, n),
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": np.full(n, 9.81),
        # colonnes gyro requises par l'export
        "gyro_x": np.zeros(n),
        "gyro_y": np.zeros(n),
        "gyro_z": np.zeros(n),
        # colonnes de contexte requises par l'export
        "heading": np.zeros(n),
        "osm_highway": pd.Series([np.nan]*n, dtype="object"),
        "road_type": pd.Series(["residential"]*n, dtype="object"),
        "altitude": np.linspace(100, 101, n),
        "slope_percent": np.zeros(n),
        "curvature": np.zeros(n),
        "sinuosity": np.zeros(n),
        # étiquette d'événement
        "event": pd.Series([np.nan]*n, dtype="object"),
    })

def test_export_csv_roundtrip(tmp_path):
    try:
        from core.utils import export_csv  # suppose une API export_csv(df, path)
    except Exception as exc:
        pytest.skip(f"export_csv indisponible: {exc}")

    df = _fake_df()
    out = tmp_path / "trace.csv"
    export_csv(df, out)

    df2 = pd.read_csv(out)
    expected = ["timestamp","lat","lon","speed","acc_x","acc_y","acc_z","event"]
    for col in expected:
        assert col in df2.columns
    assert len(df2) == len(df)