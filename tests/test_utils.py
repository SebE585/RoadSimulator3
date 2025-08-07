import pandas as pd
from core.utils import export_csv, EXPECTED_COLUMNS

def test_export_csv(tmp_path):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='s'),
        'lat': [49.0]*5,
        'lon': [1.0]*5,
        'heading': [0.0]*5,
        'speed': [10]*5,
        'acc_x': [0.0]*5,
        'acc_y': [0.0]*5,
        'acc_z': [9.81]*5,
        'curvature': [0.0]*5,
        'event': [None]*5,
        'sinuosity': ['ligne droite']*5
    })

    # Réordonne explicitement selon EXPECTED_COLUMNS
    df = df[EXPECTED_COLUMNS]

    path = tmp_path / "trace.csv"
    export_csv(df, path)

    df_exported = pd.read_csv(path)
    expected_columns = EXPECTED_COLUMNS
    assert list(df_exported.columns) == expected_columns
    assert len(df_exported) == 5
