import pandas as pd
from check import check_realism

def test_check_frequency():
    df = pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=10, freq='100ms')})
    assert check_realism.check_frequency(df)

def test_check_gps_jumps():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=5, freq='s'),
        'lat': [49.0]*5, 'lon': [1.0]*5
    })
    assert check_realism.check_gps_jumps(df)
