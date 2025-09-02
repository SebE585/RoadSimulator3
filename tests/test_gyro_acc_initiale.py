

import pandas as pd
import numpy as np
import pytest

from simulator.events.gyro import generate_gyroscope_signals


def make_df_with_event(event: str, n: int = 20, hz: int = 10):
    """Utility: build a DataFrame with constant heading, speed, and one event at center."""
    idx = pd.date_range("2025-01-01", periods=n, freq=f"{int(1000/hz)}L")
    heading = np.linspace(0, 0.1, n)  # small change in heading
    df = pd.DataFrame({
        "timestamp": idx,
        "lat": np.linspace(48.85, 48.86, n),
        "lon": np.linspace(2.35, 2.36, n),
        "heading": heading,
        "speed": np.linspace(10, 11, n),
        "event": [None] * n,
    })
    center = n // 2
    df.at[center, "event"] = event
    return df


def test_acceleration_initiale_produces_gyro_columns():
    df = make_df_with_event("acceleration_initiale", n=30)
    out = generate_gyroscope_signals(df, hz=10)

    # All three gyro columns must exist
    for col in ("gyro_x", "gyro_y", "gyro_z"):
        assert col in out.columns
        assert out[col].dtype == "float32"

    # Ensure values are not all zero
    assert out["gyro_x"].abs().sum() > 0
    assert out["gyro_z"].abs().sum() > 0

    # At least in the window around the event, gyro_x and gyro_z should be nonzero
    center = len(out) // 2
    window = 5
    sub = out.iloc[center - window:center + window + 1]
    assert sub["gyro_x"].abs().sum() > 0
    assert sub["gyro_z"].abs().sum() > 0


def test_other_events_still_work():
    for ev in ["dos_d_ane", "freinage", "acceleration", "trottoir", "nid_de_poule"]:
        df = make_df_with_event(ev, n=20)
        out = generate_gyroscope_signals(df, hz=10)
        assert {"gyro_x", "gyro_y", "gyro_z"}.issubset(out.columns)
        # ensure at least one column got modified by the event
        assert out[["gyro_x", "gyro_y", "gyro_z"]].abs().sum().sum() > 0