import pytest
import pandas as pd
import numpy as np
from simulator import events

def make_base_df(n=1000):
    return pd.DataFrame({
        "speed": np.zeros(n),
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": np.full(n, events.G),
        "event": [np.nan]*n
    })

def test_inject_initial_acceleration():
    df = pd.DataFrame({'speed': [0]*100, 'acc_x': 0, 'acc_y': 0, 'acc_z': 9.81, 'event': np.nan})
    df = events.inject_initial_acceleration(df)
    assert (df['speed'].max() > 10)
    assert 'acceleration_initiale' in df['event'].values

def test_inject_final_deceleration():
    df = pd.DataFrame({'speed': [50]*100, 'acc_x': 0, 'acc_y': 0, 'acc_z': 9.81, 'event': np.nan})
    df = events.inject_final_deceleration(df)
    assert (df['speed'].iloc[-1] < 5)
    assert 'deceleration_finale' in df['event'].values

def test_generate_acceleration_event():
    df = make_base_df()
    idx = 100
    df = events.generate_acceleration(df, idx)
    assert "acceleration" in df["event"].values
    assert (df.loc[idx:idx+9, "acc_x"] > 2).all()
