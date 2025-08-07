import pandas as pd
import numpy as np
import pytest

from simulator.events.generation import (
    generate_acceleration,
    generate_freinage,
    generate_dos_dane,
    generate_trottoir,
    generate_nid_de_poule,
)

# Paramètres globaux de test
N = 100
G = 9.81

@pytest.fixture
def base_dataframe():
    return pd.DataFrame({
        'speed': np.linspace(10, 50, N),
        'acc_x': np.zeros(N),
        'acc_y': np.zeros(N),
        'acc_z': np.full(N, G),
        'event': pd.Series([np.nan] * N, dtype=object)
    })

def test_generate_acceleration(base_dataframe):
    df = generate_acceleration(base_dataframe.copy(), max_events=3)
    assert (df['event'] == 'acceleration').sum() > 0

def test_generate_freinage(base_dataframe):
    df = generate_freinage(base_dataframe.copy(), max_events=3)
    assert (df['event'] == 'freinage').sum() > 0

def test_generate_dos_dane(base_dataframe):
    df = generate_dos_dane(base_dataframe.copy(), max_events=2)
    assert (df['event'] == 'dos_dane').sum() > 0

def test_generate_trottoir(base_dataframe):
    df = generate_trottoir(base_dataframe.copy(), max_events=2)
    assert (df['event'] == 'trottoir').sum() > 0

def test_generate_nid_de_poule(base_dataframe):
    df = generate_nid_de_poule(base_dataframe.copy(), max_events=2)
    assert (df['event'] == 'nid_de_poule').sum() > 0
