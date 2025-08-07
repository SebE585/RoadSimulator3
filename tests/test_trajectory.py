import pandas as pd
from simulator.trajectory import inject_inertial_noise

def test_inject_inertial_noise():
    df = pd.DataFrame({'speed': [30]*100, 'acc_x': 0, 'acc_y': 0, 'acc_z': 9.81, 'event': [None]*100})
    df = inject_inertial_noise(df)
    assert df['acc_x'].std() > 0
    assert df['acc_y'].std() > 0
    assert df['acc_z'].mean() > 9
