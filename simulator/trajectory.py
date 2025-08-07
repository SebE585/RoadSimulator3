"""
trajectory.py
Ajout de bruit inertiel, interpolation inertielle
"""

import numpy as np

def inject_inertial_noise(df, acc_x_std=0.3, acc_y_std=0.3, acc_z_std=0.3, g=9.81):
    """
    Injecte un bruit inertiel réaliste dans acc_x, acc_y, acc_z
    pour les points où aucun événement n'est présent.

    :param df: DataFrame contenant les colonnes acc_x, acc_y, acc_z, event
    :param acc_x_std: écart-type du bruit sur acc_x
    :param acc_y_std: écart-type du bruit sur acc_y
    :param acc_z_std: écart-type du bruit sur acc_z autour de g
    :param g: gravité (m/s²)
    :return: DataFrame avec acc_x, acc_y, acc_z bruités
    """
    mask = df['event'].isna()

    df.loc[mask, 'acc_x'] = np.random.normal(0, acc_x_std, size=mask.sum())
    df.loc[mask, 'acc_y'] = np.random.normal(0, acc_y_std, size=mask.sum())
    df.loc[mask, 'acc_z'] = np.random.normal(g, acc_z_std, size=mask.sum())

    return df
