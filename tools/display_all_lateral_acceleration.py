import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_all_lateral_acceleration_segments(df, curvature_threshold=0.005, window=50):
    """
    Affiche tous les segments avec virage marqué (gauche et droite) en tracé superposé,
    avec acc_y, vitesse et courbure.

    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'acc_y', 'speed', 'curvature', 'timestamp'
        curvature_threshold (float): seuil pour détecter un virage marqué (positif ou négatif)
        window (int): nombre de points avant/après le pic de courbure à extraire

    Returns:
        None (affiche un graphique)
    """
    # Trouver indices des virages à droite et à gauche
    virages_droite = df.index[df['curvature'] > curvature_threshold].tolist()
    virages_gauche = df.index[df['curvature'] < -curvature_threshold].tolist()

    plt.figure(figsize=(14, 8))

    # Plot virages à droite
    for idx in virages_droite:
        start_idx = max(idx - window, 0)
        end_idx = min(idx + window, len(df) - 1)
        segment = df.loc[start_idx:end_idx]
        # Rééchantillonner temps relatif
        time_rel = (segment['timestamp'] - segment['timestamp'].iloc[0]).dt.total_seconds()
        plt.plot(time_rel, segment['acc_y'], color='blue', alpha=0.3, label='Virage droite' if idx == virages_droite[0] else "")

    # Plot virages à gauche
    for idx in virages_gauche:
        start_idx = max(idx - window, 0)
        end_idx = min(idx + window, len(df) - 1)
        segment = df.loc[start_idx:end_idx]
        time_rel = (segment['timestamp'] - segment['timestamp'].iloc[0]).dt.total_seconds()
        plt.plot(time_rel, segment['acc_y'], color='red', alpha=0.3, label='Virage gauche' if idx == virages_gauche[0] else "")

    plt.xlabel('Temps relatif (s)')
    plt.ylabel('Accélération latérale acc_y (m/s²)')
    plt.title(f'Accélérations latérales acc_y autour des virages (> ±{curvature_threshold})')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage : python display_all_lateral_acceleration.py <chemin_vers_trace.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Chargement du fichier : {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    for col in ['acc_y', 'speed', 'curvature', 'timestamp']:
        if col not in df.columns:
            print(f"Erreur : colonne '{col}' absente du fichier.")
            sys.exit(2)

    plot_all_lateral_acceleration_segments(df)

if __name__ == "__main__":
    main()
