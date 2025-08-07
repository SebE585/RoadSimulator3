import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_reacceleration_profile(df, window=100):
    """
    Affiche les profils de vitesse et acc_x autour des stops détectés.

    Args:
        df (pd.DataFrame): DataFrame complet.
        window (int): Nombre de points avant et après le stop à afficher.
    """
    stop_indices = df.index[df['event'] == 'stop'].tolist()
    if not stop_indices:
        print("Aucun 'stop' trouvé dans le fichier.")
        return

    plotted = 0
    for stop_idx in stop_indices:
        # Fin du stop
        end_idx = stop_idx
        while end_idx < len(df) and df.at[end_idx, 'event'] == 'stop':
            end_idx += 1

        if end_idx >= len(df):
            continue

        start_plot = max(0, end_idx - window)
        end_plot = min(len(df), end_idx + window)

        sub_df = df.iloc[start_plot:end_plot].copy()
        t = range(len(sub_df))

        plt.figure(figsize=(12, 5))

        plt.subplot(2,1,1)
        plt.plot(t, sub_df['speed'], label='Vitesse (km/h)')
        plt.axvline(x=(end_idx - start_plot), color='red', linestyle='--', label='Fin stop')
        plt.title(f'Profil vitesse autour du stop index={stop_idx}')
        plt.ylabel('Vitesse (km/h)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(t, sub_df['acc_x'], label='Accélération longitudinale (acc_x)')
        plt.axvline(x=(end_idx - start_plot), color='red', linestyle='--', label='Fin stop')
        plt.ylabel('acc_x (m/s²)')
        plt.xlabel('Points (10Hz)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        plotted += 1
        if plotted >= 3:  # Limite à 3 exemples pour ne pas saturer
            break

    if plotted == 0:
        print("Impossible de tracer : fin de stop non trouvée ou index hors limite.")


def main():
    if len(sys.argv) != 2:
        print("Usage : python visualize_stop_reacceleration.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    if 'speed' not in df.columns or 'acc_x' not in df.columns or 'event' not in df.columns:
        print("Le fichier doit contenir les colonnes 'speed', 'acc_x' et 'event'.")
        sys.exit(2)

    plot_reacceleration_profile(df)

if __name__ == "__main__":
    main()
