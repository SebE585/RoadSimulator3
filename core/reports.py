import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from core.kinematics import compute_distance
from core.utils import ensure_csv_column_order
from core.decorators import deprecated


@deprecated
def plot_profile(df, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['distance_m'], df['altitude'], 'b-', label='Altitude (m)')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Altitude (m)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(df['distance_m'], df['slope_percent'], 'r--', label='Pente (%)')
    ax2.set_ylabel('Pente (%)', color='r')
    ax2.tick_params('y', colors='r')

    plt.title('Profil Altitude et Pente')
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


@deprecated
def plot_event_histogram(df, output_path):
    if "event" not in df.columns or df["event"].dropna().empty:
        print("[WARNING] Aucun événement à tracer dans l’histogramme.")
        return

    event_counts = df["event"].value_counts(dropna=True)

    ax = event_counts.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Répartition des événements inertiels")
    ax.set_ylabel("Nombre d’occurrences")
    ax.set_xlabel("Type d’événement")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@deprecated
def heatmap_acc_xy(df, bins=100, range_g=1.0, output_path=None):
    G = 9.81
    max_val = range_g * G

    mask = (
        (df['acc_x'] >= -max_val) & (df['acc_x'] <= max_val) &
        (df['acc_y'] >= -max_val) & (df['acc_y'] <= max_val)
    )
    acc_x = df.loc[mask, 'acc_x'].values
    acc_y = df.loc[mask, 'acc_y'].values
    distances = df.loc[mask, 'distance_m'].values if 'distance_m' in df.columns else None

    acc_x_mg = acc_x * 1000 / G
    acc_y_mg = acc_y * 1000 / G

    heatmap, xedges, yedges = np.histogram2d(
        acc_x_mg, acc_y_mg, bins=bins,
        range=[[-range_g*1000, range_g*1000], [-range_g*1000, range_g*1000]],
        weights=distances
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmap.T, extent=extent, origin='lower', aspect='auto',
        cmap='jet', norm=LogNorm(vmin=1, vmax=heatmap.max() if heatmap.max() > 0 else 1)
    )
    cbar_label = 'Distance parcourue (m, log)' if distances is not None else 'Occurrences (log)'
    plt.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xlabel('acc_x (mG)')
    ax.set_ylabel('acc_y (mG)')
    ax.set_title(f'Heatmap acc_x vs acc_y pondérée (±{range_g}g)')
    ax.set_xlim(-range_g*1000, range_g*1000)
    ax.set_ylim(-range_g*1000, range_g*1000)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
        print(f"📊 Heatmap accélérations sauvegardée : {output_path}")
    else:
        plt.show()


@deprecated
def speed_analysis(df, output_path=None):
    print("=== Statistiques globales de la vitesse (km/h) ===")
    print(df['speed'].describe())

    if 'road_type' in df.columns and not df['road_type'].isna().all():
        grouped = df.groupby('road_type')['speed']
        print("\n=== Statistiques par type de route ===")
        try:
            stats = grouped.describe()
            print(stats)
        except Exception as e:
            print(f"[ERREUR] Impossible d’afficher les stats par type de route : {e}")
            print("Affichage fallback : moyenne des vitesses par type de route")
            print(grouped.mean())

        plt.figure(figsize=(12, 6))
        sns.violinplot(x='road_type', y='speed', data=df, inner='quartile')
        plt.title('Distribution des vitesses par type de route (Violinplot)')
        plt.ylabel('Vitesse (km/h)')
        plt.xlabel('Type de route')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"📈 Analyse vitesse sauvegardée : {output_path}")
        else:
            plt.show()
    else:
        print("Colonne 'road_type' absente ou vide dans le DataFrame.")


@deprecated
def generate_reports(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    plot_profile_path = os.path.join(outdir, 'profile_altitude_pente.png')
    plot_event_hist_path = os.path.join(outdir, 'histogram_events.png')
    heatmap_path = os.path.join(outdir, 'heatmap_acc_xy.png')
    speed_analysis_path = os.path.join(outdir, 'speed_analysis_violin.png')

    if 'distance_m' not in df.columns:
        df = compute_distance(df)

    plot_profile(df, plot_profile_path)
    plot_event_histogram(df, plot_event_hist_path)
    heatmap_acc_xy(df, output_path=heatmap_path)
    speed_analysis(df, output_path=speed_analysis_path)

    print(f"✅ Rapports graphiques générés dans {outdir}")