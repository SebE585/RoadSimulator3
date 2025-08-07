import pandas as pd
import matplotlib.pyplot as plt

IGN_CSV = 'out/simulation_20250718_091252/trace_enriched_ign.csv'
SRTM_CSV = 'out/simulation_20250718_091252/trace_enriched_srtm.csv'

# Chargement des données
ign_df = pd.read_csv(IGN_CSV)
srtm_df = pd.read_csv(SRTM_CSV)

# Vérification que les distances sont équivalentes
if not ign_df['distance_m'].equals(srtm_df['distance_m']):
    raise ValueError("Les distances ne sont pas identiques entre IGN et SRTM")

# Plot
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(ign_df['distance_m'], ign_df['altitude'], label='IGN', color='blue')
axs[0].plot(srtm_df['distance_m'], srtm_df['altitude'], label='SRTM', color='green', linestyle='--')
axs[0].set_ylabel('Altitude (m)')
axs[0].set_title('Comparaison des altitudes')
axs[0].legend()

axs[1].plot(ign_df['distance_m'], ign_df['slope_percent'], label='IGN', color='blue')
axs[1].plot(srtm_df['distance_m'], srtm_df['slope_percent'], label='SRTM', color='green', linestyle='--')
axs[1].set_ylabel('Pente (%)')
axs[1].set_xlabel('Distance (m)')
axs[1].set_title('Comparaison des pentes')
axs[1].legend()

plt.tight_layout()
plt.savefig('comparaison_altitude_pente_ign_vs_srtm.png')
plt.show()
