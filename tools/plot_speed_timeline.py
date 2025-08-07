import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# --- Localisation automatique du dernier dossier simulated_* ---
out_dir = 'out'
sim_dirs = [d for d in os.listdir(out_dir) if d.startswith('simulation_')]
latest_sim_dir = sorted(sim_dirs)[-1] if sim_dirs else None

if not latest_sim_dir:
    raise FileNotFoundError("Aucun dossier simulation_* trouvé dans 'out/'")

trace_path = os.path.join(out_dir, latest_sim_dir, 'trace.csv')
print(f"[INFO] Chargement du fichier : {trace_path}")

df = pd.read_csv(trace_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Lisser la courbure
window_size = 10
df['curvature_smooth'] = df['curvature'].rolling(window=window_size, center=True, min_periods=1).mean()

# Ajuster l'échelle de la courbure pour visibilité
curvature_scale_factor = 100  # ajustable selon rendu souhaité
df['curvature_smooth_scaled'] = df['curvature_smooth'] * curvature_scale_factor

# Mapping événements
event_colors = {
    'stop': 'red',
    'wait': 'orange',
    'acceleration_initiale': 'green',
    'acceleration': 'lime',
    'freinage': 'darkred',
    'trottoir': 'brown',
    'dos_dane': 'purple',
    'nid_de_poule': 'black'
}

def get_event_onsets(df, event_name):
    event_mask = df['event'] == event_name
    onset_indices = event_mask & ~event_mask.shift(fill_value=False)
    return df[onset_indices]

fig = go.Figure()

# Vitesse
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['speed'], 
    mode='lines', name='Vitesse (km/h)', line=dict(color='blue')
))

# Courbure lissée (échelle ajustée)
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['curvature_smooth_scaled'],
    mode='lines', name=f'Courbure lissée x{curvature_scale_factor}', line=dict(color='purple', dash='solid')
))

# Accélérations inertielle X, Y, Z
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['acc_x'],
    mode='lines', name='acc_x', line=dict(color='green', dash='dot')
))
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['acc_y'],
    mode='lines', name='acc_y', line=dict(color='orange', dash='dot')
))
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['acc_z'],
    mode='lines', name='acc_z', line=dict(color='grey', dash='dot')
))

# Événements
for event_name, color in event_colors.items():
    subset = get_event_onsets(df, event_name)
    if not subset.empty:
        fig.add_trace(go.Scatter(
            x=subset['timestamp'], y=subset['speed'],
            mode='markers', name=event_name,
            marker=dict(color=color, size=9, symbol='x')
        ))

fig.update_layout(
    title=f'Timeline interactive : Vitesse, Courbure lissée, Accélérations, Événements - {latest_sim_dir}',
    xaxis_title='Temps',
    yaxis_title='Valeurs (vitesse, acc, courbure x{curvature_scale_factor})',
    hovermode='x unified',
    height=800
)

fig.show()
