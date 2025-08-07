import os
import pandas as pd
import folium
from pathlib import Path
from datetime import datetime


class TraceDebugger:
    def __init__(self, base_dir=None, prefix="step"):
        from core.utils import get_latest_simulation_dir
        default_dir = get_latest_simulation_dir()
        self.base_dir = Path(base_dir) if base_dir else Path(default_dir or "data/debug_traces")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.step = 0

    def save(self, df, label=""):
        if not {"lat", "lon"}.issubset(df.columns):
            print("[TraceDebugger] Aucune colonne lat/lon détectée.")
            return

        trace_path = self.base_dir / f"{self.prefix}_{self.step:02d}_{label}.csv"
        map_path = self.base_dir / f"{self.prefix}_{self.step:02d}_{label}.html"

        df[["lat", "lon"]].to_csv(trace_path, index=False)
        coords = df[["lat", "lon"]].dropna().values.tolist()

        fmap = folium.Map(location=coords[len(coords) // 2], zoom_start=12)
        folium.PolyLine(coords, color="red", weight=4, opacity=0.8, tooltip=f"Trace {label}").add_to(fmap)
        folium.Marker(coords[0], icon=folium.Icon(color='green'), tooltip="Départ").add_to(fmap)
        folium.Marker(coords[-1], icon=folium.Icon(color='darkred', icon='flag'), tooltip="Arrivée").add_to(fmap)

        fmap.save(str(map_path))
        print(f"[TraceDebugger] Étape {self.step} sauvegardée : {map_path}")
        self.step += 1
