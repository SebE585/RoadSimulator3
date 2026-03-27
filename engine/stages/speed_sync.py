from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from rs3_contracts.api import Result
from ..context import Context


def _haversine_series_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(p1) * np.cos(p2) * np.sin(dlmb/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


@dataclass
class SpeedSync:
    """
    Aligne la timeline sur une grille 10 Hz exacte, interpole lat/lon au temps,
    recalcule la vitesse depuis la géo + dt réel, et limite dv/dt pour éviter
    les marches de vitesse irréalistes.
    
    Points clés:
      - Utilise `ctx.meta["duration_expected_s"]` si présent (issu de LegsStitch)
        pour définir la durée exacte; sinon, fallback sur (t1 - t0).
      - Construit une grille régulière exacte à `hz` (défaut 10 Hz).
      - Interpole uniquement les colonnes nécessaires au temps (lat/lon),
        propage les colonnes contextuelles/flags par ffill/bfill.
      - Recalcule la vitesse à partir de lat/lon et du dt réel (index),
        puis applique un limiteur dv/dt doux.
      - Optionnel : force une tête à 0 (si vitesse quasi nulle) pour fiabiliser "start_zero".
      - La détection "start_zero" est plus robuste: elle force la tête à 0 si la médiane &lt; seuil OU si le quantile choisi (ex 80e) &lt; seuil relax.
    """
    hz: float | None = None            # si None -> ctx.meta['hz'] ou 10
    dvdt_max_mps2: float = 3.0         # ~10.8 km/h/s (limiteur doux)
    keep_end_zero: bool = True         # force la fin à 0 si déjà quasi nulle
    tail_window_s: float = 1.0         # fenêtre pour tester la fin quasi nulle
    keep_start_zero: bool = True       # force le début à 0 si déjà quasi nul
    head_window_s: float = 1.0         # fenêtre début pour tester la vitesse quasi nulle
    start_zero_thr_mps: float = 0.6    # seuil m/s (~2.16 km/h) pour "départ nul"
    start_zero_quantile: float = 0.8      # quantile utilisé pour le test (ex: 80e pct)
    start_zero_relax_thr_mps: float = 1.0 # seuil "relax" pour le quantile (m/s)

    name: str = "SpeedSync"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        if "timestamp" not in df.columns or not {"lat", "lon"}.issubset(df.columns):
            return Result(ok=False, message="colonnes manquantes (timestamp/lat/lon)")

        # ---------- Paramètres de cadence ----------
        hz = float(self.hz or ctx.meta.get("hz", 10))
        if hz <= 0:
            hz = 10.0
        dt_nominal = 1.0 / hz
        step_ms = int(round(1000.0 / hz))  # éviter la dérive float
        if step_ms <= 0:
            step_ms = 100  # fallback 10 Hz

        out = df.copy()
        idx = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if idx.isna().any():
            return Result(ok=False, message="timestamps invalides")

        # ---------- 1) Détermination de la durée cible ----------
        t0 = idx.iloc[0]
        t1 = idx.iloc[-1]

        # Durée observée dans le DF actuel
        duration_df_s = max((t1 - t0).total_seconds(), 0.0)

        # Si une durée "attendue" est fournie (LegsStitch), on NE LA PREND QUE
        # si elle est plus longue, afin de ne jamais rogner la trajectoire.
        duration_expected_s = float(ctx.meta.get("duration_expected_s", 0.0))
        if np.isfinite(duration_expected_s) and duration_expected_s > 0:
            duration_target_s = max(duration_df_s, duration_expected_s)
        else:
            duration_target_s = duration_df_s

        # Nombre d'échantillons à hz (inclut t0)
        n = int(round(duration_target_s * hz)) + 1
        n = max(n, 2)

        target_index = pd.date_range(
            start=t0,
            periods=n,
            freq=pd.to_timedelta(step_ms, unit="ms"),
        )

        # Index original sur le temps réel (on ne clippe plus à [t0, t_end] pour ne pas rogner)
        out = out.set_index(idx).sort_index()
        if "timestamp" in out.columns:
            out = out.drop(columns=["timestamp"])

        # --- Interpolation lat/lon sur un index union (ancres + grille) ---
        union_index = out.index.union(target_index)

        # Séries numériques propres
        lat_s = pd.to_numeric(out["lat"], errors="coerce")
        lon_s = pd.to_numeric(out["lon"], errors="coerce")

        # Interpolation temporelle uniquement sur lat/lon
        lat_u = lat_s.reindex(union_index).interpolate(method="time")
        lon_u = lon_s.reindex(union_index).interpolate(method="time")
        lat_i = lat_u.reindex(target_index).ffill().bfill()
        lon_i = lon_u.reindex(target_index).ffill().bfill()

        # Construit le dataframe sur la grille exacte
        grid = pd.DataFrame(index=target_index)
        grid.index = grid.index.tz_convert("UTC")
        grid.index.name = "timestamp"
        grid["lat"] = lat_i.to_numpy(dtype=float)
        grid["lon"] = lon_i.to_numpy(dtype=float)

        # --- Colonnes catégorielles / contextuelles : projection prudente ---
        for cat in ("road_type", "osm_highway"):
            if cat in out.columns:
                tmp = out[cat].astype("object")
                grid[cat] = (
                    tmp.reindex(union_index).ffill()
                       .reindex(target_index).ffill().bfill().fillna("unknown")
                )

        if "target_speed" in out.columns:
            ts_num = pd.to_numeric(out["target_speed"], errors="coerce")
            grid["target_speed"] = (
                ts_num.reindex(union_index).ffill().reindex(target_index).ffill().bfill()
            )

        # Propage les autres colonnes (flags, event, etc.)
        for col in out.columns:
            if col in ("lat", "lon", "road_type", "osm_highway", "target_speed"):
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                grid[col] = (
                    pd.to_numeric(out[col], errors="coerce")
                      .reindex(union_index).ffill()
                      .reindex(target_index).ffill().bfill()
                )
            else:
                grid[col] = (
                    out[col].astype("object").reindex(union_index).ffill()
                        .reindex(target_index).ffill().bfill()
                )

        # ---------- 2) Vitesse depuis lat/lon + dt réel ----------
        lat = grid["lat"].to_numpy(dtype=float)
        lon = grid["lon"].to_numpy(dtype=float)
        ns = grid.index.asi8
        tsec = (ns - ns[0]) / 1e9

        dmeters = np.zeros_like(lat, dtype=float)
        if len(lat) > 1:
            dmeters[1:] = _haversine_series_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

        dt_s = np.diff(tsec, prepend=tsec[0])
        pos = dt_s > 0
        median_dt = float(np.median(dt_s[pos])) if pos.any() else dt_nominal
        dt_s[~pos] = median_dt
        v = dmeters / dt_s

        # ---------- 3) Limiteur dv/dt doux ----------
        if self.dvdt_max_mps2 and self.dvdt_max_mps2 > 0:
            vmax_step = self.dvdt_max_mps2 * dt_s
            v_lim = v.copy()
            for i in range(1, len(v_lim)):
                dv = v_lim[i] - v_lim[i-1]
                if dv > vmax_step[i]:
                    v_lim[i] = v_lim[i-1] + vmax_step[i]
                elif dv < -vmax_step[i]:
                    v_lim[i] = max(0.0, v_lim[i-1] - vmax_step[i])
            v = v_lim

        # ---------- 4) Option: tête forcée à 0 ----------
        if self.keep_start_zero and len(v) >= int(hz * self.head_window_s):
            head_n = max(1, int(round(hz * self.head_window_s)))
            head = v[:head_n]
            med = float(np.nanmedian(head))
            try:
                q = float(np.nanpercentile(head, self.start_zero_quantile * 100.0))
            except Exception:
                q = med
            # Force le départ à 0 si (médiane < seuil strict) OU (quantile < seuil relax)
            if (med < float(self.start_zero_thr_mps)) or (q < float(self.start_zero_relax_thr_mps)):
                v[:head_n] = 0.0

        # ---------- 5) Option: queue forcée à 0 ----------
        if self.keep_end_zero and len(v) > int(hz * self.tail_window_s):
            tail_n = int(hz * self.tail_window_s)
            tail = v[-tail_n:]
            if np.nanmedian(tail) < 0.4:  # ~1.44 km/h
                v[-tail_n:] = 0.0

        grid["speed"] = np.maximum(v, 0.0)
        out = grid.reset_index()
        ctx.df = out  # garder toutes les colonnes enrichies

        # Diagnostics utiles
        ctx.meta["hz"] = int(hz)
        ctx.meta["hz_observed"] = float(hz)
        ctx.meta["samples_after_speed_sync"] = int(len(out))
        ctx.meta["duration_after_speed_sync_s"] = float(duration_target_s)

        return Result((True, "OK"))