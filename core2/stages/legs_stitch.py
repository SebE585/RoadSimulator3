# core2/stages/legs_stitch.py
from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

logger = logging.getLogger(__name__)

def _haversine_series_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(p1) * np.cos(p2) * np.sin(dlmb/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

class LegsStitch:
    """
    Concatène les legs routés, pose un axe temps continu, puis remaillage à hz.
    ✨ Insertion d'arrêts (service) après CHAQUE leg, avec position figée et
       durée exactement égale au 'service_s' du stop de destination.
       => La vitesse retombe à 0 pendant ces fenêtres après remaillage.
    """
    name = "LegsStitch"

    def run(self, ctx: Context) -> Result:
        traces = ctx.artifacts.get("legs_traces", [])
        if not traces:
            return Result((False, "legs_traces manquant"))

        # Cadence visée
        hz = int(ctx.cfg.get("sim", {}).get("hz", 10))
        step_ms = int(round(1000.0 / max(hz, 1)))
        if step_ms <= 0:
            step_ms = 100  # fallback 10 Hz

        # ---- Récup plan (stops + start time) ----
        plan = ctx.artifacts.get("legs_plan", {}) or {}
        stops = plan.get("stops", []) or []  # liste ordonnée des stops du YAML
        start_iso = plan.get("start_time_utc")
        # Durée par défaut si 'service_s' absent
        ls_cfg = ctx.cfg.get("legs_stitch", {}) if isinstance(ctx.cfg, dict) else {}
        default_service_s = float(ls_cfg.get("inject_stop_hold_s", 5.0))

        # Start time
        if start_iso:
            t0 = pd.to_datetime(start_iso, utc=True)
        else:
            t0 = pd.Timestamp.utcnow().tz_localize("UTC")

        # Durée totale de déplacement (mouvement) d'après OSRM si dispo
        summaries = ctx.artifacts.get("legs_summary", [])
        total_move_dur_s = None
        try:
            if summaries and isinstance(summaries, list):
                total_move_dur_s = float(sum(float(s.get("duration_s", 0.0)) for s in summaries))
        except Exception:
            total_move_dur_s = None

        # ---- Concat avec insertion d'arrêts après CHAQUE leg ----
        # Convention: leg i relie stop i -> stop i+1
        # On insère un bloc "service" au stop de destination (i+1), durée = stops[i+1].service_s
        frames = []
        hold_specs = []   # collecte diagnostique [(stop_id, seconds), ...]
        hold_id = 0

        n_legs = len(traces)
        n_stops = len(stops)  # attendu: n_stops = n_legs + 1 (si plan complet)

        for i, leg in enumerate(traces):
            # géo brute
            g = leg[["lat", "lon"]].copy()
            g["is_hold"] = False
            g["hold_id"] = -1
            g["stop_id"] = ""
            g["event"] = ""
            frames.append(g)

            # stop de destination = index stop (i+1) si présent
            dest_idx = i + 1
            if dest_idx < n_stops:
                stop = stops[dest_idx] or {}
                stop_id = str(stop.get("id", f"STOP_{dest_idx}"))
                service_s = float(stop.get("service_s", default_service_s) or 0.0)
            else:
                # Si on ne connaît pas le stop, on peut ignorer ou appliquer défaut 0
                stop_id = f"STOP_{dest_idx}"
                service_s = 0.0

            if service_s > 0:
                # Utilise le dernier point du leg i comme position d'arrêt
                last_lat = float(g.iloc[-1]["lat"])
                last_lon = float(g.iloc[-1]["lon"])
                n_hold = max(1, int(round(service_s * hz)))

                hold_df = pd.DataFrame({
                    "lat": [last_lat] * n_hold,
                    "lon": [last_lon] * n_hold,
                    "is_hold": [True] * n_hold,
                    "hold_id": [hold_id] * n_hold,
                    "stop_id": [stop_id] * n_hold,
                    "event": ["STOP"] * n_hold,  # on marque clairement l'arrêt service
                })
                frames.append(hold_df)
                hold_specs.append((stop_id, float(service_s)))
                hold_id += 1

        df = pd.concat(frames, ignore_index=True)

        # ---- Distances brutes ----
        lat_raw = df["lat"].to_numpy(dtype=float)
        lon_raw = df["lon"].to_numpy(dtype=float)
        n = len(df)
        dmeters = np.zeros(n, dtype=float)
        if n > 1:
            dmeters[1:] = _haversine_series_m(lat_raw[:-1], lon_raw[:-1], lat_raw[1:], lon_raw[1:])

        # ---- Séparation mouvement / hold ----
        is_hold = df["is_hold"].to_numpy(dtype=bool)
        move_mask = ~is_hold
        total_m_move = float(np.sum(dmeters[move_mask]))
        if total_m_move <= 0:
            # Sécurité si la géo est dégénérée
            total_m_move = float(max(n - 1, 1))
            tmp = np.zeros_like(dmeters)
            if n > 1:
                tmp[1:] = 1.0
            dmeters = tmp

        # ---- Durées globales ----
        # Mouvement (OSRM) sinon fallback "1 s par pas mouvement"
        if total_move_dur_s is None or total_move_dur_s <= 0:
            total_move_dur_s = float(max(np.sum(move_mask) - 1, 1))
            ctx.artifacts["legs_stitch_duration_fallback_s"] = total_move_dur_s

        # HOLDs: somme des service_s réellement injectés
        total_hold_dur_s = 0.0
        if hold_specs:
            total_hold_dur_s = float(sum(s for _, s in hold_specs))

        total_dur_s = float(total_move_dur_s + total_hold_dur_s)

        # ---- Répartition temporelle pas à pas ----
        # - Mouvement: proportionnel à la distance de chaque pas
        # - HOLD: exactement service_s, réparti sur le nb d'échantillons du bloc
        time_s = np.zeros(n, dtype=float)

        # Prépare durée par hold_id
        # On re-dérive la durée réelle de chaque bloc à partir du nombre d'échantillons et du service_s voulu
        #  => dt pas à pas = service_s / n_points_de_ce_bloc
        hold_dt_map: dict[int, float] = {}
        if is_hold.any():
            # calcule pour chaque hold_id le nombre d'échantillons du bloc
            counts = df.loc[is_hold, "hold_id"].value_counts().to_dict()
            # reconstruire un dict hold_id -> service_s à partir des stop_id présents
            # plus simple: regrouper par hold_id et prendre le stop_id du bloc
            grp = df.loc[is_hold, ["hold_id", "stop_id"]].drop_duplicates(subset=["hold_id"])
            service_by_stopid = {sid: sec for sid, sec in hold_specs}  # stop_id -> seconds
            for _, row in grp.iterrows():
                hid = int(row["hold_id"])
                sid = str(row["stop_id"])
                cnt = int(counts.get(hid, 0))
                if cnt > 0:
                    service_s = float(service_by_stopid.get(sid, default_service_s))
                    hold_dt_map[hid] = service_s / float(cnt)

        for i in range(1, n):
            if is_hold[i]:
                hid = int(df.iloc[i]["hold_id"])
                dt_i = float(hold_dt_map.get(hid, default_service_s))
                time_s[i] = time_s[i-1] + dt_i
            else:
                di = float(dmeters[i])
                frac = (di / total_m_move) if total_m_move > 0 else 0.0
                dt_i = frac * float(total_move_dur_s)
                time_s[i] = time_s[i-1] + dt_i

        # ---- Index temps ----
        t_index = pd.to_datetime(t0) + pd.to_timedelta(time_s, unit="s")
        df = df.set_index(t_index)
        df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"

        # ---- Remaillage exact à hz sur la durée totale théorique ----
        n_samples = int(np.floor(total_dur_s * hz)) + 1
        target_index = pd.date_range(
            start=df.index[0],
            periods=n_samples,
            freq=pd.to_timedelta(step_ms, unit="ms")
        )

        # Interpoler uniquement les colonnes numériques; propager les colonnes non numériques (event/stop_id...)
        union_index = df.index.union(target_index)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        obj_cols = [c for c in df.columns if c not in num_cols]

        # Numeric -> time interpolation
        if num_cols:
            num_part = df[num_cols].reindex(union_index).interpolate(method="time")
            num_part = num_part.reindex(target_index)
        else:
            num_part = pd.DataFrame(index=target_index)

        # Objects -> forward/back fill (no time interpolate on object dtype)
        if obj_cols:
            obj_part = df[obj_cols].reindex(union_index).ffill().bfill().reindex(target_index)
        else:
            obj_part = pd.DataFrame(index=target_index)

        # Recombine and ensure index name survives the ops
        df = pd.concat([num_part, obj_part], axis=1)
        df = df.loc[target_index]
        df.index.name = "timestamp"

        # ---- Vitesse depuis lat/lon + dt réel ----
        lat = df["lat"].to_numpy(dtype=float)
        lon = df["lon"].to_numpy(dtype=float)
        ns = df.index.asi8
        tsec = (ns - ns[0]) / 1e9
        dmeters = np.zeros_like(lat, dtype=float)
        if len(lat) > 1:
            dmeters[1:] = _haversine_series_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt_s = np.diff(tsec, prepend=tsec[0])
        pos = dt_s > 0
        median_dt = float(np.median(dt_s[pos])) if pos.any() else (1.0 / max(hz, 1))
        dt_s[~pos] = median_dt
        speed = np.maximum(dmeters / dt_s, 0.0)

        # Lissage léger num.
        k = max(1, int(round(0.3 * hz)))
        if k > 1:
            speed = pd.Series(speed, index=df.index)\
                    .rolling(window=k, center=True, min_periods=1)\
                    .median().to_numpy()

        df["speed"] = speed

        # ---- Sortie minimale requise par les stages suivants ----
        out = df.reset_index()
        # s'assurer que le nom d'index a bien été conservé
        if "timestamp" not in out.columns:
            out = out.rename(columns={out.columns[0]: "timestamp"})

        base_cols = [c for c in ["timestamp", "lat", "lon", "speed"] if c in out.columns]
        out = out[base_cols].copy()

        # on garde les annotations utiles si présentes
        for extra in ("event", "stop_id"):
            if extra in df.columns and extra not in out.columns:
                out[extra] = df[extra].to_numpy()

        ctx.df = out
        ctx.meta["hz"] = hz
        ctx.meta["hz_expected"] = float(hz)
        ctx.meta["samples_expected"] = int(n_samples)
        ctx.meta["duration_expected_s"] = float(total_dur_s)

        # Diagnostics : liste des arrêts réellement injectés
        ctx.artifacts["legs_stitch_holds"] = {
            "count": len(hold_specs),
            "items": [{"stop_id": sid, "service_s": sec} for sid, sec in hold_specs],
            "total_hold_s": total_hold_dur_s,
        }

        return Result((True, "OK"))