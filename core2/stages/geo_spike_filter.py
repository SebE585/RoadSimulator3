from __future__ import annotations
import numpy as np
import pandas as pd
from ..contracts import Result
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

class GeoSpikeFilter:
    """
    Supprime les sauts GPS irréalistes en bornant la distance entre échantillons
    par une vitesse maximale raisonnable.
    - Ne crée pas de points.
    - Ne suppose pas 1 Hz : dt est calculé depuis 'timestamp' (UTC).

    Paramètres
    ----------
    vmax_kmh : float
        Vitesse maximale admissible (km/h). Défaut: 180.
    hard_jump_m : float
        Seuil "hard" (m) au-delà duquel on supprime le saut, quel que soit dt. Défaut: 500 m.
    soft_margin_m : float
        Marge douce (m) ajoutée à la distance permise par la vitesse. Défaut: 30 m.
    """
    name = "GeoSpikeFilter"

    def __init__(self, vmax_kmh: float = 180.0, hard_jump_m: float = 500.0, soft_margin_m: float = 30.0) -> None:
        self.vmax_kmh = float(vmax_kmh)
        self.hard_jump_m = float(hard_jump_m)
        self.soft_margin_m = float(soft_margin_m)

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")
        if "timestamp" not in df.columns:
            return Result(ok=False, message="timestamp manquant")
        if not {"lat", "lon"}.issubset(df.columns):
            return Result(ok=False, message="lat/lon manquants")

        out = df.copy()

        def _recompute_speed_inplace(frame: pd.DataFrame) -> None:
            # Recalcule 'speed' (m/s) depuis lat/lon et le temps (timestamp UTC)
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            ns = ts.astype("int64").to_numpy()
            tsec = (ns - ns[0]) / 1e9
            lat_c = frame["lat"].to_numpy(dtype=float)
            lon_c = frame["lon"].to_numpy(dtype=float)
            dmeters = np.zeros_like(lat_c, dtype=float)
            if len(lat_c) > 1:
                dmeters[1:] = _haversine_series_m(lat_c[:-1], lon_c[:-1], lat_c[1:], lon_c[1:])
            dt_s = np.diff(tsec, prepend=tsec[0])
            pos = dt_s > 0
            median_dt = float(np.median(dt_s[pos])) if pos.any() else 0.1
            dt_s[~pos] = median_dt
            speed = dmeters / dt_s
            frame["speed"] = np.maximum(speed, 0.0)

        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            return Result(ok=False, message="timestamps invalides")

        ns = ts.astype("int64").to_numpy()
        tsec = (ns - ns[0]) / 1e9
        dt = np.diff(tsec, prepend=tsec[0])
        # dt nul ou négatif -> on prend un pas médian pour éviter les infinities
        valid = dt > 0
        median_dt = float(np.median(dt[valid])) if valid.any() else 0.1
        dt[~valid] = median_dt

        lat = out["lat"].to_numpy(dtype=float, copy=True)
        lon = out["lon"].to_numpy(dtype=float, copy=True)

        dmeters = np.zeros_like(lat)
        dmeters[1:] = _haversine_series_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

        vmax_mps = self.vmax_kmh * (1000.0/3600.0)
        # Distance permise par la vitesse + marge douce, puis bornée par un CAP "dur"
        permitted = vmax_mps * dt + self.soft_margin_m
        max_step = np.minimum(permitted, self.hard_jump_m)

        spikes = (dmeters > max_step)
        n_spikes = int(spikes.sum())

        # Debug: top 5 plus grands sauts (avant correction)
        if n_spikes > 0:
            try:
                top_idx = np.argsort(dmeters)[-5:][::-1]
                ctx.artifacts["geo_spikes_top"] = [
                    {
                        "i": int(i),
                        "d_m": float(dmeters[i]),
                        "max_step_m": float(max_step[i]),
                        "dt_s": float(dt[i]),
                    }
                    for i in top_idx
                ]
            except Exception:
                pass

        ctx.artifacts["geo_spikes_n_raw"] = n_spikes

        if n_spikes > 0:
            idx = np.where(spikes)[0]

            # 1) Première passe: clamp au point précédent pour supprimer le saut brut
            for i in idx:
                lat[i] = lat[i-1]
                lon[i] = lon[i-1]

            # 2) Seconde passe: interpolation douce pour les blocs de spikes consécutifs
            runs = []
            start = None
            for i in range(1, len(spikes)):
                if spikes[i] and not spikes[i-1]:
                    start = i
                if start is not None and (i == len(spikes)-1 or (spikes[i] and not spikes[i+1])):
                    end = i
                    runs.append((start, end))
                    start = None

            for a, b in runs:
                i0 = a - 1  # dernier point valide avant le run
                # premier point non-spike après le run ; s'il n'existe pas, on s'arrête au dernier index
                has_next = (b + 1) < len(lat)
                i1 = (b + 1) if has_next else b
                # Conditions minimales pour interpoler : points bornes valides et une vraie tranche à remplir
                if i0 >= 0 and has_next and i1 > a:
                    # Longueur exacte de la tranche à remplir (points internes) : [a, i1) → n_fill
                    n_fill = i1 - a
                    # Nombre de segments entre i0 (inclus) et i1 (inclus)
                    steps = i1 - i0
                    if steps > 1 and n_fill > 0:
                        # Alphas internes: 1/steps, 2/steps, ..., n_fill/steps  (n_fill valeurs)
                        alphas = np.linspace(1.0/steps, n_fill/steps, n_fill, dtype=float)
                        lat[a:i1] = (1.0 - alphas) * lat[i0] + alphas * lat[i1]
                        lon[a:i1] = (1.0 - alphas) * lon[i0] + alphas * lon[i1]
                else:
                    # Pas de point suivant fiable : on reste au clamp (déjà appliqué en passe 1)
                    pass

            out["lat"] = lat
            out["lon"] = lon
            out["flag_geo_spike"] = 0
            out.loc[idx, "flag_geo_spike"] = 1
            ctx.artifacts["geo_spikes_n"] = n_spikes

            # Recalcul de la vitesse après correction
            _recompute_speed_inplace(out)
        else:
            out["flag_geo_spike"] = 0
            ctx.artifacts["geo_spikes_n"] = 0

            # Assure la cohérence de la vitesse même sans correction
            if "speed" not in out.columns or out["speed"].isna().any():
                _recompute_speed_inplace(out)

        ctx.df = out
        return Result()
