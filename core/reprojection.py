import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from core.kinematics import compute_cumulative_distance
from core.decorators import deprecated

logger = logging.getLogger(__name__)


def _compute_step_m(df: pd.DataFrame) -> np.ndarray:
    """Retourne la distance entre points consécutifs (m)."""
    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return np.array([], dtype=float)
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    step = np.zeros(len(lat), dtype=float)
    if len(lat) > 1:
        step[1:] = [haversine_m(lat[i-1], lon[i-1], lat[i], lon[i]) for i in range(1, len(lat))]
    return step



# === Helper: _recompute_distance_columns ===
def _recompute_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """(Utilitaire) Met à jour step_m, delta_m et distance_m depuis lat/lon."""
    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df
    # Reutilise le calcul déjà fait dans _recompute_speed_from_gps qui renseigne step_m
    df = _recompute_speed_from_gps(df)
    if "step_m" in df.columns:
        step = df["step_m"].to_numpy(dtype=float)
        df.loc[:, "delta_m"] = step
        df.loc[:, "distance_m"] = np.cumsum(step)
    return df


def _clamp_gps_jumps(
    df: pd.DataFrame,
    *,
    max_step_m: float = 3.0,
    protect_labels=("stop", "wait"),
    max_iters: int = 20,
    use_adaptive_threshold: bool = True,
    adaptive_gain: float = 1.6,
    adaptive_margin_m: float = 0.5,
) -> pd.DataFrame:
    """Limite les sauts GPS jusqu'à stabilisation.

    Principe:
      - seuil fixe `max_step_m` ET/OU seuil adaptatif basé sur la vitesse locale:
            thr_i = max(max_step_m, adaptive_gain * speed_mps[i] * dt[i] + adaptive_margin_m)
      - ne modifie pas les segments étiquetés (stop/wait…)
      - après correction, lissage médian 3 points sur (lat,lon) hors segments protégés
    """
    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df

    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    n = len(lat)

    # masque de protection (stop/wait)
    ev_cols = [c for c in df.columns if c.lower() in ("event", "events", "tag", "tags", "state")]
    protect_mask = np.zeros(n, dtype=bool)
    if ev_cols:
        ev = df[ev_cols[0]].astype(str).str.lower()
        for L in protect_labels:
            protect_mask |= ev.str.contains(L, na=False).to_numpy()

    # dt et vitesse pour seuil adaptatif
    dt = _compute_dt_seconds(df)
    v_kmh = df["speed_kmh"].to_numpy(dtype=float) if "speed_kmh" in df.columns else np.zeros(n, dtype=float)
    v_mps = v_kmh / 3.6

    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def step_at(i):
        return haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])

    def thr_at(i):
        if not use_adaptive_threshold:
            return max_step_m
        # seuil adaptatif proportionnel au déplacement attendu sur dt[i]
        return max(max_step_m, adaptive_gain * v_mps[i] * dt[i] + adaptive_margin_m)

    it = 0
    while it < max_iters:
        it += 1
        changed = False
        i = 1
        while i < n:
            d = step_at(i)
            if (d > thr_at(i)) and not protect_mask[i] and not protect_mask[i-1]:
                # Corriger i vers le barycentre des voisins (si possible)
                if i + 1 < n:
                    lat[i] = 0.5 * (lat[i-1] + lat[i+1])
                    lon[i] = 0.5 * (lon[i-1] + lon[i+1])
                else:
                    lat[i] = lat[i-1]
                    lon[i] = lon[i-1]

                # Si toujours trop grand, rapprocher encore vers i-1
                if step_at(i) > thr_at(i) and not protect_mask[i-1]:
                    lat[i] = 0.75 * lat[i-1] + 0.25 * lat[i]
                    lon[i] = 0.75 * lon[i-1] + 0.25 * lon[i]

                # Limiter aussi le saut suivant si besoin
                if i + 1 < n and not protect_mask[i+1]:
                    d_next = haversine_m(lat[i], lon[i], lat[i+1], lon[i+1])
                    if d_next > thr_at(i+1):
                        if i + 2 < n:
                            lat[i+1] = 0.5 * (lat[i] + lat[i+2])
                            lon[i+1] = 0.5 * (lon[i] + lon[i+2])
                        else:
                            lat[i+1] = lat[i]
                            lon[i+1] = lon[i]
                changed = True
                i = max(1, i - 1)
                continue
            i += 1

        # Petit lissage médian 3 points hors segments protégés
        if n >= 3:
            lat_sm = lat.copy()
            lon_sm = lon.copy()
            for j in range(1, n - 1):
                if not (protect_mask[j-1] or protect_mask[j] or protect_mask[j+1]):
                    lat_sm[j] = np.median([lat[j-1], lat[j], lat[j+1]])
                    lon_sm[j] = np.median([lon[j-1], lon[j], lon[j+1]])
            lat, lon = lat_sm, lon_sm

        if not changed:
            break

    df.loc[:, "lat"] = lat
    df.loc[:, "lon"] = lon
    return df


def _assign_event_nearest_with_tolerance(df_resampled: pd.DataFrame, df_original: pd.DataFrame, tol_ms: int = 500) -> pd.DataFrame:
    """Assigne la colonne `event` aux timestamps rééchantillonnés par appariement nearest avec tolérance.
    - Pas de propagation illimitée (évite la dilution massive des événements).
    """
    if df_original is None or df_original.empty or "timestamp" not in df_original.columns:
        # nothing to assign
        if "event" not in df_resampled.columns:
            df_resampled["event"] = np.nan
        if df_resampled["event"].dtype != "object":
            df_resampled["event"] = df_resampled["event"].astype("object")
        return df_resampled

    if "event" not in df_original.columns:
        # aucune étiquette d'origine
        if "event" not in df_resampled.columns:
            df_resampled["event"] = np.nan
        if df_resampled["event"].dtype != "object":
            df_resampled["event"] = df_resampled["event"].astype("object")
        return df_resampled

    src = df_original[["timestamp", "event"]].dropna(subset=["timestamp"]).copy()
    src["timestamp"] = pd.to_datetime(src["timestamp"])  # sécurité

    # merge_asof nearest avec tolérance
    out = pd.merge_asof(
        df_resampled.sort_values("timestamp"),
        src.sort_values("timestamp").rename(columns={"event": "event_src"}),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=tol_ms),
    )
    if "event_src" in out.columns:
        out["event"] = out["event_src"].combine_first(out.get("event"))
        out = out.drop(columns=["event_src"])

    if "event" not in out.columns:
        out["event"] = np.nan
    if out["event"].dtype != "object":
        out["event"] = out["event"].astype("object")
    return out


def _enforce_immobility(df: pd.DataFrame, labels=("stop", "wait")) -> pd.DataFrame:
    """Impose l'immobilité stricte sur les segments labellisés.
    - speed/speed_kmh = 0
    - lat/lon constants durant le segment
    Ne modifie pas t/dt.
    """
    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df

    # Trouver la colonne d'événements
    ev_cols = [c for c in df.columns if c.lower() in ("event", "events", "tag", "tags", "state")]
    if not ev_cols:
        return df
    ev_col = ev_cols[0]

    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    spd = df["speed"].to_numpy() if "speed" in df.columns else None
    spd_kmh = df["speed_kmh"].to_numpy() if "speed_kmh" in df.columns else None

    lab = df[ev_col].astype(str).str.lower()

    def _segments(mask_series: pd.Series):
        idx = mask_series.to_numpy().nonzero()[0]
        if idx.size == 0:
            return []
        segs = []
        s = idx[0]; p = idx[0]
        for i in idx[1:]:
            if i == p + 1:
                p = i; continue
            segs.append((s, p)); s = i; p = i
        segs.append((s, p))
        return segs

    # Masque global stop|wait
    m_all = None
    for L in labels:
        cur = lab.str.contains(L, na=False)
        m_all = cur if m_all is None else (m_all | cur)

    if m_all is None:
        return df

    for a, b in _segments(m_all):
        rlat = float(np.median(lat[a:b+1]))
        rlon = float(np.median(lon[a:b+1]))
        lat[a:b+1] = rlat
        lon[a:b+1] = rlon
        if spd is not None:
            spd[a:b+1] = 0.0
        if spd_kmh is not None:
            spd_kmh[a:b+1] = 0.0

    df.loc[:, "lat"] = lat
    df.loc[:, "lon"] = lon
    if spd is not None:
        df.loc[:, "speed"] = spd
    if spd_kmh is not None:
        df.loc[:, "speed_kmh"] = spd_kmh

    for col in ("step_m", "distance_m", "delta_m"):
        if col in df.columns:
            df.loc[m_all, col] = 0.0
    return df


def _compute_dt_seconds(df: pd.DataFrame) -> np.ndarray:
    """Retourne dt (s) entre échantillons successifs, estimé depuis `timestamp` si dispo,
    sinon heuristique par fréquence ~10 Hz.
    """
    n = len(df)
    dt = np.full(n, np.nan, dtype=float)
    if n == 0:
        return dt
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]").to_numpy()
        # diff en secondes
        d = (t[1:] - t[:-1]).astype('timedelta64[ns]').astype(np.int64) / 1e9
        if d.size:
            dt[1:] = d
            # Remplir les NaN/0 par la médiane (sécurise les pas anormaux)
            med = np.nanmedian(d[d > 0]) if np.any(d > 0) else 0.1
            dt = np.where((dt <= 0) | ~np.isfinite(dt), med, dt)
        else:
            dt[:] = 0.1
    else:
        dt[:] = 0.1
    return dt


def _recompute_speed_from_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcule `speed`/`speed_kmh` à partir de lat/lon et dt (Haversine).
    Ne modifie pas lat/lon, s'applique après clamp.
    """
    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    n = len(df)
    step_m = np.zeros(n, dtype=float)
    if n > 1:
        # Haversine vectorisé simple
        R = 6371000.0
        lat1 = np.radians(lat[:-1])
        lat2 = np.radians(lat[1:])
        dlat = lat2 - lat1
        dlon = np.radians(lon[1:] - lon[:-1])
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        step_m[1:] = R * c
    dt = _compute_dt_seconds(df)
    with np.errstate(divide='ignore', invalid='ignore'):
        speed_mps = np.where(dt > 0, step_m / dt, 0.0)
    speed_kmh = speed_mps * 3.6
    # Écrire/mettre à jour les colonnes
    df.loc[:, "step_m"] = step_m
    df.loc[:, "speed"] = speed_mps
    df.loc[:, "speed_kmh"] = speed_kmh
    return df


def _cap_delta_speed(df: pd.DataFrame, dv_max_kmh_per_step: float = 5.0) -> pd.DataFrame:
    """Borne la variation de vitesse instantanée |Δv| par pas (par défaut 5 km/h à 10 Hz).
    S'applique sur `speed_kmh`; recalcule `speed` en m/s ensuite.
    """
    if df is None or df.empty or "speed_kmh" not in df.columns:
        return df
    v = df["speed_kmh"].to_numpy(dtype=float)
    if v.size <= 1:
        return df
    v_capped = v.copy()
    for i in range(1, v.size):
        dv = v_capped[i] - v_capped[i-1]
        if dv > dv_max_kmh_per_step:
            v_capped[i] = v_capped[i-1] + dv_max_kmh_per_step
        elif dv < -dv_max_kmh_per_step:
            v_capped[i] = v_capped[i-1] - dv_max_kmh_per_step
    # Écrire vitesse bornée
    df.loc[:, "speed_kmh"] = v_capped
    df.loc[:, "speed"] = v_capped / 3.6
    return df


# === Helper: _cap_absolute_speed ===
def _cap_absolute_speed(df: pd.DataFrame, vmax_kmh: float = 130.0) -> pd.DataFrame:
    """Borne la vitesse absolue en limitant le déplacement entre i-1 et i.

    Stratégie robuste (multi-passes) :
      - On effectue plusieurs balayages successifs (sweeps) de 1→N-1.
      - Si le segment (i-1→i) dépasse la borne, on rapproche i vers i-1 pour fixer
        step(i-1,i) = vmax * dt[i].
      - On contrôle aussitôt le segment suivant (i→i+1) et on rapproche i+1 si besoin.
      - On s'arrête quand plus aucune correction n'est nécessaire ou après max_sweeps.
    """
    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df
    n = len(df)
    if n <= 1:
        return df

    # Copies numpy
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()

    # dt réel par point (i s'applique au segment i-1→i)
    dt = _compute_dt_seconds(df)
    vmax_mps = float(vmax_kmh) / 3.6

    # Colonnes d'événements pour protéger les segments 'stop'/'wait'
    ev_cols = [c for c in df.columns if c.lower() in ("event", "events", "tag", "tags", "state")]
    protect = np.zeros(n, dtype=bool)
    if ev_cols:
        ev = df[ev_cols[0]].astype(str).str.lower()
        for L in ("stop", "wait"):
            protect |= ev.str.contains(L, na=False).to_numpy()

    # Haversine utilitaires
    def step_between(i0: int, i1: int) -> float:
        """Distance (m) entre points i0 et i1."""
        R = 6371000.0
        dlat = np.radians(lat[i1] - lat[i0])
        dlon = np.radians(lon[i1] - lon[i0])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat[i0]))*np.cos(np.radians(lat[i1]))*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def move_towards(i_from: int, i_to: int, target_step_m: float) -> None:
        """Rapproche le point i_from vers i_to pour obtenir exactement target_step_m."""
        if i_from == i_to:
            return
        dlat = lat[i_from] - lat[i_to]
        dlon = lon[i_from] - lon[i_to]
        if dlat == 0.0 and dlon == 0.0:
            return
        cur = step_between(i_to, i_from)
        if cur <= 1e-6:
            return
        s = float(target_step_m) / float(cur)
        s = np.clip(s, 0.0, 1.0)  # on rapproche uniquement
        lat[i_from] = lat[i_to] + s * dlat
        lon[i_from] = lon[i_to] + s * dlon

    # Balayages successifs jusqu'à stabilisation
    max_sweeps = 8
    eps_step = 1e-3  # m — garde-fou si dt≈0
    for _ in range(max_sweeps):
        changed = False
        # i va de 1..n-1 pour corriger le segment (i-1→i)
        for i in range(1, n):
            if protect[i] or protect[i-1]:
                continue
            vmax_step = vmax_mps * max(dt[i], 1e-3)
            vmax_step = max(vmax_step, eps_step)
            d = step_between(i-1, i)
            if d > vmax_step + 1e-6:
                # corriger le point i vers i-1
                move_towards(i, i-1, vmax_step)
                changed = True

                # contrôler le segment suivant (i→i+1) aussitôt
                if i + 1 < n and not protect[i+1]:
                    vmax_next = vmax_mps * max(dt[i+1], 1e-3)
                    vmax_next = max(vmax_next, eps_step)
                    d_next = step_between(i, i+1)
                    if d_next > vmax_next + 1e-6:
                        move_towards(i+1, i, vmax_next)
        if not changed:
            break

    # Écrire lat/lon et recalculer cinématique à partir du résultat final
    df.loc[:, "lat"] = lat
    df.loc[:, "lon"] = lon
    df = _recompute_speed_from_gps(df)
    return df


@deprecated
def spatial_reprojection(df: pd.DataFrame, speed_target, dt: float = 0.1) -> pd.DataFrame:
    """(Deprecated) Reconstruct a regularly sampled frame while preserving GPS columns.

    - Keeps `lat`/`lon` as-is (no spatial interpolation).
    - Generates a regular timebase inferred from ``dt``.
    - Interpolates numeric columns along cumulative distance to avoid temporal gaps.
    - Leaves non-numeric columns untouched, except `timestamp` which is rebuilt.

    Notes
    -----
    The parameter ``speed_target`` is kept for backward API compatibility but is
    not used by this deprecated path.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame with at least ``timestamp``, ``lat``, ``lon`` and numeric signals.
    speed_target : Any
        Unused; kept for compatibility.
    dt : float, optional
        Target time step in seconds (default: 0.1 → 10 Hz).

    Returns
    -------
    pd.DataFrame
        New frame of the same length as input, with rebuilt timestamps and
        numeric columns re-interpolated.
    """
    df = df.reset_index(drop=True).copy()

    # cumulative distance (used as monotonic abscissa for interpolation)
    cumdist = compute_cumulative_distance(df)

    # Ensure monotonic increasing abscissa for interp1d
    if not np.all(np.diff(cumdist) >= 0):
        sort_idx = np.argsort(cumdist)
        cumdist_sorted = cumdist[sort_idx]
        df_sorted = df.iloc[sort_idx].reset_index(drop=True)
        work_df = df_sorted
        xref = cumdist_sorted
        logger.debug("cumdist not monotonic; sorted before interpolation")
    else:
        work_df = df
        xref = cumdist

    # Build output frame
    new_df = pd.DataFrame()
    for col in work_df.columns:
        if col in ("lat", "lon"):
            # Preserve raw GPS coordinates (no reinterpolation)
            new_df[col] = work_df[col].to_numpy(copy=True)
            continue
        if col == "timestamp":
            # Regular timestamps from start/end at the requested frequency
            start = pd.to_datetime(work_df["timestamp"].iloc[0])
            end = pd.to_datetime(work_df["timestamp"].iloc[-1])
            freq_hz = int(round(1.0 / dt))
            # use ms granularity to avoid rounding warnings
            new_timestamps = pd.date_range(start=start, end=end, freq=f"{int(1000/freq_hz)}ms")
            # keep same length as input to avoid shape drift
            new_df[col] = new_timestamps[: len(work_df)]
            continue
        if pd.api.types.is_numeric_dtype(work_df[col]):
            y = work_df[col].to_numpy()
            # Defensive: if xref has duplicates (zero-length segments), use kind='previous' fallback
            if np.any(np.diff(xref) == 0):
                # simple forward fill over duplicates context
                new_df[col] = pd.Series(y).ffill().bfill().to_numpy()
            else:
                f_col = interp1d(xref, y, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
                new_df[col] = f_col(xref)
        else:
            new_df[col] = work_df[col]

    # Clamp anti-sauts GPS puis immobilité stricte
    new_df = _clamp_gps_jumps(new_df, max_step_m=3.0, max_iters=20)
    new_df = _enforce_immobility(new_df, labels=("stop", "wait"))

    # Recalculer la vitesse depuis GPS puis borner Δv
    new_df = _recompute_speed_from_gps(new_df)
    new_df = _cap_delta_speed(new_df, dv_max_kmh_per_step=5.0)

    # Appliquer la borne ABSOLUE de vitesse en DERNIER (garantie anti > vmax)
    new_df = _cap_absolute_speed(new_df, vmax_kmh=130.0)
    # Réimposer l'immobilité stricte si des points ont été légèrement déplacés
    new_df = _enforce_immobility(new_df, labels=("stop", "wait"))

    # Recalcule les colonnes de distance pour cohérence
    new_df = _recompute_distance_columns(new_df)

    try:
        from simulator.events.tracker import EventCounter

        tracker = EventCounter()
        tracker.count_from_dataframe(new_df)
        tracker.show("Après reprojection (spatial_reprojection)")
    except Exception as e:
        logger.debug("EventCounter unavailable in spatial_reprojection: %s", e)

    return new_df


@deprecated
def resample_time(df: pd.DataFrame, freq_hz: int = 10) -> pd.DataFrame:
    """(Deprecated) Resample the frame on a regular time grid (default 10 Hz).

    - Ensures strictly increasing timestamps.
    - Interpolates numeric columns linearly (with ffill/bfill guards).
    - Forward-fills/bfills non-numeric columns (y compris `event`).
    - Reapplies original `event` labels *only at exact timestamps* to avoid label propagation.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame containing a ``timestamp`` column.
    freq_hz : int, optional
        Target frequency in Hertz (default 10).

    Returns
    -------
    pd.DataFrame
        Resampled frame.
    """
    df = df.copy()

    # Normalize and sort time
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # idempotent
    df = df.sort_values("timestamp")

    # Drop duplicated timestamps to avoid reindex errors
    if df["timestamp"].duplicated().any():
        df = df.loc[~df["timestamp"].duplicated(keep="first")]
        logger.warning("Doublons timestamp détectés et supprimés avant rééchantillonnage.")

    # Preserve original event labels at their exact timestamps (no propagation)
    original_event = None
    if "event" in df.columns:
        tmp = df[["timestamp", "event"]].copy()
        # Keep only non-null labels (copy to avoid SettingWithCopyWarning later)
        original_event = tmp.dropna(subset=["event"]).copy()  # could be empty

    # Build new regular time index
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    new_timestamps = pd.date_range(start=start, end=end, freq=f"{int(1000 / freq_hz)}ms")

    # Interpolate numeric columns
    df = df.set_index("timestamp").reindex(new_timestamps)
    df.index.name = "timestamp"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").ffill().bfill()

    # Forward/backward fill non-numeric columns SAUF 'event' (géré par nearest+tolérance)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if "event" in non_numeric_cols:
        non_numeric_cols.remove("event")
    if non_numeric_cols:
        df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill()

    # Reset index back to column
    df = df.reset_index()

    # Assigner 'event' par nearest avec tolérance courte (par défaut 500 ms)
    if original_event is not None and not original_event.empty:
        df = _assign_event_nearest_with_tolerance(df_resampled=df, df_original=original_event, tol_ms=500)
    else:
        if "event" not in df.columns:
            df["event"] = np.nan
        if df["event"].dtype != "object":
            df["event"] = df["event"].astype("object")

    # Clamp anti-sauts GPS puis immobilité stricte
    df = _clamp_gps_jumps(df, max_step_m=3.0, max_iters=20)
    df = _enforce_immobility(df, labels=("stop", "wait"))

    # Recalculer la vitesse depuis GPS puis borner Δv
    df = _recompute_speed_from_gps(df)
    df = _cap_delta_speed(df, dv_max_kmh_per_step=5.0)

    # Appliquer la borne ABSOLUE de vitesse en DERNIER (garantie anti > vmax)
    df = _cap_absolute_speed(df, vmax_kmh=130.0)
    # Réimposer l'immobilité stricte si des points ont été légèrement déplacés
    df = _enforce_immobility(df, labels=("stop", "wait"))

    # Recalcule les colonnes de distance pour cohérence
    df = _recompute_distance_columns(df)

    try:
        from simulator.events.tracker import EventCounter

        tracker = EventCounter()
        tracker.count_from_dataframe(df)
        tracker.show("Après reprojection (resample_time)")
    except Exception as e:
        logger.debug("EventCounter unavailable in resample_time: %s", e)

    return df


# === Composition sans reprojection (sans toucher aux timestamps, sans rééchantillonnage) ===
def compose_without_reprojection(df: pd.DataFrame, vmax_kmh: float = 130.0, dv_max_kmh_per_step: float = 5.0) -> pd.DataFrame:
    """
    Composition "sans reprojection" :
      - ne touche pas aux timestamps
      - ne rééchantillonne pas
      - impose l'immobilité stricte sur stop/wait
      - clamp anti-sauts GPS très léger (pour supprimer des outliers extrêmes sans lisser la trajectoire)
      - recalcule la vitesse depuis GPS
      - borne Δv par pas (doucement)
      - borne absolue de vitesse
      - réimpose l'immobilité sur stop/wait (au cas où)
      - recalcule step/distance

    Retourne un DataFrame de même longueur que l'entrée.
    """
    if df is None or df.empty:
        return df

    out = df.reset_index(drop=True).copy()

    # Légère suppression d'outliers GPS (seuil bas pour ne pas déformer la route)
    out = _clamp_gps_jumps(out, max_step_m=3.0, max_iters=10)

    # Immobilité stricte sur les segments annotés
    out = _enforce_immobility(out, labels=("stop", "wait"))

    # Cinématique depuis GPS
    out = _recompute_speed_from_gps(out)

    # Lissage des variations de vitesse par pas
    out = _cap_delta_speed(out, dv_max_kmh_per_step=dv_max_kmh_per_step)

    # Borne absolue — ne déplace pas les points stop/wait
    out = _cap_absolute_speed(out, vmax_kmh=vmax_kmh)

    # Réimposer immobilité (au cas où des micro-ajustements auraient bougé des points)
    out = _enforce_immobility(out, labels=("stop", "wait"))

    # Colonnes distance
    out = _recompute_distance_columns(out)

    try:
        from simulator.events.tracker import EventCounter
        tracker = EventCounter()
        tracker.count_from_dataframe(out)
        tracker.show("Final après composition sans reprojection")
    except Exception as e:
        logger.debug("EventCounter unavailable in compose_without_reprojection: %s", e)

    return out
