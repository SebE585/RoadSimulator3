import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from core.kinematics import compute_cumulative_distance
from core.decorators import deprecated

logger = logging.getLogger(__name__)


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
    - Forward-fills/bfills non-numeric columns **except** `event`.
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
        # Keep only non-null labels
        original_event = tmp.dropna(subset=["event"])  # could be empty

    # Build new regular time index
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    new_timestamps = pd.date_range(start=start, end=end, freq=f"{int(1000 / freq_hz)}ms")

    # Interpolate numeric columns
    df = df.set_index("timestamp").reindex(new_timestamps)
    df.index.name = "timestamp"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").ffill().bfill()

    # Forward/backward fill non-numeric columns except 'event'
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    non_event_cols = [c for c in non_numeric_cols if c != "event"]
    if non_event_cols:
        df[non_event_cols] = df[non_event_cols].ffill().bfill()

    # Reset index back to column
    df = df.reset_index()

    # Ensure 'event' column is object-typed before any assignment to avoid FutureWarning
    if "event" in df.columns and df["event"].dtype != "object":
        df["event"] = df["event"].astype("object")

    # Re-apply original event labels **only at exact timestamps** via merge
    if original_event is not None and not original_event.empty:
        # Ensure dtype compatibility for merge
        original_event["timestamp"] = pd.to_datetime(original_event["timestamp"])
        df = df.merge(
            original_event.drop_duplicates(subset=["timestamp"], keep="last"),
            on="timestamp",
            how="left",
            suffixes=("", "__orig"),
        )
        # Prefer the original label where available
        if "event__orig" in df.columns:
            df["event"] = df["event__orig"].combine_first(df["event"]).astype("object")
            df = df.drop(columns=["event__orig"])
    else:
        # If there was no 'event' initially, ensure column exists as object and NaN values
        if "event" not in df.columns:
            df["event"] = np.nan
        if df["event"].dtype != "object":
            df["event"] = df["event"].astype("object")

    try:
        from simulator.events.tracker import EventCounter

        tracker = EventCounter()
        tracker.count_from_dataframe(df)
        tracker.show("Après reprojection (resample_time)")
    except Exception as e:
        logger.debug("EventCounter unavailable in resample_time: %s", e)

    return df
