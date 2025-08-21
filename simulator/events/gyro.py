# simulator/events/gyro.py
import numpy as np
import pandas as pd

__all__ = [
    "simulate_gyroscope_from_heading",
    "inject_gyroscope_from_events",
    "generate_gyroscope_signals",
]

# --- Helpers ---
def _as_radians(heading_series: pd.Series) -> np.ndarray:
    """Return heading as radians (auto-detects degrees vs radians)."""
    vals = pd.to_numeric(heading_series, errors="coerce").fillna(0).to_numpy()
    # Heuristic: if the magnitude is larger than ~2π, assume degrees
    if np.nanmax(np.abs(vals)) > 2 * np.pi + 1e-6:
        return np.radians(vals)
    return vals

# --- Back-compat shim 1 ---
def simulate_gyroscope_from_heading(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Compute gyro_z (rad/s) from heading. Accepts heading in degrees or radians.

    This is a back-compat shim kept for older imports from pipeline.
    """
    if "heading" not in df.columns:
        raise ValueError("'heading' column is required to compute gyro_z from heading.")
    df = df.copy()
    heading_rad = _as_radians(df["heading"])  # radians

    # Unwrapped gradient to avoid ±π discontinuities
    d_heading = np.unwrap(np.gradient(heading_rad))
    df["gyro_z"] = d_heading * float(hz)  # rad/s
    return df

# --- Back-compat shim 2 ---
def inject_gyroscope_from_events(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Add gyro_x/gyro_y base noise and apply event-driven signatures.

    This is a back-compat shim kept for older imports from pipeline.
    """
    df = df.copy()

    # Ensure columns exist
    if "gyro_x" not in df.columns:
        np.random.seed(42)
        df["gyro_x"] = np.random.normal(0.01, 0.02, size=len(df))
    if "gyro_y" not in df.columns:
        np.random.seed(42)
        df["gyro_y"] = np.random.normal(0.01, 0.02, size=len(df))
    if "gyro_z" not in df.columns:
        df["gyro_z"] = 0.0
    if "event" not in df.columns:
        df["event"] = pd.Series([np.nan] * len(df), index=df.index)

    n = len(df)
    for i, row in df.iterrows():
        evt = row.get("event", None)
        if pd.isna(evt):
            continue
        window = 5  # ~0.5 s at 10 Hz
        i0 = max(0, i - window)
        i1 = min(n, i + window + 1)
        n_pts = i1 - i0
        if n_pts <= 0:
            continue

        if evt == "dos_dane":
            df.loc[i0:i1 - 1, "gyro_x"] += np.sin(np.linspace(0, np.pi, n_pts)) * 2.0
        elif evt == "freinage":
            df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(0.5, -0.5, n_pts)
        elif evt == "acceleration":
            df.loc[i0:i1 - 1, "gyro_x"] += np.linspace(-0.5, 0.5, n_pts)
        elif evt == "trottoir":
            df.loc[i0:i1 - 1, "gyro_y"] += np.sin(np.linspace(0, np.pi, n_pts)) * 2.0
        elif evt == "nid_de_poule":
            df.loc[i0:i1 - 1, "gyro_z"] += np.random.normal(0, 3.0, n_pts)

    df[["gyro_x", "gyro_y", "gyro_z"]] = df[["gyro_x", "gyro_y", "gyro_z"]].round(4)
    return df

# --- Unified modern API ---
def generate_gyroscope_signals(df: pd.DataFrame, hz: int = 10) -> pd.DataFrame:
    """
    Modern one-shot entrypoint: compute gyro_z from heading, then apply event signatures
    and base noise for gyro_x/gyro_y. Returns a new DataFrame with gyro_* columns.
    """
    df = simulate_gyroscope_from_heading(df, hz=hz)
    df = inject_gyroscope_from_events(df, hz=hz)
    return df