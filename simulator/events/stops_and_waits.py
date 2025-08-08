"""Stop/Wait long-duration events generator.

This module handles:
- Injection of engine-off stops (`stop`) and temporary halts (`wait`),
- Temporal expansion of these events to target durations,
- Application of realistic decel/accel inertial profiles around stops/waits,
- Optional inertial noise during `wait` phases.

Configuration (durations, spacing, counts, noise, etc.) is read from `config/events.yaml`.

Public API:
- `generate_stops(df)`
- `generate_waits(df)`
- `expand_stop_and_wait(df)`
- `apply_stop_or_wait_profile(df)`
- `apply_inertial_noise_on_wait(df)`
- `inject_stops_and_waits(df)` (deprecated pipeline)
"""

import numpy as np
import pandas as pd
import logging
from simulator.events.utils import ensure_event_column_object
from core.decorators import deprecated

from simulator.events.config import CONFIG

from typing import Optional, Tuple

def _ensure_event_object(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the `event` column exists and is dtype=object.
    Uses `ensure_event_column_object` and enforces object dtype to avoid FutureWarnings.
    """
    df = ensure_event_column_object(df)
    if df["event"].dtype != "object":
        df["event"] = df["event"].astype("object")
    return df

logger = logging.getLogger(__name__)


HZ = 10


def _pick_start_index(
    occupancy: np.ndarray,
    duration_pts: int,
    spacing_pts: int,
    rng: np.random.Generator,
) -> Optional[int]:
    """Pick a random start index such that the window [start-spacing_pts, start+duration_pts+spacing_pts)
    has no occupied points. Returns None if no valid slot is found after reasonable attempts.
    """
    n = occupancy.shape[0]
    lo = spacing_pts
    hi = n - duration_pts - spacing_pts
    if hi <= lo:
        return None

    # Build a boolean mask of valid starts using convolution-like prefix sums for speed
    # A start `i` is valid if the slice occupancy[i-spacing_pts : i+duration_pts+spacing_pts] has no True
    # To avoid O(n*duration) checks, precompute a sliding window sum over occupancy.
    occ_int = occupancy.astype(np.int32)
    window = duration_pts + 2 * spacing_pts
    # cumulative sum trick
    csum = np.concatenate(([0], np.cumsum(occ_int)))

    valid = np.zeros(n, dtype=bool)
    # We only evaluate for i in [lo, hi]
    idxs = np.arange(lo, hi)
    # for each i, sum in [i-spacing_pts, i+duration_pts+spacing_pts)
    left = idxs - spacing_pts
    right = idxs + duration_pts + spacing_pts
    segment_sum = csum[right] - csum[left]
    valid[idxs] = segment_sum == 0

    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return None
    return int(rng.choice(candidates))



def _reserve(occupancy: np.ndarray, start: int, duration_pts: int, spacing_pts: int) -> None:
    """Mark the region [start-spacing_pts, start+duration_pts+spacing_pts) as occupied in-place."""
    n = occupancy.shape[0]
    a = max(0, start - spacing_pts)
    b = min(n, start + duration_pts + spacing_pts)
    occupancy[a:b] = True


# --- New helpers for dilation and config-driven merge gap ---
import numpy as np

def _dilate_occupancy(occ: np.ndarray, spacing_pts: int) -> np.ndarray:
    """Return a boolean occupancy dilated by `spacing_pts` on each side (1D morphological dilation).
    Uses a convolution with a ones kernel for efficiency.
    """
    if spacing_pts <= 0 or occ.size == 0:
        return occ.copy()
    occ_int = occ.astype(np.int32)
    kernel = np.ones(2 * spacing_pts + 1, dtype=np.int32)
    # 'same' keeps length; any positive value means there was an occupied point within the window
    dil = np.convolve(occ_int, kernel, mode="same") > 0
    return dil


def _get_merge_gap_pts_from_config(default: int = 1) -> int:
    """Read merge gap from CONFIG for stop/wait; falls back to `default` if missing."""
    try:
        stop_gap = int(CONFIG.get("stop", {}).get("merge_gap_pts", default))
    except Exception:
        stop_gap = default
    try:
        wait_gap = int(CONFIG.get("wait", {}).get("merge_gap_pts", default))
    except Exception:
        wait_gap = default
    return max(default, stop_gap, wait_gap)

def _block_starts(df: pd.DataFrame, event_name: str) -> pd.Index:
    """Retourne les index de début de blocs contigus pour un évènement donné.
    On considère qu'un index est un début de bloc si `event == event_name` et que
    l'index précédent n'est pas ce même évènement.
    """
    event_col = df['event']
    mask = event_col == event_name
    # début de bloc = True quand mask est True et mask.shift(1) est False/NaN
    starts = mask & (~mask.shift(1, fill_value=False))
    return df.index[starts]

def _generate_long_event(
    df: pd.DataFrame,
    label: str,
    cfg: dict,
    default_min_spacing_s: float,
    hz_fallback: int = HZ,
) -> pd.DataFrame:
    """Generic long-event injector used by `generate_stops` and `generate_waits`.

    It reserves non-overlapping windows using an occupancy array and writes a minimal
    plateau (speed/acc=0) for the configured minimal duration; full expansion is
    performed later by `expand_stop_and_wait`.
    """
    max_events = int(cfg.get("max_events", 0))
    min_duration_s = float(cfg.get("min_duration_s", 0))
    hz = int(cfg.get("hz", hz_fallback))
    max_attempts = int(cfg.get("max_attempts_per_event", 10))
    min_spacing_s = float(cfg.get("min_spacing_s", default_min_spacing_s))

    # Ensure event column exists and is dtype=object to avoid FutureWarning
    try:
        from simulator.events.utils import ensure_event_column_object
        df = ensure_event_column_object(df)
    except Exception:
        # Fallback: create column if missing
        if "event" not in df.columns:
            df["event"] = np.nan
    if df["event"].dtype != "object":
        df["event"] = df["event"].astype("object")

    duration_pts = max(1, int(min_duration_s * hz))
    spacing_pts = max(0, int(min_spacing_s * hz))

    # Start from current labels and dilate by spacing so we respect min spacing vs pre-existing events
    occupancy_raw = df["event"].notna().to_numpy(copy=True)
    occupancy = _dilate_occupancy(occupancy_raw, spacing_pts)
    seed = cfg.get("seed", None)
    rng = np.random.default_rng(int(seed)) if isinstance(seed, (int, np.integer)) and seed != 0 else np.random.default_rng(None)

    count = 0
    attempts = 0

    while count < max_events and attempts < max_events * max_attempts:
        attempts += 1
        start = _pick_start_index(occupancy, duration_pts, spacing_pts, rng)
        if start is None:
            break

        # Reserve region *before* mutating DataFrame
        _reserve(occupancy, start, duration_pts, spacing_pts)

        end = start + duration_pts
        df.loc[start:end - 1, ["speed", "acc_x", "acc_y", "acc_z"]] = 0
        df.loc[start:end - 1, "event"] = label
        logger.debug("[%s] Bloc injecté de l'index %d à %d (durée min %.1fs, spacing %.1fs)", label.upper(), start, end - 1, min_duration_s, min_spacing_s)
        count += 1

    if count == 0:
        logger.warning("[%s] Aucun événement injecté (contraintes trop fortes ?)", label.upper())
    else:
        logger.info("[%s] %d bloc(s) injecté(s) (min_duration=%.1fs, spacing=%.1fs)", label.upper(), count, min_duration_s, min_spacing_s)

    return df

def generate_stops(df) -> pd.DataFrame:
    cfg = CONFIG["stop"]
    return _generate_long_event(df, label="stop", cfg=cfg, default_min_spacing_s=60.0)

@deprecated
def inject_stops_and_waits(df, max_events_per_type=5, hz=10, min_stop_duration=120, min_wait_duration=30):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    logger.info("[INJECTION] Génération initiale des stops...")
    df = generate_stops(df, max_events=max_events_per_type, min_duration=min_stop_duration)

    logger.info("[INJECTION] Génération initiale des waits...")
    df = generate_waits(df, max_events=max_events_per_type, min_duration=min_wait_duration)

    logger.info("[INJECTION] Expansion des stops et waits à leur durée réelle avec profils inertiels...")
    df = expand_stop_and_wait(df, hz=hz)

    logger.info("[INJECTION] Application des profils inertiels autour des stops/waits...")
    df = apply_stop_or_wait_profile(df, hz=hz)

    logger.info("[INJECTION] Application du bruit inertiel spécifique sur les waits...")
    df = apply_inertial_noise_on_wait(df, hz=hz)

    return df


def generate_waits(df) -> pd.DataFrame:
    cfg = CONFIG["wait"]
    return _generate_long_event(df, label="wait", cfg=cfg, default_min_spacing_s=20.0)


def apply_stop_or_wait_profile(df, v_target_kmh=40, decel_amplitude=-3.0, accel_amplitude=3.0, hz=10) -> pd.DataFrame:
    df = _ensure_event_object(df)
    # Ne traiter chaque bloc stop/wait qu'une seule fois pour éviter les doublons
    stop_starts = _block_starts(df, 'stop')
    wait_starts = _block_starts(df, 'wait')
    block_starts = list(stop_starts) + list(wait_starts)

    if not block_starts:
        logger.info("[INFO] Aucun événement stop ou wait détecté, aucun profil inertiel appliqué.")
        return df

    for idx_stop in sorted(block_starts):
        decel_duration_s = 3
        decel_window = int(decel_duration_s * hz)
        decel_start = max(0, idx_stop - decel_window)

        v0 = df.at[decel_start, 'speed'] / 3.6
        for i in range(decel_window):
            idx = decel_start + i
            if idx >= idx_stop:
                break
            v = max(v0 + decel_amplitude * (i / hz), 0)
            df.at[idx, 'speed'] = v * 3.6
            df.at[idx, 'acc_x'] = decel_amplitude

        df.at[idx_stop, 'speed'] = 0
        df.at[idx_stop, 'acc_x'] = 0

        accel_duration_s = 3
        accel_window = int(accel_duration_s * hz)
        accel_start = idx_stop + 1
        v0 = 0

        for i in range(accel_window):
            idx = accel_start + i
            if idx >= len(df):
                break
            v = v0 + accel_amplitude * (i / hz)
            if v * 3.6 >= v_target_kmh:
                break
            df.at[idx, 'speed'] = v * 3.6
            df.at[idx, 'acc_x'] = accel_amplitude

    return df


def apply_inertial_noise_on_wait(df, hz=10, noise_std=0.05, verbose=False) -> pd.DataFrame:
    mask_wait = df['event'] == 'wait'
    n_points = mask_wait.sum()

    if n_points == 0:
        if verbose:
            logger.info("[INFO] Aucun événement 'wait' trouvé pour bruit inertiel.")
        return df

    if verbose:
        logger.info(f"[INFO] Application du bruit inertiel sur {n_points} points 'wait'...")

    df.loc[mask_wait, 'acc_x'] += np.random.normal(0, noise_std, n_points)
    df.loc[mask_wait, 'acc_y'] += np.random.normal(0, noise_std, n_points)
    df.loc[mask_wait, 'acc_z'] += np.random.normal(0, noise_std, n_points)

    return df


def _merge_contiguous_blocks(df: pd.DataFrame, label: str, max_gap_pts: int = 1) -> pd.DataFrame:
    """Merge contiguous blocks (or blocks separated by <= max_gap_pts empty points) for a given label.
    Keeps `event` dtype as object and does not alter other labels.
    """
    df = df.copy()
    df = _ensure_event_object(df)
    if df["event"].dtype != "object":
        df["event"] = df["event"].astype("object")

    is_label = (df["event"] == label).to_numpy()
    n = len(df)
    i = 0
    while i < n:
        if not is_label[i]:
            i += 1
            continue
        # start of a block
        start = i
        j = i + 1
        # advance while within label (allow small gaps to be merged)
        while j < n:
            if is_label[j]:
                j += 1
                continue
            # if small gap, look ahead to see if label resumes within max_gap_pts
            gap_end = min(n, j + max_gap_pts)
            resumed = False
            k = j
            while k < gap_end:
                if is_label[k]:
                    # fill gap as label
                    df.loc[j:k-1, "event"] = label
                    is_label[j:k] = True
                    j = k
                    resumed = True
                    break
                k += 1
            if not resumed:
                break
        end = j  # [start, end)
        # consolidate: already labeled as `label`
        i = j
    return df


def merge_contiguous_stop_wait(df: pd.DataFrame, max_gap_pts: Optional[int] = None) -> pd.DataFrame:
    """Merge contiguous/near-contiguous segments for 'stop' and 'wait'.
    If `max_gap_pts` is None, reads the value from CONFIG (stop/wait.merge_gap_pts).
    """
    if max_gap_pts is None:
        max_gap_pts = _get_merge_gap_pts_from_config(default=1)
    df = _merge_contiguous_blocks(df, "stop", max_gap_pts=max_gap_pts)
    df = _merge_contiguous_blocks(df, "wait", max_gap_pts=max_gap_pts)
    return df


def expand_stop_and_wait(df, hz=10, stop_duration_s=120, wait_duration_s=30, verbose=False) -> pd.DataFrame:
    df = df.copy()
    # Ensure event column is dtype=object before any assignment
    df = _ensure_event_object(df)

    occupied = df["event"].notna().to_numpy(copy=True)
    stop_duration_pts = int(stop_duration_s * hz)
    wait_duration_pts = int(wait_duration_s * hz)

    total_stops = df['event'].value_counts().get('stop', 0)
    total_waits = df['event'].value_counts().get('wait', 0)

    if verbose:
        logger.info(f"[INFO] Expansion des {total_stops} stops et {total_waits} waits.")

    indices_to_expand_stop = []
    indices_to_expand_wait = []

    # Débuts de blocs uniquement
    stop_starts = _block_starts(df, 'stop')
    wait_starts = _block_starts(df, 'wait')

    for idx in stop_starts:
        end = min(idx + stop_duration_pts, len(df))
        # only extend into free cells or cells already labelled as 'stop'
        slice_idx = np.arange(idx, end)
        keep = ~occupied[slice_idx] | (df.loc[slice_idx, "event"].values == "stop")
        indices_to_expand_stop.extend(slice_idx[keep].tolist())

    for idx in wait_starts:
        end = min(idx + wait_duration_pts, len(df))
        slice_idx = np.arange(idx, end)
        keep = ~occupied[slice_idx] | (df.loc[slice_idx, "event"].values == "wait")
        indices_to_expand_wait.extend(slice_idx[keep].tolist())

    if indices_to_expand_stop:
        df.loc[indices_to_expand_stop, 'speed'] = 0
        df.loc[indices_to_expand_stop, 'acc_x'] = 0
        df.loc[indices_to_expand_stop, 'acc_y'] = 0
        df.loc[indices_to_expand_stop, 'acc_z'] = 0
        df.loc[indices_to_expand_stop, 'event'] = 'stop'

    if indices_to_expand_wait:
        df.loc[indices_to_expand_wait, 'speed'] = 0
        df.loc[indices_to_expand_wait, 'acc_x'] = 0
        df.loc[indices_to_expand_wait, 'acc_y'] = 0
        df.loc[indices_to_expand_wait, 'acc_z'] = 0
        df.loc[indices_to_expand_wait, 'event'] = 'wait'

    if verbose:
        logger.info(f"[INFO] Expansion complète terminée : {len(indices_to_expand_stop)} points stop, {len(indices_to_expand_wait)} points wait.")

    # Merge fragmented blocks to avoid artificial multiple short segments
    df = merge_contiguous_stop_wait(df, max_gap_pts=None)
    # Keep `event` as plain Python objects
    df["event"] = df["event"].astype("object")

    return df
