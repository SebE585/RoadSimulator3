# simulator/pipeline/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from core import reprojection, validation
import core.kinematics as kinematics
from core.osmnx.client import enrich_road_type_stream
from core.utils import ensure_strictly_increasing_timestamps

# vitesse plateaux (si dispo)
try:
    from core.kinematics_speed import simulate_variable_speed
except Exception:
    simulate_variable_speed = None  # type: ignore

# events
from simulator.pipeline_utils import inject_all_events, complete_trajectory_fields
from simulator.events.tracker import EventCounter
from simulator.events.utils import clean_invalid_events
from simulator.events.gyro import generate_gyroscope_signals
try:
    from simulator.events.stop_wait import apply_stop_wait_at_positions
except Exception:
    apply_stop_wait_at_positions = None  # type: ignore

try:
    from simulator.events.initial_final import inject_initial_acceleration, inject_final_deceleration
except Exception:
    inject_initial_acceleration = inject_final_deceleration = None  # type: ignore

try:
    from simulator.events.stops_and_waits import merge_contiguous_stop_wait
except Exception:
    merge_contiguous_stop_wait = None  # type: ignore

try:
    from simulator.events.noise import inject_inertial_noise
except Exception:
    inject_inertial_noise = None  # type: ignore


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [pipeline_utils] %(message)s')


@dataclass
class PipelineOptions:
    inject_punctual_events: bool = True
    enforce_final_stop_sec: float = 5.0
    hz: int = 10


# ---------- helpers stop/wait ----------
def _count_stop_starts(df: pd.DataFrame) -> int:
    if "event" not in df.columns:
        return 0
    ev = df["event"].astype("object")
    is_stop = ev.eq("stop")
    starts = is_stop & (~is_stop.shift(1, fill_value=False))
    return int(starts.sum())


def _limit_and_space_stops(df: pd.DataFrame, max_events: int, min_sep_pts: int) -> pd.DataFrame:
    if "event" not in df.columns or df.empty:
        return df
    ev = df["event"].astype("object")
    is_stop = ev.eq("stop")
    starts = is_stop & (~is_stop.shift(1, fill_value=False))
    start_idxs = df.index[starts]
    if start_idxs.empty:
        return df

    selected = []
    last = -10**9
    for idx in start_idxs:
        if idx - last >= min_sep_pts:
            selected.append(idx)
            last = idx
            if len(selected) >= max_events:
                break

    keep = pd.Series(False, index=df.index)
    for s in selected:
        j = s
        while j < len(df) and df.at[j, "event"] == "stop":
            keep.iloc[j] = True
            j += 1

    to_clear = df.index[(ev.eq("stop")) & (~keep)]
    if len(to_clear):
        df.loc[to_clear, "event"] = np.nan
    return df


def _enforce_min_stop_spacing(df: pd.DataFrame, min_spacing_s: float, hz: int = 10) -> pd.DataFrame:
    if "event" not in df.columns or df.empty:
        return df
    min_gap = int(min_spacing_s * hz)
    ev = df["event"].astype("object")
    is_stop = ev.eq("stop")
    starts = is_stop & (~is_stop.shift(1, fill_value=False))
    starts_idx = df.index[starts].to_list()

    last_kept = None
    to_clear = []
    for s in starts_idx:
        if last_kept is None or (s - last_kept) >= min_gap:
            last_kept = s
        else:
            j = s
            while j < len(df) and df.at[j, "event"] == "stop":
                to_clear.append(j)
                j += 1
    if to_clear:
        df.loc[to_clear, "event"] = np.nan
    return df


# ---------- helpers vitesse ----------
def _clamp_speed_changes(df: pd.DataFrame, hz: int = 10, a_max_mps2: float = 2.0) -> pd.DataFrame:
    if "speed" not in df.columns or df.empty:
        return df
    dv_max = (a_max_mps2 * 3.6) / max(hz, 1)  # km/h per tick
    v = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).to_numpy(copy=True)
    for i in range(1, len(v)):
        dv = v[i] - v[i - 1]
        if dv > dv_max:
            v[i] = v[i - 1] + dv_max
        elif dv < -dv_max:
            v[i] = v[i - 1] - dv_max
    df["speed"] = v
    return df


class SimulationPipeline:
    def __init__(self, config: Dict, options: Optional[PipelineOptions] = None) -> None:
        self.cfg = config or {}
        self.opt = options or PipelineOptions()

    # ---- misc helpers ----
    @staticmethod
    def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns and not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df

    @staticmethod
    def _ensure_event_column(df: pd.DataFrame) -> pd.DataFrame:
        if "event" not in df.columns:
            df["event"] = pd.Series(index=df.index, dtype="object")
        else:
            df["event"] = df["event"].astype("object")
        return df

    @staticmethod
    def _log_events(df: pd.DataFrame, title: str) -> None:
        c = EventCounter()
        c.count_from_dataframe(df)
        c.show(label=title)

    @staticmethod
    def _call_with_supported_kwargs(func, **kwargs):
        if func is None:
            return None, {}
        import inspect
        params = set(inspect.signature(func).parameters.keys())
        return func, {k: v for k, v in kwargs.items() if k in params}

    def _apply_stop_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        policy = self.cfg.get("policy", {})
        keep_only = bool(policy.get("keep_only_scenario_stops", True))
        preserve_after = bool(policy.get("preserve_stops_after_reprojection", False))
        sw_cfg = self.cfg.get("events", {}).get("stop_wait", {})
        max_events = int(sw_cfg.get("max_events", 6))
        min_sep_pts = int(sw_cfg.get("min_separation_pts", 200))
        min_gap_s = float(sw_cfg.get("min_spacing_s", 60.0))

        # merge petits trous
        if merge_contiguous_stop_wait is not None:
            try:
                df = merge_contiguous_stop_wait(df, max_gap_pts=1)
            except Exception:
                pass

        if keep_only:
            df = _limit_and_space_stops(df, max_events=max_events, min_sep_pts=min_sep_pts)
            if not preserve_after:
                df = _enforce_min_stop_spacing(df, min_spacing_s=min_gap_s, hz=self.opt.hz)
        return df

    def _ensure_final_full_stop(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "speed" not in df.columns:
            return df
        n = len(df)
        window = max(1, min(int(self.opt.enforce_final_stop_sec * self.opt.hz), n))
        start = n - window
        v0 = float(df.iloc[start]["speed"]) if not pd.isna(df.iloc[start]["speed"]) else 0.0
        df.loc[start:n - 1, "speed"] = np.linspace(v0, 0.0, window)
        if "acc_x" in df.columns:
            df.loc[start:n - 1, "acc_x"] = -1.5
        for col in ("acc_y", "acc_z"):
            if col in df.columns:
                df.loc[start:n - 1, col] = 0.0
        df["event"] = df["event"].astype("object")
        mask = df.loc[start:n - 1, "event"].isna()
        if mask.any():
            df.loc[start:n - 1, "event"] = df.loc[start:n - 1, "event"].where(~mask, "stop")
        return df

    # ---- pipeline ----
    def run(self, df: pd.DataFrame, scenario_stops_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        logger.info("🎯 Lancement du pipeline de simulation...")

        df = self._ensure_dt(df)
        df = self._ensure_event_column(df)

        # 1) Contexte route
        df = enrich_road_type_stream(df)

        # 2) Stops/waits depuis scénario
        if apply_stop_wait_at_positions is not None and scenario_stops_df is not None and not scenario_stops_df.empty:
            events_df = scenario_stops_df[["lat", "lon", "event"]] if "event" in scenario_stops_df else scenario_stops_df
            window_m = float(self.cfg.get("events", {}).get("stop_wait", {}).get("from_positions_window_m", 250.0))
            min_d = int(self.cfg.get("events", {}).get("stop_wait", {}).get("min_duration_pts", 20))
            max_d = int(self.cfg.get("events", {}).get("stop_wait", {}).get("max_duration_pts", 120))
            fn, kwargs = self._call_with_supported_kwargs(
                apply_stop_wait_at_positions,
                df=df,
                events_df=events_df,
                positions_df=events_df,
                coords_df=events_df,
                events=events_df,
                window_m=window_m,
                min_duration_pts=min_d,
                max_duration_pts=max_d,
            )
            if fn is not None:
                df = fn(df, **{k: v for k, v in kwargs.items() if k != "df"})  # type: ignore
                logger.info("[stop_wait] Scénario appliqué (%d points).", len(scenario_stops_df))

        # 3) Vitesse plateaux
        try:
            if simulate_variable_speed is not None:
                df = simulate_variable_speed(df, self.cfg)
                logger.info("[speed] simulate_variable_speed appliqué (target_speed/speed définis).")
            else:
                raise RuntimeError("simulate_variable_speed indisponible")
        except Exception as e:
            logger.warning("[speed] simulate_variable_speed KO (%s). Fallback par type de route.", e)
            default_map = {
                "motorway": 110.0, "primary": 70.0, "secondary": 50.0,
                "tertiary": 40.0, "residential": 30.0, "service": 50.0, "unknown": 50.0,
            }
            df["target_speed"] = df["road_type"].map(default_map).fillna(50.0).astype(float)
            df["speed"] = df["target_speed"]

        # mini “snap” hors événements (réduit l’écart à la cible)
        if {"speed", "target_speed", "event"}.issubset(df.columns):
            mask = df["event"].isna()
            tol = float(self.cfg.get("simulation", {}).get("speed_adjust_tolerance_kmh", 2.0))
            diff = (df["speed"] - df["target_speed"]).abs()
            df.loc[mask & (diff <= tol), "speed"] = df.loc[mask & (diff <= tol), "target_speed"]

        # ré-aligner légèrement la cible vers la vitesse réalisée pour améliorer la cohérence
        # (évite un écart artificiel entre speed moyen GPS et "vitesse déclarée")
        if {"speed", "target_speed", "event"}.issubset(df.columns):
            mask = df["event"].isna()
            tol_decl = float(self.cfg.get("simulation", {}).get("target_snap_tolerance_kmh", 3.0))
            diff = (df["speed"] - df["target_speed"]).abs()
            df.loc[mask & (diff <= tol_decl), "target_speed"] = df.loc[mask & (diff <= tol_decl), "speed"]

        # 4) Accélération initiale (avec garde-fou)
        if inject_initial_acceleration is not None and not df["event"].astype("object").eq("acceleration_initiale").any():
            try:
                v_kmh = float(self.cfg.get("simulation", {}).get("target_speed_kmh", 40.0))
                df = inject_initial_acceleration(df, v_kmh, duration=5.0)
                logger.info("[INITIAL] Accélération initiale injectée.")
            except Exception as e:
                logger.warning("[INITIAL] Ignorée (%s)", e)

        # 5) Événements ponctuels
        if self.opt.inject_punctual_events:
            df = inject_all_events(df, self.cfg)

        # 6) Décélération finale (avec garde-fou)
        if inject_final_deceleration is not None and not df["event"].astype("object").eq("deceleration_finale").any():
            try:
                v_kmh = float(self.cfg.get("simulation", {}).get("target_speed_kmh", 40.0))
                df = inject_final_deceleration(df, v_kmh, duration=5.0)
                logger.info("[FINAL] Décélération finale injectée.")
            except Exception as e:
                logger.warning("[FINAL] Ignorée (%s)", e)

        # 7) Nettoyage + policy stops (avant reprojection)
        df = clean_invalid_events(df)
        df = self._apply_stop_policy(df)
        df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
        if {"lat", "lon", "timestamp"}.issubset(df.columns):
            df = df[df[["lat", "lon", "timestamp"]].notna().all(axis=1)]
        df = df.reset_index(drop=True)

        # 8) Reprojection
        df = reprojection.spatial_reprojection(df, speed_target=float(self.cfg.get("simulation", {}).get("target_speed_kmh", 40.0)))
        self._log_events(df, "Après reprojection (spatial_reprojection)")

        # 9) Cinématique + resampling
        df = kinematics.calculate_heading(df)
        if {"heading", "target_speed"}.issubset(df.columns):
            df = df[df["heading"].notna() & df["target_speed"].notna()].reset_index(drop=True)
        df = kinematics.calculate_linear_acceleration(df, freq_hz=self.opt.hz)
        df = kinematics.calculate_angular_velocity(df, freq_hz=self.opt.hz)
        df = reprojection.resample_time(df, freq_hz=self.opt.hz)
        self._log_events(df, "Après reprojection (resample_time)")

        # 10) Clamp dv/dt + stop final + gyro/acc bruit
        df = _clamp_speed_changes(df, hz=self.opt.hz, a_max_mps2=float(self.cfg.get("simulation", {}).get("a_max_mps2", 2.0)))
        # Sécurité: interdire toute vitesse négative résiduelle
        if "speed" in df.columns:
            df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
            df["speed"] = df["speed"].clip(lower=0.0)
        df = self._apply_stop_policy(df)  # consolidation/espacement post-resample
        df = self._ensure_final_full_stop(df)

        df = generate_gyroscope_signals(df)

        if inject_inertial_noise is not None:
            noise_cfg = self.cfg.get("inertial_noise", {})
            def _std(axis, default):
                try:
                    return float(noise_cfg.get(axis, {}).get("std", default))
                except Exception:
                    return default
            acc_std = max(_std("acc_x", 0.25), _std("acc_y", 0.25), _std("acc_z", 0.25))
            gyro_std = max(_std("gyro_x", 0.2), _std("gyro_y", 0.2), _std("gyro_z", 0.2))
            df = inject_inertial_noise(df, {"acc_std": acc_std, "gyro_std": gyro_std, "acc_bias": 0.02, "gyro_bias": 0.005})

        # 11) Validations légères
        df = complete_trajectory_fields(df, self.cfg)
        df = ensure_strictly_increasing_timestamps(df)
        try:
            validation.validate_timestamps(df)
            validation.validate_spatial_coherence(df, max_speed=130)
        except Exception as e:
            logger.warning("Validation légère: %s", e)

        self._log_events(df, "Final (avant export)")
        return df
