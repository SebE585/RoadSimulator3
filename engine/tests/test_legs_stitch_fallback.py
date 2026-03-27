import pandas as pd
import numpy as np

from engine.stages.legs_stitch import LegsStitch
from engine.context import Context
from rs3_contracts.api import Result


def _mk_leg(latlon):
    # un "leg" minimal comme produit par LegsRoute: DataFrame avec lat/lon
    return pd.DataFrame(latlon, columns=["lat", "lon"])

def test_no_final_hold_and_reasonable_duration_when_no_summaries():
    # Deux legs (3 stops): depot start -> client -> depot end
    leg1 = _mk_leg([(49.0, 1.0), (49.0005, 1.001)])           # ~130 m
    leg2 = _mk_leg([(49.0005, 1.001), (49.0, 1.0)])           # ~130 m (retour)

    # Plan avec un arrêt (service) au client uniquement; depot end n'a pas de service
    stops = [
        {"id": "DEPOT_START", "is_depot": True,  "is_start": True,  "is_end": False, "service_s": 0},
        {"id": "CLIENT_1",    "is_depot": False, "is_start": False, "is_end": False, "service_s": 120},
        {"id": "DEPOT_END",   "is_depot": True,  "is_start": False, "is_end": True,  "service_s": 0},
    ]

    ctx = Context(cfg={})
    ctx.cfg = {
        "sim": {"hz": 10},
        # important: pas de legs_summary -> on trigger le fallback
        "legs_stitch": {
            "inject_stop_hold_s": 5.0,
            # la correction ajoute ce paramètre:
            "fallback_move_speed_mps": 11.11,  # ≈40 km/h
        },
    }
    ctx.artifacts["legs_traces"] = [leg1, leg2]
    ctx.artifacts["legs_plan"] = {"stops": stops}

    res = LegsStitch().run(ctx)
    # Be tolerant to different Result interfaces
    if isinstance(res, Result):
        ok = getattr(res, "ok", True)
        msg = getattr(res, "message", "")
    elif isinstance(res, tuple) and len(res) == 2:
        ok, msg = res
    else:
        ok, msg = True, ""
    assert ok, msg

    df = ctx.df
    # 1) On ne doit pas avoir de STOP injecté au dernier timestamp (depot end)
    last_row = df.iloc[-1]
    if "event" in df.columns:
        assert last_row["event"] != "STOP"

    # 2) Durée totale raisonnable: ~260 m à 40 km/h (~23.4 s) + 120 s de service
    #    On autorise une marge (±20 %) pour l'interpolation et l'arrondi.
    start_ts, end_ts = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    duration_s = (pd.to_datetime(end_ts) - pd.to_datetime(start_ts)).total_seconds()
    assert 115 <= duration_s <= 175, f"duration_s={duration_s}"

    # 3) Index strictement croissant, sans doublons
    ts = pd.to_datetime(df["timestamp"])
    assert ts.is_monotonic_increasing
    assert ts.duplicated().sum() == 0