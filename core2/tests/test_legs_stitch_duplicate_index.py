import pandas as pd
import numpy as np

from core2.stages.legs_stitch import LegsStitch
from core2.context import Context

def _leg_with_duplicate_positions():
    # 5 points dont 3 sont identiques -> distances nulles -> dt_i=0 -> timestamps dupliqués
    return pd.DataFrame({
        "lat": [49.0, 49.0, 49.0, 49.001, 49.002],
        "lon": [1.0, 1.0, 1.0, 1.0005, 1.001],
    })

def _ctx_for_test():
    ctx = Context()
    ctx.cfg = {
        "sim": {"hz": 10},
        "legs_stitch": {"inject_stop_hold_s": 2.0},
    }
    ctx.artifacts = {
        "legs_traces": [_leg_with_duplicate_positions()],
        "legs_plan": {
            "stops": [
                {"id": "S0", "service_s": 0},
                {"id": "S1", "service_s": 2.0},
            ],
            "start_time_utc": "2024-01-01T00:00:00Z",
        },
        "legs_summary": [{"duration_s": 10.0}],
    }
    return ctx

def test_legs_stitch_reindex_crashes_with_duplicate_timestamp_index():
    """
    Version avant correctif: points identiques -> timestamps dupliqués -> ValueError attendu
    """
    stage = LegsStitch()
    ctx = _ctx_for_test()

    error = None
    try:
        stage.run(ctx)
    except Exception as e:
        error = e

    assert error is not None, "Le test doit provoquer une exception avant correction"
    assert isinstance(error, ValueError)
    assert "duplicate labels" in str(error)