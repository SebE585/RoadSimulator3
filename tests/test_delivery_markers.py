

import pandas as pd
import numpy as np
import pytest

from enrichments.delivery_markers import apply_delivery_markers


def make_df(events):
    n = len(events)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="100L"),
        "lat": np.linspace(48.85, 48.86, n),
        "lon": np.linspace(2.35, 2.36, n),
        "event": events,
    })


def test_defaults_when_no_events():
    df = make_df([None, None, None, None])
    out = apply_delivery_markers(df.copy())

    assert "in_delivery" in out.columns
    assert "delivery_state" in out.columns
    # default: all 0 / "in_vehicle"
    assert out["in_delivery"].tolist() == [0, 0, 0, 0]
    assert out["delivery_state"].tolist() == ["in_vehicle"] * 4


def test_toggle_states_on_start_end_events():
    events = [None, "start_delivery", None, None, "end_delivery", None]
    df = make_df(events)
    out = apply_delivery_markers(df.copy())

    # indices: 0..5
    # after start at 1 → in_delivery=1 from 1..4, then end at 4 → 0 from 4..
    assert out["in_delivery"].tolist() == [0, 1, 1, 1, 0, 0]
    assert out["delivery_state"].tolist() == [
        "in_vehicle", "on_delivery", "on_delivery", "on_delivery", "in_vehicle", "in_vehicle"
    ]


def test_multiple_delivery_blocks_and_propagation():
    events = ["start_delivery", None, "end_delivery", None, "start_delivery", None]
    df = make_df(events)
    out = apply_delivery_markers(df.copy())

    assert out["in_delivery"].tolist() == [1, 1, 0, 0, 1, 1]
    assert out["delivery_state"].tolist() == [
        "on_delivery", "on_delivery", "in_vehicle", "in_vehicle", "on_delivery", "on_delivery"
    ]


def test_non_string_events_are_ignored():
    events = [np.nan, {"k": 1}, 42, "start_delivery", None]
    df = make_df(events)
    out = apply_delivery_markers(df.copy())

    # Only switches at index 3
    assert out["in_delivery"].tolist() == [0, 0, 0, 1, 1]
    assert out["delivery_state"].tolist() == [
        "in_vehicle", "in_vehicle", "in_vehicle", "on_delivery", "on_delivery"
    ]


def test_dtype_and_values():
    df = make_df(["start_delivery", None, "end_delivery"])  # 3 rows
    out = apply_delivery_markers(df.copy())

    assert out["in_delivery"].dtype == "int8"
    assert set(out["in_delivery"].unique().tolist()).issubset({0, 1})
    assert out.loc[0, "delivery_state"] in ("on_delivery", "in_vehicle")