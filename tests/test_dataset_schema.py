

import pandas as pd
import yaml
import pytest

from core.exporters import enforce_schema_order


def write_schema(tmp_path, columns):
    schema_path = tmp_path / "dataset_schema.yaml"
    with open(schema_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"version": "1.0", "columns": [{"name": c} for c in columns]}, f, sort_keys=False)
    return str(schema_path)


def test_enforce_adds_missing_and_orders_columns(tmp_path):
    # Given a schema order
    ordered = [
        "timestamp", "lat", "lon", "altitude_m", "speed",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "in_delivery", "delivery_state", "event",
        "event_infra", "event_behavior", "event_context",
    ]
    schema_path = write_schema(tmp_path, ordered)

    # And a dataframe with shuffled/subset columns + an extra column
    df = pd.DataFrame({
        "lon": [2.35, 2.36],
        "lat": [48.85, 48.86],
        "timestamp": pd.date_range("2025-01-01", periods=2, freq="1s"),
        "speed": [10.0, 11.0],
        "extra_col": [1, 2],
    })

    out = enforce_schema_order(df.copy(), cfg={"schema_path": schema_path})

    # 1) Columns appear in the schema's order first
    assert list(out.columns[: len(ordered)]) == ordered

    # 2) Extra columns are preserved and appended at the end
    assert out.columns[-1] == "extra_col"

    # 3) Missing columns from schema are added (as NA) and lengths match
    missing = set(ordered) - set(df.columns)
    for col in missing:
        assert col in out.columns
        assert len(out[col]) == len(df)
        # NA-friendly check (pandas may use dtype 'object' or 'Float64')
        assert out[col].isna().sum() == len(df)


def test_idempotent_on_already_ordered_df(tmp_path):
    cols = ["a", "b", "c"]
    schema_path = write_schema(tmp_path, cols)
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    out1 = enforce_schema_order(df.copy(), cfg={"schema_path": schema_path})
    out2 = enforce_schema_order(out1.copy(), cfg={"schema_path": schema_path})

    # Idempotence: applying twice should keep the same order and values
    assert list(out1.columns) == cols
    assert out1.equals(out2)


def test_empty_dataframe_returns_as_is(tmp_path):
    schema_path = write_schema(tmp_path, ["x", "y"])  # any schema
    empty = pd.DataFrame()
    out = enforce_schema_order(empty.copy(), cfg={"schema_path": schema_path})
    # For empty input, we keep as-is (function is a no-op by design)
    assert out.empty
    assert list(out.columns) == []