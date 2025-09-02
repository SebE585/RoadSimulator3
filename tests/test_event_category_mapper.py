

import pandas as pd
import numpy as np
import pytest

from enrichments.event_category_mapper import project_event_categories


def make_df(events):
    n = len(events)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="100L"),
        "lat": np.linspace(48.85, 48.86, n),
        "lon": np.linspace(2.35, 2.36, n),
        "event": events,
    })


def test_inline_config_mapping():
    df = make_df(["nid_de_poule", "acceleration_initiale", "meteo_pluie", None])

    config = {
        "dataset": {
            "events": {
                "categories": {
                    "categories": ["infra", "behavior", "context"],
                    "mapping": {
                        "nid_de_poule": "infra",
                        "acceleration_initiale": "behavior",
                        "meteo_pluie": "context",
                    },
                }
            }
        }
    }

    out = project_event_categories(df.copy(), config=config)

    assert {"event_infra", "event_behavior", "event_context"}.issubset(out.columns)
    assert out.loc[0, "event_infra"] == 1
    assert out.loc[1, "event_behavior"] == 1
    assert out.loc[2, "event_context"] == 1
    # None should not trigger any flag
    assert out.loc[3, ["event_infra", "event_behavior", "event_context"]].sum() == 0


def test_yaml_fallback_mapping(tmp_path: pytest.TempPathFactory):
    # Create a minimal schema_path and event_categories.yaml in the same dir
    schema_path = tmp_path / "dataset_schema.yaml"
    schema_path.write_text("version: '1.0'\ncolumns: []\n", encoding="utf-8")

    yaml_path = tmp_path / "event_categories.yaml"
    yaml_path.write_text(
        """
        version: "1.0"
        categories: [infra, behavior, context]
        mapping:
          dos_d_ane: infra
          freinage_fort: behavior
          zone_urbaine_dense: context
        """.strip(),
        encoding="utf-8",
    )

    df = make_df(["dos_d_ane", "freinage_fort", "zone_urbaine_dense", "inconnu"])  # last is unknown
    config = {"schema_path": str(schema_path)}

    out = project_event_categories(df.copy(), config=config)

    assert {"event_infra", "event_behavior", "event_context"}.issubset(out.columns)
    assert out.loc[0, "event_infra"] == 1
    assert out.loc[1, "event_behavior"] == 1
    assert out.loc[2, "event_context"] == 1
    # Unknown should not set any flag
    assert out.loc[3, ["event_infra", "event_behavior", "event_context"]].sum() == 0


def test_no_event_column_creates_zeros(tmp_path):
    # YAML defines two categories; DataFrame has no 'event' column
    schema_path = tmp_path / "dataset_schema.yaml"
    schema_path.write_text("version: '1.0'\ncolumns: []\n", encoding="utf-8")
    (tmp_path / "event_categories.yaml").write_text(
        """
        version: "1.0"
        categories: [infra, behavior]
        mapping: {nid_de_poule: infra}
        """.strip(),
        encoding="utf-8",
    )

    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="100L"),
        "lat": [48.0, 48.0, 48.0],
        "lon": [2.0, 2.0, 2.0],
    })

    out = project_event_categories(df.copy(), config={"schema_path": str(schema_path)})

    assert {"event_infra", "event_behavior"}.issubset(out.columns)
    # no event column initially → created and all zeros in category flags
    assert out[["event_infra", "event_behavior"]].sum().sum() == 0


def test_preserves_event_column():
    df = make_df(["nid_de_poule"])  # one row
    out = project_event_categories(df.copy(), config={
        "dataset": {
            "events": {
                "categories": {
                    "categories": ["infra"],
                    "mapping": {"nid_de_poule": "infra"},
                }
            }
        }
    })
    # event column still present
    assert "event" in out.columns
    # and category projected
    assert out.loc[0, "event_infra"] == 1