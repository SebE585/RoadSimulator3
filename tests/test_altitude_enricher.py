

import pandas as pd
import pytest

import enrichments.altitude_enricher as ae


def make_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "lat": [48.85 + i*0.001 for i in range(n)],
        "lon": [2.35 + i*0.001 for i in range(n)],
    })


def test_returns_df_with_altitude_fields(monkeypatch):
    df = make_df(3)

    fake_json = [
        {"altitude": 100+i, "altitude_smoothed": 101+i, "slope_percent": 2.0+i}
        for i in range(len(df))
    ]

    class FakeResp:
        def raise_for_status(self): return None
        def json(self): return fake_json

    def fake_post(url, json, timeout=30.0):
        return FakeResp()

    monkeypatch.setattr(ae.requests, "post", fake_post)

    out = ae.enrich_terrain_via_api(df.copy())

    for col in ("altitude", "altitude_smoothed", "slope_percent", "altitude_m"):
        assert col in out.columns
        assert len(out[col]) == len(df)
    # altitude_m must equal altitude
    assert out["altitude_m"].equals(out["altitude"].astype("float32"))


def test_invalid_json_length(monkeypatch):
    df = make_df(2)

    class FakeResp:
        def raise_for_status(self): return None
        def json(self): return [{"altitude": 1}]  # wrong length

    def fake_post(url, json, timeout=30.0):
        return FakeResp()

    monkeypatch.setattr(ae.requests, "post", fake_post)

    with pytest.raises(ValueError):
        ae.enrich_terrain_via_api(df.copy())


def test_enrich_altitude_respects_provider_none(monkeypatch):
    df = make_df(2)
    out = ae.enrich_altitude(df.copy(), config={"dataset": {"altitude": {"provider": "none"}}})
    # Should not call requests.post, just return unchanged
    assert "altitude" not in out.columns
    assert out.equals(df)