import json
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

try:
    from core.osmnx.client import enrich_road_type_stream
except Exception:
    enrich_road_type_stream = None

@pytest.fixture()
def sample_df():
    return pd.DataFrame({"lat":[49.2935,49.2900], "lon":[1.1103,1.1150]})

def _mocked_post(*args, **kwargs):
    class Resp:
        status_code = 200
        def json(self): return {"stream_id":"mock-stream-123"}
    return Resp()

def _mocked_get(*args, **kwargs):
    class Raw:
        def __iter__(self):
            for i in range(2):
                yield MagicMock(data=json.dumps({"index":i,"road_type":"residential"}))
    return MagicMock(raw=Raw())

@patch("core.osmnx.client.requests.post", side_effect=_mocked_post)
@patch("core.osmnx.client.requests.get", side_effect=_mocked_get)
@patch("core.osmnx.client.SSEClient", autospec=True)
def test_enrich_stream_mocked(mock_sse, mock_get, mock_post, sample_df):
    if enrich_road_type_stream is None:
        pytest.skip("API OSMnx client indisponible")
    mock_sse.return_value.__iter__.return_value = [
        MagicMock(data=json.dumps({"index":0,"road_type":"primary"})),
        MagicMock(data=json.dumps({"index":1,"road_type":"secondary"})),
    ]
    out = enrich_road_type_stream(sample_df)
    assert "road_type" in out.columns
    assert list(out["road_type"]) == ["primary","secondary"]