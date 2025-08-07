import pytest
import json
import requests
from unittest.mock import patch, MagicMock
from core.osmnx.client import enrich_road_type_stream
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "lat": [49.2935, 49.2900],
        "lon": [1.1103, 1.1150]
    })


def mocked_post(*args, **kwargs):
    """Mock pour requests.post (stream init)."""
    class MockResponse:
        def __init__(self):
            self.status_code = 200
        def json(self):
            return {"stream_id": "mock-stream-123"}

    return MockResponse()


def mocked_get(*args, **kwargs):
    """Mock pour requests.get (flux SSE)."""
    class MockSSEClient:
        def __iter__(self):
            for i in range(2):
                yield MagicMock(data=json.dumps({
                    "index": i,
                    "road_type": "residential"
                }))

    return MagicMock(raw=MockSSEClient())  # imite resp.raw

@patch("core.osmnx.client.requests.post", side_effect=mocked_post)
@patch("core.osmnx.client.requests.get", side_effect=mocked_get)
@patch("core.osmnx.client.SSEClient", autospec=True)
def test_enrich_road_type_stream_mocked(mock_sse, mock_get, mock_post, sample_df):
    """
    Teste enrich_road_type_stream() avec mocks pour POST, GET et SSEClient.
    """
    mock_sse.return_value.__iter__.return_value = [
        MagicMock(data=json.dumps({"index": 0, "road_type": "primary"})),
        MagicMock(data=json.dumps({"index": 1, "road_type": "secondary"})),
    ]

    df = enrich_road_type_stream(sample_df)

    assert "road_type" in df.columns
    assert list(df["road_type"]) == ["primary", "secondary"]
    print("[✅] Test mocké enrich_road_type_stream passé.")
