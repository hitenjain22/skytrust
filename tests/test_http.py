from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from skytrust.config import HttpSettings
from skytrust.data.http import BadResponseError, HttpClient, SourceUnavailableError

SETTINGS = HttpSettings(
    timeout_s=30, max_attempts=5, backoff_base_s=1.0, polite_delay_s=0.0, user_agent="test-agent"
)


def fake_response(status: int, body: bytes = b'{"ok": true}') -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.content = body
    resp.text = body.decode()
    resp.json.side_effect = lambda: requests.models.complexjson.loads(body)
    return resp


def make_client(*responses) -> tuple[HttpClient, MagicMock, list[float]]:
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = list(responses)
    sleeps: list[float] = []
    return HttpClient(SETTINGS, session=session, sleep=sleeps.append), session, sleeps


def test_sets_user_agent_and_timeout():
    client, session, _ = make_client(fake_response(200))
    client.get("https://x.test", {"a": 1})
    assert session.headers["User-Agent"] == "test-agent"
    assert session.get.call_args.kwargs["timeout"] == 30


def test_503_retries_then_raises_source_unavailable():
    client, session, sleeps = make_client(*[fake_response(503)] * 5)
    with pytest.raises(SourceUnavailableError):
        client.get("https://x.test", {})
    assert session.get.call_count == 5
    assert len(sleeps) == 4  # no pointless sleep after the final attempt


def test_retry_recovers_after_transient_429():
    client, session, _ = make_client(fake_response(429), fake_response(200))
    assert client.get_json("https://x.test", {}) == {"ok": True}
    assert session.get.call_count == 2


def test_connection_error_is_retried():
    client, session, _ = make_client(requests.ConnectionError("boom"), fake_response(200))
    client.get("https://x.test", {})
    assert session.get.call_count == 2


def test_4xx_is_bad_response_without_retry():
    client, session, _ = make_client(fake_response(400, b'{"error": true}'))
    with pytest.raises(BadResponseError):
        client.get("https://x.test", {})
    assert session.get.call_count == 1


@pytest.mark.parametrize(
    "body", [b"", b"   ", b"<html>not json", b'{"error": true, "reason": "x"}']
)
def test_empty_or_malformed_payload_is_bad_response(body):
    client, _, _ = make_client(fake_response(200, body))
    with pytest.raises(BadResponseError):
        client.get_json("https://x.test", {})


def test_backoff_grows_exponentially():
    client, _, _ = make_client()
    # Full jitter: each wait is uniform in [0, base * 2**attempt].
    assert all(0 <= client._backoff(0) <= 1.0 for _ in range(50))
    assert all(0 <= client._backoff(3) <= 8.0 for _ in range(50))
