"""The single HTTP client every data source goes through.

Why one module: retries, timeouts, politeness, and error types are the same for every
API, so they live in one place and are tested once. Callers get either a successful
`requests.Response` or one of two typed exceptions, never a raw network error.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import requests

from skytrust.config import HttpSettings

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class SourceUnavailableError(Exception):
    """The source could not be reached or kept failing after all retries (outage, rate limit)."""


class BadResponseError(Exception):
    """The source answered, but the payload is unusable (4xx, empty, malformed)."""


class HttpClient:
    def __init__(
        self,
        settings: HttpSettings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = settings.user_agent
        # Injected so tests don't actually wait during backoff.
        self._sleep = sleep
        self._last_call = 0.0
        # Every real network attempt is counted, so a run can prove it was served from cache.
        self.n_requests = 0

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter: spreads retries out so we don't hammer
        a struggling server in lockstep."""
        return random.uniform(0, self.settings.backoff_base_s * 2**attempt)

    def _polite_wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.settings.polite_delay_s:
            self._sleep(self.settings.polite_delay_s - elapsed)

    def get(self, url: str, params: dict[str, Any] | list[tuple[str, Any]]) -> requests.Response:
        """GET with retries on 429/5xx and connection errors. Raises typed errors."""
        last_problem = ""
        for attempt in range(self.settings.max_attempts):
            self._polite_wait()
            self.n_requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.settings.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 400:
                    if not resp.content.strip():
                        raise BadResponseError(f"Empty response from {url}")
                    return resp
                if resp.status_code not in RETRY_STATUS:
                    raise BadResponseError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}")
                last_problem = f"HTTP {resp.status_code}"
            finally:
                self._last_call = time.monotonic()
            wait = self._backoff(attempt)
            log.warning(
                "%s from %s (attempt %d); retrying in %.1fs", last_problem, url, attempt + 1, wait
            )
            if attempt < self.settings.max_attempts - 1:
                self._sleep(wait)
        raise SourceUnavailableError(
            f"{url} failed after {self.settings.max_attempts} attempts: {last_problem}"
        )

    def get_json(self, url: str, params: dict[str, Any] | list[tuple[str, Any]]) -> dict[str, Any]:
        resp = self.get(url, params)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise BadResponseError(f"Malformed JSON from {url}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise BadResponseError(f"Error payload from {url}: {str(payload)[:300]}")
        return payload
