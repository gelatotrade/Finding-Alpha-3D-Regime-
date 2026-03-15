"""
Resilience primitives: retry with backoff, circuit breaker, rate limiter.
Institutional-grade fault tolerance for all external API calls.
"""
import logging
import time
import threading
from collections import deque
from enum import Enum
from functools import wraps
from typing import Callable, Optional, Set, Tuple
import requests
from config.settings import RetryPolicy, CircuitBreakerConfig

logger = logging.getLogger(__name__)


# ── Rate Limiter (token bucket) ──────────────────────────────────────────

class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, max_per_minute: int):
        self._max = max_per_minute
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a token is available. Returns wait time in seconds."""
        with self._lock:
            now = time.monotonic()
            # Purge timestamps older than 60s
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                wait = 60 - (now - self._timestamps[0]) + 0.1
                time.sleep(wait)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > 60:
                    self._timestamps.popleft()
            self._timestamps.append(now)
            return 0.0


# ── Circuit Breaker ──────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures from flaky APIs."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        cfg = config or CircuitBreakerConfig()
        self._failure_threshold = cfg.failure_threshold
        self._recovery_timeout = cfg.recovery_timeout_s
        self._half_open_max = cfg.half_open_max_calls
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time > self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker '%s' OPEN after %d failures",
                               self.name, self._failure_count)

    def allow_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self._half_open_max:
                    self._half_open_calls += 1
                    return True
            return False
        return False


# ── Resilient HTTP Client ────────────────────────────────────────────────

class ResilientClient:
    """
    HTTP client with retry, circuit breaker, rate limiting, and
    response validation. Every data fetcher should use this.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        timeout: float = 15.0,
        retry_policy: Optional[RetryPolicy] = None,
        rate_limit_per_min: int = 60,
        default_headers: Optional[dict] = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry = retry_policy or RetryPolicy()
        self.rate_limiter = RateLimiter(rate_limit_per_min)
        self.circuit_breaker = CircuitBreaker(name)
        self.session = requests.Session()
        if default_headers:
            self.session.headers.update(default_headers)

        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    def get(self, path: str, params: Optional[dict] = None,
            headers: Optional[dict] = None) -> Optional[requests.Response]:
        return self._request("GET", path, params=params, headers=headers)

    def post(self, path: str, json: Optional[dict] = None,
             headers: Optional[dict] = None) -> Optional[requests.Response]:
        return self._request("POST", path, json=json, headers=headers)

    def _request(self, method: str, path: str, **kwargs) -> Optional[requests.Response]:
        if not self.circuit_breaker.allow_request():
            logger.warning("[%s] Circuit breaker OPEN — skipping request to %s",
                           self.name, path)
            return None

        url = f"{self.base_url}/{path.lstrip('/')}" if path else self.base_url

        for attempt in range(self.retry.max_retries + 1):
            try:
                self.rate_limiter.acquire()
                self._request_count += 1
                t0 = time.monotonic()

                resp = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                self._total_latency += time.monotonic() - t0

                if resp.status_code in self.retry.retry_on_status:
                    delay = min(
                        self.retry.base_delay_s * (self.retry.exponential_base ** attempt),
                        self.retry.max_delay_s,
                    )
                    logger.warning("[%s] HTTP %d on %s, retry %d/%d in %.1fs",
                                   self.name, resp.status_code, path,
                                   attempt + 1, self.retry.max_retries, delay)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                self.circuit_breaker.record_success()
                return resp

            except requests.exceptions.Timeout:
                self._error_count += 1
                logger.warning("[%s] Timeout on %s (attempt %d/%d)",
                               self.name, path, attempt + 1, self.retry.max_retries + 1)
            except requests.exceptions.ConnectionError:
                self._error_count += 1
                logger.warning("[%s] Connection error on %s (attempt %d/%d)",
                               self.name, path, attempt + 1, self.retry.max_retries + 1)
            except requests.exceptions.HTTPError as e:
                self._error_count += 1
                self.circuit_breaker.record_failure()
                logger.error("[%s] HTTP error: %s", self.name, e)
                return None

            if attempt < self.retry.max_retries:
                delay = min(
                    self.retry.base_delay_s * (self.retry.exponential_base ** attempt),
                    self.retry.max_delay_s,
                )
                time.sleep(delay)

        self.circuit_breaker.record_failure()
        logger.error("[%s] All %d retries exhausted for %s",
                     self.name, self.retry.max_retries + 1, path)
        return None

    @property
    def metrics(self) -> dict:
        return {
            "name": self.name,
            "requests": self._request_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "avg_latency_ms": (self._total_latency / max(self._request_count, 1)) * 1000,
            "circuit_state": self.circuit_breaker.state.value,
        }
