"""Shared Redis client construction for portfolio applications."""

from __future__ import annotations

from redis import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

DEFAULT_CONNECT_TIMEOUT_SECONDS = 2.0
DEFAULT_SOCKET_TIMEOUT_SECONDS = 5.0


def create_redis_client(
    redis_url: str,
    *,
    decode_responses: bool = False,
) -> Redis:
    """Create a pooled Redis client with bounded connection and command timeouts."""
    return Redis.from_url(
        redis_url,
        decode_responses=decode_responses,
        socket_connect_timeout=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=DEFAULT_SOCKET_TIMEOUT_SECONDS,
        health_check_interval=30,
        retry=Retry(ExponentialBackoff(cap=0.5, base=0.05), retries=2),
    )
