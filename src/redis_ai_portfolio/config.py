"""Central, side-effect-light configuration for every portfolio example."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import find_dotenv, load_dotenv

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_WORKBENCH_MODEL_MODE = "live"
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_NAMESPACE = "portfolio"
DEFAULT_STM_TTL_MINUTES = 1440
DEFAULT_CACHE_LLM_INPUT_COST_PER_MILLION = 0.20
DEFAULT_CACHE_LLM_OUTPUT_COST_PER_MILLION = 1.20
DEFAULT_CACHE_EMBEDDING_COST_PER_MILLION = 0.02


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}")


def _parse_int(name: str, value: str | None, *, default: int, minimum: int = 0) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _parse_float(
    name: str,
    value: str | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def build_redis_url(
    *,
    host: str = DEFAULT_REDIS_HOST,
    port: int = DEFAULT_REDIS_PORT,
    database: int = DEFAULT_REDIS_DB,
    username: str | None = None,
    password: str | None = None,
    ssl: bool = False,
) -> str:
    """Build a Redis URL without emitting an empty authentication segment."""
    clean_host = host.strip()
    if not clean_host:
        raise ValueError("REDIS_HOST cannot be empty")
    if not 1 <= port <= 65535:
        raise ValueError("REDIS_PORT must be between 1 and 65535")
    if database < 0:
        raise ValueError("REDIS_DB cannot be negative")

    host_part = clean_host
    if ":" in host_part and not host_part.startswith("["):
        host_part = f"[{host_part}]"

    clean_username = (username or "").strip()
    clean_password = password or ""
    auth = ""
    if clean_username:
        auth = quote(clean_username, safe="")
        if clean_password:
            auth += f":{quote(clean_password, safe='')}"
        auth += "@"
    elif clean_password:
        auth = f":{quote(clean_password, safe='')}@"

    scheme = "rediss" if ssl else "redis"
    return f"{scheme}://{auth}{host_part}:{port}/{database}"


def redact_redis_url(redis_url: str) -> str:
    """Return a printable Redis URL with credentials removed."""
    parsed = urlsplit(redis_url)
    host = parsed.hostname or DEFAULT_REDIS_HOST
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))


def load_portfolio_environment(env_file: str | Path | None = None) -> None:
    """Load a local .env without replacing variables supplied by the caller."""
    dotenv_path = str(env_file) if env_file else find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


@dataclass(frozen=True, slots=True)
class PortfolioSettings:
    redis_url: str
    redis_namespace: str
    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    cache_distance_threshold: float
    cache_ttl_seconds: int
    stm_ttl_minutes: int = DEFAULT_STM_TTL_MINUTES
    stm_refresh_ttl_on_read: bool = True
    cache_llm_input_cost_per_million: float = (
        DEFAULT_CACHE_LLM_INPUT_COST_PER_MILLION
    )
    cache_llm_output_cost_per_million: float = (
        DEFAULT_CACHE_LLM_OUTPUT_COST_PER_MILLION
    )
    cache_embedding_cost_per_million: float = (
        DEFAULT_CACHE_EMBEDDING_COST_PER_MILLION
    )
    workbench_model_mode: str = DEFAULT_WORKBENCH_MODEL_MODE

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> PortfolioSettings:
        load_portfolio_environment(env_file)

        explicit_url = os.getenv("REDIS_URL", "").strip()
        if explicit_url:
            parsed = urlsplit(explicit_url)
            if parsed.scheme not in {"redis", "rediss"}:
                raise ValueError("REDIS_URL must use the redis:// or rediss:// scheme")
            redis_url = explicit_url
        else:
            redis_url = build_redis_url(
                host=os.getenv("REDIS_HOST", "").strip() or DEFAULT_REDIS_HOST,
                port=_parse_int(
                    "REDIS_PORT",
                    os.getenv("REDIS_PORT"),
                    default=DEFAULT_REDIS_PORT,
                    minimum=1,
                ),
                database=_parse_int(
                    "REDIS_DB",
                    os.getenv("REDIS_DB"),
                    default=DEFAULT_REDIS_DB,
                ),
                username=os.getenv("REDIS_USERNAME"),
                password=os.getenv("REDIS_PASSWORD"),
                ssl=_parse_bool(os.getenv("REDIS_SSL")),
            )

        namespace = os.getenv("REDIS_NAMESPACE", DEFAULT_REDIS_NAMESPACE).strip(": ")
        if not namespace:
            raise ValueError("REDIS_NAMESPACE cannot be empty")

        embedding_model = (
            os.getenv("OPENAI_EMBEDDING_MODEL")
            or os.getenv("OPEN_AI_EMBEDDING_MODEL")
            or DEFAULT_OPENAI_EMBEDDING_MODEL
        )

        workbench_model_mode = os.getenv(
            "WORKBENCH_MODEL_MODE",
            DEFAULT_WORKBENCH_MODEL_MODE,
        ).strip().casefold()
        if workbench_model_mode not in {"live", "demo"}:
            raise ValueError("WORKBENCH_MODEL_MODE must be live or demo")

        return cls(
            redis_url=redis_url,
            redis_namespace=namespace,
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
            openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
            or DEFAULT_OPENAI_MODEL,
            openai_embedding_model=embedding_model.strip(),
            cache_distance_threshold=_parse_float(
                "CACHE_DISTANCE_THRESHOLD",
                os.getenv("CACHE_DISTANCE_THRESHOLD"),
                default=0.2,
                minimum=0.0,
                maximum=2.0,
            ),
            cache_ttl_seconds=_parse_int(
                "CACHE_TTL_SECONDS",
                os.getenv("CACHE_TTL_SECONDS"),
                default=3600,
                minimum=1,
            ),
            stm_ttl_minutes=_parse_int(
                "STM_TTL_MINUTES",
                os.getenv("STM_TTL_MINUTES"),
                default=DEFAULT_STM_TTL_MINUTES,
                minimum=1,
            ),
            stm_refresh_ttl_on_read=_parse_bool(
                os.getenv("STM_REFRESH_TTL_ON_READ"),
                default=True,
            ),
            cache_llm_input_cost_per_million=_parse_float(
                "CACHE_LLM_INPUT_COST_PER_MILLION",
                os.getenv("CACHE_LLM_INPUT_COST_PER_MILLION"),
                default=DEFAULT_CACHE_LLM_INPUT_COST_PER_MILLION,
                minimum=0.0,
                maximum=1000.0,
            ),
            cache_llm_output_cost_per_million=_parse_float(
                "CACHE_LLM_OUTPUT_COST_PER_MILLION",
                os.getenv("CACHE_LLM_OUTPUT_COST_PER_MILLION"),
                default=DEFAULT_CACHE_LLM_OUTPUT_COST_PER_MILLION,
                minimum=0.0,
                maximum=1000.0,
            ),
            cache_embedding_cost_per_million=_parse_float(
                "CACHE_EMBEDDING_COST_PER_MILLION",
                os.getenv("CACHE_EMBEDDING_COST_PER_MILLION"),
                default=DEFAULT_CACHE_EMBEDDING_COST_PER_MILLION,
                minimum=0.0,
                maximum=1000.0,
            ),
            workbench_model_mode=workbench_model_mode,
        )

    def redis_name(self, *parts: str) -> str:
        """Build a lowercase, colon-separated Redis key or index name."""
        segments = [self.redis_namespace, *parts]
        cleaned = [segment.strip(": ").lower().replace(" ", "-") for segment in segments]
        if any(not segment for segment in cleaned):
            raise ValueError("Redis name segments cannot be empty")
        return ":".join(cleaned)


@lru_cache(maxsize=1)
def get_settings() -> PortfolioSettings:
    return PortfolioSettings.from_env()
