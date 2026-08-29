"""Preflight checks for a runnable Redis AI portfolio environment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from redis import Redis
from redis.exceptions import RedisError
from rich.console import Console
from rich.table import Table

from .config import PortfolioSettings, redact_redis_url
from .redis import create_redis_client

REQUIRED_RESOURCES = (
    "resources/movies.json",
    "resources/10-K-Q4-2023-As-Filed.pdf",
    "resources/2022-chevrolet-commercial-colorado-ebrochure.pdf",
    "resources/nke-10k-2023.pdf",
)


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


def _command_available(client: Redis, command: str) -> bool:
    response: Any = client.execute_command("COMMAND", "INFO", command)
    if isinstance(response, dict):
        return bool(response)
    return bool(response and response[0])


def run_checks(
    settings: PortfolioSettings,
    *,
    root: Path,
    skip_redis: bool = False,
) -> list[Check]:
    checks = [
        Check(
            "Python",
            Status.PASS if sys.version_info >= (3, 13) else Status.FAIL,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        Check("OpenAI model", Status.PASS, settings.openai_model),
        Check(
            "STM expiry",
            Status.PASS,
            (
                f"{settings.stm_ttl_minutes} minutes "
                f"({'sliding' if settings.stm_refresh_ttl_on_read else 'fixed'})"
            ),
        ),
        Check(
            "Semantic cache",
            Status.PASS,
            (
                f"distance ≤ {settings.cache_distance_threshold:g} "
                f"(similarity ≥ {1 - settings.cache_distance_threshold:g}), "
                f"TTL {settings.cache_ttl_seconds}s"
            ),
        ),
        Check(
            "OpenAI API key",
            Status.PASS if settings.openai_api_key else Status.WARN,
            "configured" if settings.openai_api_key else "missing; OpenAI-backed demos will not run",
        ),
    ]

    missing = [resource for resource in REQUIRED_RESOURCES if not (root / resource).is_file()]
    checks.append(
        Check(
            "Portfolio resources",
            Status.FAIL if missing else Status.PASS,
            f"missing: {', '.join(missing)}" if missing else f"{len(REQUIRED_RESOURCES)} files found",
        )
    )
    checks.append(
        Check(
            "Environment template",
            Status.PASS if (root / ".env.example").is_file() else Status.FAIL,
            ".env.example",
        )
    )

    if skip_redis:
        checks.append(Check("Redis", Status.WARN, "connection checks skipped"))
        return checks

    client = create_redis_client(settings.redis_url, decode_responses=True)
    try:
        client.ping()
        server = client.info("server")
        checks.append(
            Check(
                "Redis connection",
                Status.PASS,
                f"{redact_redis_url(settings.redis_url)} (Redis {server.get('redis_version', 'unknown')})",
            )
        )
        search_available = _command_available(client, "FT.CREATE")
        json_available = _command_available(client, "JSON.SET")
        checks.append(
            Check(
                "Redis Search",
                Status.PASS if search_available else Status.FAIL,
                "FT.CREATE available" if search_available else "FT.CREATE unavailable",
            )
        )
        checks.append(
            Check(
                "Redis JSON",
                Status.PASS if json_available else Status.FAIL,
                "JSON.SET available" if json_available else "JSON.SET unavailable",
            )
        )
    except RedisError as exc:
        checks.append(
            Check(
                "Redis connection",
                Status.FAIL,
                f"{redact_redis_url(settings.redis_url)}: {exc}",
            )
        )
    finally:
        client.close()

    return checks


def _render(checks: list[Check]) -> None:
    table = Table(title="Redis AI Portfolio Doctor", show_lines=False)
    table.add_column("Status", no_wrap=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Detail")

    colors = {Status.PASS: "green", Status.WARN: "yellow", Status.FAIL: "red"}
    for check in checks:
        table.add_row(
            f"[{colors[check.status]}]{check.status}[/{colors[check.status]}]",
            check.name,
            check.detail,
        )
    Console().print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis connectivity checks")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    try:
        settings = PortfolioSettings.from_env(args.root / ".env")
        checks = run_checks(settings, root=args.root.resolve(), skip_redis=args.skip_redis)
    except ValueError as exc:
        checks = [Check("Configuration", Status.FAIL, str(exc))]

    _render(checks)
    failed = any(check.status is Status.FAIL for check in checks)
    warned = any(check.status is Status.WARN for check in checks)
    if failed or (args.strict and warned):
        Console().print("\nRun [bold]make redis-start[/bold] and review [bold].env.example[/bold], then retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
