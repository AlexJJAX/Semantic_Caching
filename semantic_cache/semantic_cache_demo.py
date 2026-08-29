"""Run, calibrate, benchmark, and invalidate the local Redis semantic cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

from rich.console import Console
from rich.table import Table

from redis_ai_portfolio.config import PortfolioSettings, get_settings, redact_redis_url
from redis_ai_portfolio.redis import create_redis_client
from redis_ai_portfolio.semantic_cache import (
    CacheOutcome,
    CachePartition,
    CachePricing,
    CacheRequest,
    CacheResult,
    CalibrationPair,
    CalibrationReport,
    OpenAIBackend,
    RedisSemanticCacheStore,
    SemanticCache,
    calibrate_thresholds,
)

DEFAULT_CALIBRATION_DATASET = "semantic_cache/calibration_cases.json"
DEFAULT_TENANT = "acme"
DEFAULT_TASK = "support-faq"
DEFAULT_PROMPT_VERSION = "support-v1"
DEFAULT_PERMISSIONS = ("customer",)
DEFAULT_BASE_PROMPT = "What did Acme report as revenue in 2025?"
DEFAULT_SEMANTIC_PROMPT = "How much revenue did Acme report for 2025?"
DEFAULT_FALSE_HIT_PROMPT = "How much revenue did Acme report for 2024?"
DEFAULT_VOLATILE_PROMPT = "What is Acme's live share price right now?"
DEFAULT_INVALIDATION_TAG = "acme-annual-report-v1"

CONSOLE = Console()


@dataclass(slots=True)
class SemanticCacheApplication:
    cache: SemanticCache
    store: RedisSemanticCacheStore
    backend: OpenAIBackend
    redis_client: object

    def close(self) -> None:
        self.backend.close()
        self.redis_client.close()


def create_application(settings: PortfolioSettings) -> SemanticCacheApplication:
    """Create owned Redis/OpenAI resources and the exact-first cache-aside service."""
    if not settings.openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    redis_client = create_redis_client(settings.redis_url)
    backend = None
    try:
        redis_client.ping()
        backend = OpenAIBackend(
            api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
        )
        store = RedisSemanticCacheStore(settings, redis_client)
        semantic_cache = SemanticCache(
            store,
            backend,
            pricing=CachePricing.from_settings(settings),
            distance_threshold=settings.cache_distance_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        return SemanticCacheApplication(semantic_cache, store, backend, redis_client)
    except Exception:
        if backend is not None:
            backend.close()
        redis_client.close()
        raise


def _partition(args: argparse.Namespace, settings: PortfolioSettings) -> CachePartition:
    permissions = tuple(
        permission.strip()
        for permission in args.permissions.split(",")
        if permission.strip()
    )
    return CachePartition(
        tenant=args.tenant,
        task=args.task,
        model=settings.openai_model,
        prompt_version=args.prompt_version,
        permissions=permissions,
    )


def _result_table(results: Iterable[tuple[str, CacheResult]]) -> Table:
    table = Table(title="Cache-aside comparison")
    table.add_column("Scenario")
    table.add_column("Outcome")
    table.add_column("Latency", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("LLM tokens", justify="right")
    table.add_column("Saved tokens", justify="right")
    table.add_column("Request cost", justify="right")
    for scenario, result in results:
        hit = result.outcome in {CacheOutcome.EXACT_HIT, CacheOutcome.SEMANTIC_HIT}
        generated_tokens = result.generation_input_tokens + result.generation_output_tokens
        table.add_row(
            scenario,
            result.outcome.value,
            f"{result.latency_ms:.2f} ms",
            f"{result.similarity:.3f}" if result.similarity is not None else "—",
            "0" if hit else str(generated_tokens),
            str(generated_tokens) if hit else "0",
            f"${result.estimated_cost_usd:.8f}",
        )
    return table


def _metrics_table(cache: SemanticCache) -> Table:
    snapshot = cache.metrics.snapshot()
    table = Table(title="Semantic cache metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = (
        ("Requests", str(snapshot.requests)),
        ("Cache hit rate", f"{snapshot.hit_rate:.1%}"),
        ("Feedback false-hit rate", f"{snapshot.false_hit_rate:.1%}"),
        ("Guard rejections", str(snapshot.guard_rejections)),
        ("LLM calls", str(snapshot.llm_calls)),
        ("Generation tokens consumed", str(
            snapshot.generation_input_tokens + snapshot.generation_output_tokens
        )),
        ("Generation tokens saved", str(snapshot.generation_tokens_saved)),
        ("Embedding tokens", str(snapshot.embedding_tokens)),
        ("Latency p50", f"{snapshot.latency_p50_ms:.2f} ms"),
        ("Latency p95", f"{snapshot.latency_p95_ms:.2f} ms"),
        ("Actual estimated API cost", f"${snapshot.estimated_cost_usd:.8f}"),
        ("Avoided generation cost", f"${snapshot.estimated_cost_saved_usd:.8f}"),
        ("Net estimated savings", f"${snapshot.estimated_net_cost_savings_usd:.8f}"),
    )
    for name, value in rows:
        table.add_row(name, value)
    for outcome, latency in sorted(snapshot.latency_by_outcome.items()):
        table.add_row(
            f"{outcome} latency p50 / p95",
            f"{latency['p50_ms']:.2f} / {latency['p95_ms']:.2f} ms",
        )
    return table


def run_ask(args: argparse.Namespace, settings: PortfolioSettings) -> None:
    app = create_application(settings)
    try:
        request = CacheRequest(
            prompt=args.prompt,
            partition=_partition(args, settings),
            ttl_seconds=args.ttl,
            invalidation_tags=tuple(args.tag),
            force_miss=args.force_miss,
        )
        result = app.cache.answer(request)
        CONSOLE.print(_result_table([("request", result)]))
        if result.guard_rejections:
            CONSOLE.print(f"Guard rejections: {', '.join(result.guard_rejections)}")
        if result.bypass_reason:
            CONSOLE.print(f"Bypass reason: {result.bypass_reason}")
        CONSOLE.print("\nAnswer:\n", result.answer)
    finally:
        app.close()


def run_benchmark(args: argparse.Namespace, settings: PortfolioSettings) -> None:
    app = create_application(settings)
    partition = _partition(args, settings)
    tag = args.tag or DEFAULT_INVALIDATION_TAG
    try:
        removed = app.store.invalidate_partition(partition)
        CONSOLE.print(
            f"Starting with a cold partition ({removed} prior entr{'y' if removed == 1 else 'ies'} removed)."
        )
        results: list[tuple[str, CacheResult]] = []
        base_request = CacheRequest(
            args.base_prompt,
            partition,
            ttl_seconds=args.ttl,
            invalidation_tags=(tag,),
        )
        cold = app.cache.answer(base_request)
        results.append(("cold", cold))
        if cold.cache_key:
            ttl = app.redis_client.ttl(cold.cache_key)
            CONSOLE.print(f"Stored key: {cold.cache_key} (TTL: {ttl}s)")

        for iteration in range(args.iterations):
            exact = app.cache.answer(base_request)
            app.cache.record_feedback(exact, correct=True)
            results.append((f"exact hit {iteration + 1}", exact))

        semantic_request = CacheRequest(
            args.semantic_prompt,
            partition,
            ttl_seconds=args.ttl,
            invalidation_tags=(tag,),
        )
        for iteration in range(args.iterations):
            semantic = app.cache.answer(semantic_request)
            if semantic.outcome in {CacheOutcome.EXACT_HIT, CacheOutcome.SEMANTIC_HIT}:
                app.cache.record_feedback(semantic, correct=True)
            results.append((f"semantic hit {iteration + 1}", semantic))

        forced = app.cache.answer(
            CacheRequest(args.semantic_prompt, partition, force_miss=True)
        )
        results.append(("forced miss", forced))

        protected = app.cache.answer(
            CacheRequest(
                args.false_hit_prompt,
                partition,
                ttl_seconds=args.ttl,
                invalidation_tags=(tag,),
            )
        )
        results.append(("false-hit guard", protected))

        bypassed = app.cache.answer(
            CacheRequest(args.volatile_prompt, partition, force_miss=False)
        )
        results.append(("volatile bypass", bypassed))

        CONSOLE.print(_result_table(results))
        CONSOLE.print(_metrics_table(app.cache))
        invalidated = app.store.invalidate_tag(partition, tag)
        CONSOLE.print(f"Invalidation tag '{tag}' removed {invalidated} cache entries.")
    finally:
        app.close()


def _load_calibration_pairs(path: str) -> list[CalibrationPair]:
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    records = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Calibration dataset must contain a 'pairs' list")
    return [
        CalibrationPair(
            query=str(record["query"]),
            candidate=str(record["candidate"]),
            should_hit=bool(record["should_hit"]),
            name=str(record.get("name", "")),
        )
        for record in records
    ]


def _calibration_table(report: CalibrationReport) -> Table:
    table = Table(title=f"Threshold calibration ({report.pairs} labeled pairs)")
    table.add_column("Distance", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Hit rate", justify="right")
    table.add_column("False-hit rate", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Selected")
    for score in report.scores:
        selected = score.distance_threshold == report.recommended.distance_threshold
        table.add_row(
            f"{score.distance_threshold:.3f}",
            f"{score.similarity_threshold:.3f}",
            f"{score.hit_rate:.1%}",
            f"{score.false_hit_rate:.1%}",
            f"{score.precision:.1%}",
            f"{score.recall:.1%}",
            f"{score.f1:.3f}",
            "✓" if selected else "",
        )
    return table


def run_calibration(args: argparse.Namespace, settings: PortfolioSettings) -> None:
    if not settings.openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    backend = OpenAIBackend(
        api_key=settings.openai_api_key,
        embedding_model=settings.openai_embedding_model,
    )
    try:
        pairs = _load_calibration_pairs(args.dataset)
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
        report = calibrate_thresholds(
            pairs,
            backend,
            thresholds=thresholds,
            max_false_hit_rate=args.max_false_hit_rate,
        )
        CONSOLE.print(_calibration_table(report))
        CONSOLE.print(
            "Recommended configuration: "
            f"CACHE_DISTANCE_THRESHOLD={report.recommended.distance_threshold:.3f} "
            f"(similarity ≥ {report.recommended.similarity_threshold:.3f}); "
            f"guard rejected {report.guard_rejections} labeled pairs before thresholding."
        )
    finally:
        backend.close()


def run_invalidation(args: argparse.Namespace, settings: PortfolioSettings) -> None:
    redis_client = create_redis_client(settings.redis_url)
    partition = _partition(args, settings)
    try:
        redis_client.ping()
        store = RedisSemanticCacheStore(settings, redis_client)
        if args.scope == "prompt":
            if not args.prompt:
                raise ValueError("--prompt is required for prompt invalidation")
            deleted = store.invalidate_prompt(partition, args.prompt)
        elif args.scope == "tag":
            if not args.tag:
                raise ValueError("--tag is required for tag invalidation")
            deleted = store.invalidate_tag(partition, args.tag)
        elif args.scope == "partition":
            deleted = store.invalidate_partition(partition)
        else:
            deleted = store.invalidate_task(partition.tenant, partition.task)
        CONSOLE.print(f"Invalidated {deleted} cache entr{'y' if deleted == 1 else 'ies'}.")
    finally:
        redis_client.close()


def _add_partition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument(
        "--permissions",
        default=",".join(DEFAULT_PERMISSIONS),
        help="Comma-separated permission scope included in the cache partition",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Run one request through the cache-aside flow")
    ask.add_argument("prompt")
    ask.add_argument("--ttl", type=int, default=None)
    ask.add_argument("--tag", action="append", default=[])
    ask.add_argument("--force-miss", action="store_true")
    _add_partition_arguments(ask)

    benchmark = commands.add_parser(
        "benchmark",
        help="Compare cold, exact, semantic, forced-miss, protected, and bypass paths",
    )
    benchmark.add_argument("--base-prompt", default=DEFAULT_BASE_PROMPT)
    benchmark.add_argument("--semantic-prompt", default=DEFAULT_SEMANTIC_PROMPT)
    benchmark.add_argument("--false-hit-prompt", default=DEFAULT_FALSE_HIT_PROMPT)
    benchmark.add_argument("--volatile-prompt", default=DEFAULT_VOLATILE_PROMPT)
    benchmark.add_argument("--iterations", type=int, default=3)
    benchmark.add_argument("--ttl", type=int, default=None)
    benchmark.add_argument("--tag", default=DEFAULT_INVALIDATION_TAG)
    _add_partition_arguments(benchmark)

    calibrate = commands.add_parser(
        "calibrate",
        help="Evaluate distance thresholds against labeled semantic pairs",
    )
    calibrate.add_argument("--dataset", default=DEFAULT_CALIBRATION_DATASET)
    calibrate.add_argument(
        "--thresholds",
        default="0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.35,0.40",
    )
    calibrate.add_argument("--max-false-hit-rate", type=float, default=0.01)

    invalidate = commands.add_parser("invalidate", help="Invalidate scoped cache entries")
    invalidate.add_argument(
        "--scope",
        choices=("prompt", "tag", "partition", "task"),
        required=True,
    )
    invalidate.add_argument("--prompt")
    invalidate.add_argument("--tag")
    _add_partition_arguments(invalidate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.command in {"ask", "benchmark", "invalidate"}:
        CONSOLE.print(f"Redis: {redact_redis_url(settings.redis_url)}")
    if args.command == "ask":
        run_ask(args, settings)
    elif args.command == "benchmark":
        if args.iterations < 1:
            raise ValueError("--iterations must be at least 1")
        run_benchmark(args, settings)
    elif args.command == "calibrate":
        run_calibration(args, settings)
    else:
        run_invalidation(args, settings)


if __name__ == "__main__":
    main()
