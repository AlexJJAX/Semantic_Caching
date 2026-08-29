# Exact-First Semantic Caching with Redis & GPT-5.6 Luna

![Tests](https://img.shields.io/badge/focused%20tests-14%20passing-2E7D32)
![Integration](https://img.shields.io/badge/Redis%20integration-2%20passing-00796B)
![Model](https://img.shields.io/badge/model-gpt--5.6--luna-412991)
![Redis](https://img.shields.io/badge/Redis-8%20Search%20%7C%20JSON-DC382D)
![License](https://img.shields.io/badge/license-MIT-455A64)

A runnable cache-aside layer that checks a deterministic Redis key before paying for an
embedding, falls back to partition-filtered vector search, and calls GPT-5.6 Luna only when reuse
is unsafe, unavailable, or deliberately disabled.

The example combines TTL, scoped invalidation, permission-aware partitions, task-specific
threshold calibration, conservative false-hit guards, bypass policy, latency percentiles, token
accounting, and configurable cost estimates in one runnable workflow.

This is a generic, demonstrational working primitive intended to showcase Redis semantic caching.
It is not intended or suitable for production use: its policies, partitions, thresholds, TTLs,
guards, pricing assumptions, and metrics demonstrate the cache lifecycle but are not universal
safety rules, service-level objectives, or production controls.

## Architecture Overview

| Component | Responsibility |
| --- | --- |
| `CachePartition` | Canonicalizes tenant, task, model, prompt version, and permission scope into a stable boundary |
| `SemanticCache` | Orchestrates bypass → exact lookup → embedding → semantic lookup → model → cache write |
| `RedisSemanticCacheStore` | Implements direct RedisJSON reads, filtered vector search, TTL writes, and scoped invalidation |
| `CacheBypassPolicy` | Prevents caching for sensitive, volatile, side-effecting, or explicitly uncacheable requests |
| `FalseHitGuard` | Rejects close vectors when correctness-critical prompt facts or intent differ |
| `OpenAIBackend` | Calls the OpenAI Responses and Embeddings APIs with bounded retries and usage accounting |
| `CacheMetrics` | Tracks outcomes, p50/p95 latency, tokens, guard decisions, hit feedback, and cost estimates |
| `calibrate_thresholds()` | Evaluates labeled prompt pairs and recommends the broadest threshold within a false-hit budget |
| `semantic_cache_demo.py` | Exposes `ask`, `benchmark`, `calibrate`, and `invalidate` commands |
| `calibration_cases.json` | Supplies five positive and five deliberately close negative support-FAQ pairs |

The reusable implementation lives in
[`src/redis_ai_portfolio/semantic_cache.py`](../src/redis_ai_portfolio/semantic_cache.py); the
CLI lives in [`semantic_cache_demo.py`](./semantic_cache_demo.py).

## Cache-aside contract

```text
Prompt
  ↓
Cacheability policy
  ├─ bypass ────────────────────────────────→ GPT-5.6 Luna → return only
  ↓ cacheable
Canonical exact key → RedisJSON GET
  ├─ hit ───────────────────────────────────→ cached answer
  ↓ miss
OpenAI embedding → partition-filtered vector range query
  ├─ safe candidate ────────────────────────→ cached answer
  ├─ guarded candidate → inspect next result
  ↓ no acceptable candidate
GPT-5.6 Luna → transactional JSON + TTL write → generated answer
```

An exact hit performs neither embedding nor generation. A semantic hit pays for one query
embedding but avoids generation. A miss reuses that same lookup embedding when writing the new
entry, so the prompt is not embedded twice.

## What it demonstrates

- Exact lookup before any external embedding call.
- Semantic reuse across paraphrases through Redis vector range search.
- Partitioning by tenant, task, model, prompt version, and canonical permissions.
- Readable tenant-first keys with partition-bound prompt digests.
- Redis TTL plus explicit prompt, tag, partition, and tenant/task invalidation.
- Bypass rules for content that should not be retained or reused.
- False-hit protection beyond vector distance alone.
- Task-specific threshold calibration from labeled positive and negative pairs.
- Cold, exact-hit, semantic-hit, forced-miss, guard-rejection, and volatile-bypass comparisons.
- p50/p95 latency by outcome, cache hit rate, feedback false-hit rate, token savings, and cost
  estimates.
- Optional operation-level tracing for exact lookup, embedding, semantic lookup, model calls,
  and cache writes.
- Scoped Redis names, pooled connections, and explicit client cleanup.

## Key Design Decisions

- **Put exact lookup first** — case and whitespace variants resolve to one deterministic key,
  avoiding embedding latency and cost for direct repeats.
- **Partition before similarity** — every semantic query applies all reuse boundaries as Redis
  TAG filters before vector ranking. Cross-tenant or cross-permission candidates never enter the
  guard loop.
- **Keep permissions canonical** — permission names are trimmed, case-folded, deduplicated, and
  sorted before hashing, so equivalent sets share a partition regardless of input order.
- **Use distance plus rules** — vector proximity identifies candidates; the false-hit guard then
  rejects conflicts in numbers, identifiers, quoted literals, polarity, question intent, action
  intent, or extreme prompt length.
- **Try the next candidate** — rejecting the nearest result does not force an immediate miss.
  Up to five distance-qualified candidates are inspected in nearest-first order.
- **Expire twice conceptually** — Redis TTL removes the JSON key, while `expires_at` excludes
  stale documents from vector search even before physical expiry is observed.
- **Keep invalidation scoped** — prompt, tag, and partition deletion use exact boundaries; task
  invalidation scans only `{namespace}:cache:{tenant}:{task}:*` and unlinks in batches.
- **Measure correctness explicitly** — false-hit rate is calculated only from cache hits that
  receive `record_feedback(..., correct=...)`, rather than assuming every hit is correct.
- **Make pricing configurable** — token counts come from provider usage, but dollar values use
  caller-controlled estimates so pricing changes do not require code changes.

## Partition and key model

Every entry uses this format:

```text
{REDIS_NAMESPACE}:cache:{tenant}:{task}:{digest}
```

With the defaults:

```text
portfolio:cache:acme:support-faq:{32-character-sha256-prefix}
```

The digest binds the canonical prompt to a fingerprint containing:

| Dimension | Purpose |
| --- | --- |
| `tenant` | Prevent reuse across organizations |
| `task` | Keep different answer contracts and risk profiles separate |
| `model` | Prevent reuse after changing the generating model |
| `prompt_version` | Invalidate behavior changes without mutating old entries |
| `permissions_scope` | Prevent reuse across different authorization scopes |

Exact normalization applies Unicode NFKC normalization, leading/trailing trim, case folding, and
whitespace collapse. It deliberately does not remove punctuation or rewrite meaning.

Permission scope stores only a 20-character SHA-256 prefix in indexed partition fields. The
canonical raw permission list remains in the RedisJSON document for inspection but is not
indexed.

## Redis data model

| Setting | Default |
| --- | --- |
| Search index | `{REDIS_NAMESPACE}:idx:semantic-cache` |
| JSON key prefix | `{REDIS_NAMESPACE}:cache:` |
| Vector field | 512-dimensional `FLOAT32`, `FLAT`, cosine distance |
| Embedding model | `text-embedding-3-small` with 512 requested dimensions |
| Semantic candidates | 5 |
| Distance threshold | `0.20`, equivalent to similarity ≥ `0.80` |
| TTL | 3,600 seconds, fixed rather than sliding |
| OpenAI client | Maximum 2 retries, 20-second timeout |
| Generation | Responses API, low reasoning effort, maximum 300 output tokens, remote storage disabled |

Indexed fields are limited to partition filters, invalidation tags, expiry, and the vector:

```text
tenant                  TAG
task                    TAG
model                   TAG
prompt_version          TAG
permissions_scope       TAG
partition_fingerprint   TAG
invalidation_tags       TAG[]
expires_at              NUMERIC
embedding               VECTOR
```

The complete JSON document also stores original and normalized prompts, answer, canonical
permissions, timestamps, generation token counts, estimated generation cost, and guard version.
Answers and prompt text are not Search fields. Semantic search returns candidate IDs and
distances, then fetches the corresponding JSON documents in one non-transactional pipeline.

## Bypass policy

Bypass is evaluated before any Redis read or embedding call:

| Rule | Examples detected | Outcome |
| --- | --- | --- |
| Sensitive identifier | Email, payment-card pattern, US SSN pattern | Generate without reading or writing cache |
| Secret-like content | API key, bearer token, password or access-token assignment | Generate without reading or writing cache |
| Volatile query | Live prices, weather, availability, breaking news, scores, traffic | Generate without reading or writing cache |
| Side-effecting request | Cancel, transfer, purchase, update, or delete a nearby owned resource | Generate without reading or writing cache |
| Uncacheable task | Application-configured task denylist | Generate without reading or writing cache |
| Forced miss | Explicit `force_miss=True` or CLI `--force-miss` | Generate without reading or writing cache |

**Bypass is a cache-retention policy, not a transmission block.** The prompt is still sent to
GPT-5.6 Luna. Applications must reject or redact content that must not leave their trust boundary
before calling this cache.

The CLI does not expose the programmatic uncacheable-task denylist; applications can construct
`CacheBypassPolicy(uncacheable_tasks=...)` when embedding the library.

## False-hit protection

For every semantic candidate, the guard compares the query with the cached prompt:

| Signal | Rejection example |
| --- | --- |
| Numeric facts | `2024 invoice` versus `2025 invoice` |
| Mixed identifiers | `order-a12` versus `order-a13` |
| Quoted literals | `plan "Starter"` versus `plan "Enterprise"` |
| Polarity | `Show invoices` versus `Do not show invoices` |
| Question intent | `How many...` versus `When...` |
| Action intent | `Show my account` versus `Delete my account` |
| Prompt length | Longer prompt exceeds a 3:1 token-count ratio |

These regex-based rules are conservative heuristics, not semantic proof. They reduce known
classes of false hits but cannot establish that two prompts have the same constraints or desired
answer.

## Metrics

`CacheMetrics` records process-local measurements for the current application instance:

| Metric | Definition |
| --- | --- |
| Cache hit rate | `(exact hits + semantic hits) / cacheable requests`; bypass and forced miss excluded |
| Feedback false-hit rate | Incorrect evaluated hits / all evaluated hits |
| Guard rejections | Candidate prompts rejected before reuse |
| LLM calls | Miss, bypass, and forced-miss generation calls |
| Generation tokens consumed | Input + output tokens from calls made during the run |
| Generation tokens saved | Original cached generation tokens attributed to cache hits |
| Embedding tokens | Lookup/write embedding usage during the run |
| Actual estimated API cost | Generation and embedding usage multiplied by configured rates |
| Avoided generation cost | Stored generation cost attributed to cache hits |
| Net estimated savings | Avoided generation cost minus cache embedding overhead |
| p50 / p95 latency | Overall and per-outcome percentiles from up to 10,000 latency samples |

When no cache hit has been evaluated, false-hit rate is displayed as `0%`; it means “no observed
feedback,” not “proven correct.” Metrics reset with each CLI process and are not billing records
or a durable observability system.

Default cost assumptions are configurable USD estimates per one million tokens:

| Variable | Default |
| --- | ---: |
| `CACHE_LLM_INPUT_COST_PER_MILLION` | `$0.20` |
| `CACHE_LLM_OUTPUT_COST_PER_MILLION` | `$1.20` |
| `CACHE_EMBEDDING_COST_PER_MILLION` | `$0.02` |

Review these values for the selected provider, model, region, and date. The estimate excludes
Redis infrastructure, networking, long-context multipliers, tools, retries, and other provider
charges.

## Threshold calibration

Semantic thresholds are task-specific. The included support-FAQ dataset contains ten labeled
pairs: five valid paraphrases and five close negative examples.

```bash
uv run python semantic_cache/semantic_cache_demo.py calibrate
```

Calibration embeds each unique prompt once, applies the false-hit guard, and evaluates every
configured cosine-distance threshold. It reports true/false positives and negatives indirectly
through hit rate, false-hit rate, precision, recall, and F1.

The recommendation is the threshold with the highest hit rate, then F1, then broadest distance,
among results that satisfy `--max-false-hit-rate`. If no threshold meets the budget, it selects
the lowest observed false-hit rate with F1 and strictness as tie-breakers.

```bash
uv run python semantic_cache/semantic_cache_demo.py calibrate \
  --dataset semantic_cache/calibration_cases.json \
  --thresholds 0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.35,0.40 \
  --max-false-hit-rate 0.01
```

Redis uses cosine distance, so lower is stricter:

```text
similarity = 1 - distance
```

The repository default of `0.20` is an illustrative support-FAQ baseline. Ten labeled pairs are
enough to demonstrate the method, not to establish a production false-hit guarantee. Calibrate
each task with representative traffic and costly negative examples.

## TTL and invalidation

TTL is assigned atomically with the JSON write and is not refreshed on read. Override it for one
request with `ask --ttl 300`, or set the application default through `CACHE_TTL_SECONDS`.

| Scope | Boundary | Command requirement |
| --- | --- | --- |
| Prompt | One canonical prompt in one complete partition | `--prompt` |
| Tag | Matching content tag in one complete partition | `--tag` |
| Partition | Exact tenant/task/model/prompt-version/permission scope | None |
| Task | All models, prompt versions, and permission scopes for one tenant/task | None |

```bash
# One prompt in the default partition.
uv run python semantic_cache/semantic_cache_demo.py invalidate \
  --scope prompt \
  --prompt "How do I reset my account password?"

# Entries associated with one content release in the default partition.
uv run python semantic_cache/semantic_cache_demo.py invalidate \
  --scope tag \
  --tag help-center-v3

# The complete default partition.
uv run python semantic_cache/semantic_cache_demo.py invalidate --scope partition

# Every variant for the default tenant/task pair.
uv run python semantic_cache/semantic_cache_demo.py invalidate --scope task
```

All operations use `UNLINK` and remain within the configured cache namespace. No invalidation
command flushes Redis or removes unrelated application keys.

## Run it

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A local Redis 8 instance with Search and JSON commands available.
- An OpenAI API key for embeddings and GPT-5.6 Luna generation.

From the repository root:

The commands below use `uv` directly. `make setup`, `make doctor`, and `make verify` are optional
aliases. `make redis-start` is an optional Homebrew-oriented launcher for the already-installed
Redis server; you may use your normal service manager instead.

```bash
# Install the locked dependencies.
uv sync --locked

# Create local configuration, then add OPENAI_API_KEY to .env.
cp .env.example .env

# Optional: start Redis with the repository's Homebrew-oriented wrapper.
make redis-start

# Validate the runtime directly.
uv run portfolio-doctor
```

Run one request:

```bash
uv run python semantic_cache/semantic_cache_demo.py ask \
  "How do I reset my account password?" \
  --tenant acme \
  --task support-faq \
  --prompt-version support-v1 \
  --permissions customer \
  --tag help-center-v3
```

Repeat the command to observe an exact hit. Change only the phrasing to exercise semantic lookup:

```bash
uv run python semantic_cache/semantic_cache_demo.py ask \
  "How can I recover access to my account?" \
  --tenant acme \
  --task support-faq \
  --prompt-version support-v1 \
  --permissions customer
```

Force an uncached model baseline:

```bash
uv run python semantic_cache/semantic_cache_demo.py ask \
  "How can I recover access to my account?" \
  --force-miss
```

Compare every outcome with repeated exact and semantic hits:

```bash
uv run python semantic_cache/semantic_cache_demo.py benchmark --iterations 5
```

The shared `.env` settings used by this example are:

| Variable | Required | Default / behavior |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes for `ask`, `benchmark`, and `calibrate` | Not required for invalidation |
| `OPENAI_MODEL` | No | `gpt-5.6-luna`; included in every cache partition |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` |
| `REDIS_URL` | No | Takes precedence over individual Redis connection fields |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | No | `localhost`, `6379`, `0` |
| `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL` | No | Optional authentication and TLS settings |
| `REDIS_NAMESPACE` | No | `portfolio` |
| `CACHE_DISTANCE_THRESHOLD` | No | `0.20` cosine distance |
| `CACHE_TTL_SECONDS` | No | `3600` |
| `CACHE_*_COST_PER_MILLION` | No | Configurable estimates shown above |

## Expected output

`ask` prints the outcome, latency, similarity when applicable, token/cost accounting, guard or
bypass reason, and answer:

```text
Redis: redis://localhost:6379/0
                         Cache-aside comparison
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Scenario ┃ Outcome      ┃ Latency  ┃ Similarity ┃ LLM tokens ┃ Saved tokens ┃
...
┃ request  ┃ semantic_hit ┃ ... ms   ┃ 0.9...     ┃ 0          ┃ ...          ┃
...
Answer:
...
```

The benchmark starts from an empty partition and reports:

- `cold`: exact miss → semantic miss → model → cache write;
- `exact hit`: direct RedisJSON reuse;
- `semantic hit`: vector-qualified and guard-approved reuse;
- `forced miss`: cache reads and writes deliberately disabled;
- `false-hit guard`: changed-year candidate rejected before generation;
- `volatile bypass`: live-price prompt generated without cache interaction.

It then prints aggregate metrics, displays the stored TTL, and removes entries carrying the
benchmark's invalidation tag.

A recorded local smoke run on 21 August 2026 with `--iterations 1` observed 6,871 ms cold,
0.68 ms exact-hit, 143 ms semantic-hit, and 2,414 ms forced-miss latency. It reported a 50%
cache hit rate, two evaluated hits with no observed false hit, 210 generation tokens saved, and
approximately `$0.000157` net estimated savings. This confirms that the paths executed in that
environment; it is not a portable benchmark or performance target.

## Trust and limitations

Exact and semantic hits prove that the Redis cache paths work; they do not prove that a reused
answer remains correct, current, authorized, or safe. Production adoption requires task-specific
validation and a broader security, observability, reliability, and invalidation design.

1. The bypass and false-hit patterns are regex heuristics. They cannot recognize every secret,
   volatile fact, side effect, constraint change, or semantic mismatch.
2. Bypassed prompts still reach the model provider; only Redis lookup, embedding, and retention
   are skipped.
3. The bypass policy evaluates prompt text, not generated answers. A response containing
   sensitive information can be cached if the request itself was considered cacheable.
4. Permission partitioning prevents reuse across different declared scopes but does not perform
   authentication or determine what content a caller is authorized to request.
5. There is no request coalescing or distributed lock. Concurrent cold requests for the same key
   can all call the model and race to write equivalent entries.
6. Changing a model, prompt version, or permission set creates a new partition but leaves old
   data until TTL expiry or explicit invalidation.
7. Cost savings are counterfactual estimates based on token usage stored with the original
   answer. They exclude infrastructure and do not represent provider invoices.
8. Process-local latency and hit metrics reset on restart and should be exported to a durable
   observability system before production use.
9. Feedback quality depends on the evaluator supplying correct labels. Unevaluated hits do not
   contribute to the false-hit denominator.
10. Cached answers are returned as stored; the example does not revalidate sources, citations,
    policy compliance, or downstream side effects at read time.

## Test it

Fourteen focused tests use fakes and mocks, so they do not contact Redis or OpenAI:

```bash
uv run python -m unittest tests.test_semantic_cache -v
```

Two namespaced integration tests validate exact and semantic hits, TTL, permission isolation,
and every invalidation scope against local Redis:

```bash
uv run python -m unittest tests.test_semantic_cache_redis -v
```

The opt-in live test performs one real OpenAI embedding and GPT-5.6 Luna generation, writes the
answer and TTL through the example's Redis cache store, then proves an identical request is an
exact hit without another API call:

```bash
RUN_LIVE_OPENAI_TESTS=1 \
  uv run python -m unittest tests.test_live_integrations.LiveOpenAIRedisTests -v
```

Run the complete repository quality gate directly with:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.
See the repository [test strategy](../TESTING.md) for cost controls and the coverage matrix.

## License

This project is available under the repository's [MIT License](../LICENSE).
