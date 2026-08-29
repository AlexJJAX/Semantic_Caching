# Redis AI Workbench

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Redis](https://img.shields.io/badge/Redis-8%20Search%20%7C%20JSON-DC382D)
![Model](https://img.shields.io/badge/live%20model-gpt--5.6--luna-412991)
![Transport](https://img.shields.io/badge/live%20events-SSE-00695C)
![Frontend](https://img.shields.io/badge/frontend-vanilla%20HTML%20%7C%20CSS%20%7C%20JS-E65100)
![License](https://img.shields.io/badge/license-MIT-455A64)

An interactive local system for observing how Redis participates in AI request processing. Four
focused demonstrations—semantic caching, STM/LTM, authorization-aware RAG, and retrieval
evaluation—share one visible lifecycle:

```text
Prompt → cache decision → retrieval → model → STM/LTM write → metrics
```

The browser streams each decision as it happens, then renders measured metrics, sanitized source
cards, comparison tables, charts, and Redis key/index metadata. Redis operations are always real.
The model layer can use live OpenAI calls or an explicitly selected local simulator.

This is a generic, demonstrational working primitive intended to showcase Redis semantic caching,
STM, LTM, authorization-aware RAG, and retrieval evaluation through one interface. It is not
intended or suitable for production use: real Redis and model calls make the paths observable, but
the fixed scenarios, synthetic identities, local server, metrics, controls, and lifecycle do not
form a production application.

## Experience at a Glance

| Demonstration | Decision being made | Redis capability | Model behavior | Persisted evidence |
| ------------- | ------------------- | ---------------- | -------------- | ------------------ |
| [Semantic cache](http://127.0.0.1:8123/?demo=cache) | Reuse an exact or meaning-equivalent answer, or generate a new one | Direct RedisJSON lookup, partition-filtered vector range search, TTL, invalidation | Skipped on safe hits; called on misses and bypasses | Expiring cache entry on eligible misses |
| [STM / LTM](http://127.0.0.1:8123/?demo=memory) | Use recent thread state and retain only explicit durable preferences | RedisJSON, expiring STM, persistent provenance-rich LTM | Receives memory inside an untrusted-context boundary | 15-minute STM; LTM without Redis expiry |
| [RBAC RAG](http://127.0.0.1:8123/?demo=rbac) | Admit only evidence authorized for the requester's role | RedisJSON Search, TAG pre-filter, vector range threshold | Called only when authorized evidence is available | Reusable four-document RAG corpus |
| [Retrieval evaluation](http://127.0.0.1:8123/?demo=evaluation) | Compare unfiltered retrieval with role-filtered, thresholded retrieval | Repeatable Redis Search queries and expiring JSON run configuration | Generates from the frozen after-result set | Aggregate run configuration for one hour |

These links assume the server is running on its default local address.

## Architecture Overview

| Component | Responsibility |
| --------- | -------------- |
| [`static/index.html`](static/index.html) | Semantic structure, four tab panels, recorder, results, inspector, and reset dialog |
| [`static/styles.css`](static/styles.css) | Responsive visual system, focus treatments, reduced-motion behavior, and component layouts |
| [`static/app.js`](static/app.js) | Tab navigation, form submission, SSE consumption, polling fallback, result rendering, inspection, and reset |
| [`server.py`](server.py) | Local threaded HTTP server, static delivery, JSON API, SSE streaming, request limits, and security headers |
| [`RedisAIWorkbench`](../src/redis_ai_portfolio/workbench.py) | Validates requests, runs the four demonstrations, emits lifecycle events, and shapes sanitized results |
| [`RunStore`](../src/redis_ai_portfolio/workbench.py) | Keeps at most 100 replayable run records in process memory for SSE and polling |
| [`SemanticCache`](../src/redis_ai_portfolio/semantic_cache.py) | Implements exact-first, semantic-second cache-aside behavior with bypass and false-hit controls |
| Redis 8 | Stores JSON state, applies TTLs, indexes vectors and permissions, and supplies inspector metadata |
| OpenAI backend | Provides live 512-dimensional cache embeddings and GPT-5.6 Luna generation when live mode is selected |

The frontend has no compilation or package-install step. The Python server delivers the static
assets and same-origin API from one local process.

## Request Processing and Event Streaming

Submitting a form starts a daemon worker thread and immediately returns a run identifier. The
browser opens an SSE connection for that run and receives ordered lifecycle events. If the event
stream disconnects, the client falls back to polling the run snapshot.

Every demonstration emits the same six stage names, marking irrelevant stages as skipped rather
than hiding them:

| Stage | What it communicates |
| ----- | -------------------- |
| `prompt` | Input acceptance, sanitization, role, partition, or evaluation intent |
| `cache` | Exact lookup, miss, hit, bypass, forced miss, or an explicit no-cache decision |
| `retrieval` | Semantic candidates, memory context, role-filtered evidence, or frozen evaluation results |
| `model` | Generation start, token usage, avoided call, safe abstention, or simulator activity |
| `memory` | Cache write, STM/LTM persistence, evaluation record, or an explicit no-write decision |
| `metrics` | Terminal outcome, latency, authorization count, retention measurement, or evaluation result |

Run events and completed result cards are held in `RunStore`; they are not persisted to Redis.
The store is thread-safe, bounded to 100 runs, and removes the oldest in-memory record when full.
A server restart clears this event history without affecting Redis-backed demo state.

## Demonstration Details

### 1. Semantic Cache

The cache follows an exact-first cache-aside flow:

```text
prompt
  ↓
policy check
  ↓
exact RedisJSON key
  ├─ hit  → return cached answer
  └─ miss → embed prompt → partition-filtered vector range search
                              ├─ guarded hit → return cached answer
                              └─ miss        → generate → store with TTL
```

Cache entries are partitioned by tenant, task, model, prompt version, and permission scope. The
workbench uses:

```text
tenant          workbench
task            semantic-cache
model           OPENAI_MODEL
prompt version  workbench-v2-live or workbench-v2-demo
permission      workbench-reviewer
```

Separate live/demo prompt versions prevent simulated answers from being reused by live requests.
Exact keys normalize case, Unicode, and inconsequential whitespace. Semantic retrieval uses a
fixed cosine-distance threshold of `0.28`, equivalent to similarity `0.72`, and applies the full
partition filter inside Redis before accepting candidates.

A false-hit guard rejects candidates whose numbers, identifiers, quoted literals, polarity,
question intent, or action intent conflict with the new prompt. Sensitive identifiers, secret-like
content, volatile questions, and side-effecting requests bypass cache reads and persistence.

The request-path selector demonstrates:

| Scenario | Behavior |
| -------- | -------- |
| Automatic | Uses the current cache state naturally |
| Cold miss | Invalidates the exact prompt before running the request |
| Prime exact hit | Writes the same prompt first, then demonstrates direct reuse |
| Prime semantic hit | Seeds the canonical cache prompt before running the supplied paraphrase |
| Forced miss | Skips exact and semantic reads, then exercises generation and eligible storage |

The workbench TTL is the smaller of `CACHE_TTL_SECONDS` and 3,600 seconds. Process-level cache
metrics track hit rate, outcome latency, token use, and estimated provider cost or savings.

### 2. Short- and Long-Term Memory

The memory demonstration uses two RedisJSON retention horizons for the fixed demonstration user
`demo-user`:

| Property | STM | LTM |
| -------- | --- | --- |
| Scope | User plus sanitized `thread_id` | User plus content digest |
| Content | Bounded recent user/assistant turns | Explicit preference statement |
| Retrieval | Direct thread-key read | Prefix scan followed by application-side token overlap |
| Expiry | Fixed 15-minute Redis TTL | No Redis TTL |
| Provenance | Thread and turn timestamps | Source, run ID, thread ID, and creation time |
| Deletion | Automatic expiry | Confirmed workbench reset |

LTM is written only when the prompt explicitly contains an intent such as `remember`,
`I prefer`, or `my preference is`. Sensitive or volatile prompts can still receive a response but
are not written to STM or LTM.

This tab is a focused RedisJSON retention demonstration. It does not instantiate the LangGraph
checkpointer or vector-memory index from the separate
[`agentic/Memory`](../agentic/Memory/README.md) example. Long-term recall here is intentionally
small and inspectable rather than a production semantic-memory retrieval strategy.

### 3. Authorization-Aware RAG

The RAG tab lazily creates a four-document RedisJSON corpus with a 512-dimensional `FLOAT32`
`FLAT` cosine index. Documents retain `document_id`, title, content, source, page, and
`allowed_roles` metadata.

The requester's `finance`, `sales`, or `people` role becomes a Redis TAG pre-filter inside the
vector query:

```text
role TAG filter
      ↓
authorized candidate set
      ↓
VECTOR_RANGE distance ≤ 0.72
      ↓
at most three passages
      ↓
grounded generation or safe abstention
```

The threshold corresponds to similarity of at least `0.28`. Retrieved content is wrapped as
authorized but untrusted evidence, so document text cannot change the requester's role or override
model instructions. Source names and page numbers are preserved for `[source, p. N]` citations.
If no authorized evidence clears the threshold, the workbench abstains without making a model
call.

RBAC query and document embeddings always use the deterministic local semantic sketch—even in
live mode—so authorization and retrieval comparisons remain repeatable. Live mode affects the
grounded answer generation step.

### 4. Retrieval Evaluation

The evaluation tab runs three fixed cases against two configurations:

| Configuration | Role filter | Distance threshold | Maximum documents |
| ------------- | ----------- | ------------------ | ----------------- |
| Before | Disabled | `1.25` | 4 |
| After | Redis TAG pre-filter | `0.72` | 2 |

Each configuration retrieves exactly once per question. The three after-result lists are retained
in memory and passed unchanged to both expected-document scoring and answer generation. No second
retrieval can introduce context drift.

The tab reports hit rate, context precision, and retrieval p95. It then writes the complete run
configuration—dataset name, backend mode, model, token counts, embedding dimensions, threshold,
case count, retrieval policy, and timestamp—to RedisJSON with a one-hour TTL. Raw evaluation
prompts, retrieved passages, and generated answers are not included in that record.

This compact comparison is separate from the repository's broader
[`evaluation`](../evaluation/README.md) workflow, which covers test-set generation, Batch API
collection, reproducible manifests, and Ragas metrics.

## Model Backends

Live mode is the default. Backend selection is explicit and startup fails if live mode lacks an
API key; a live API error never silently falls back to simulated output.

| Capability | `live` | `demo` |
| ---------- | ------ | ------ |
| Redis operations | Real local Redis | Real local Redis |
| Semantic-cache embeddings | OpenAI Embeddings API, 512 dimensions | Deterministic local 512-dimensional sketch |
| RBAC/evaluation embeddings | Deterministic local sketch | Deterministic local sketch |
| Cache-miss generation | Configured OpenAI model | Repeatable local response |
| Memory generation | Configured OpenAI model | Application-provided fallback response |
| RAG/evaluation generation | Configured OpenAI model when evidence exists | Grounded application-provided fallback |
| Token accounting | Provider-reported usage | Local estimate |
| External network required | Yes | No |

Live generation uses the OpenAI Responses API with low reasoning effort, a 30-second client
timeout, at most two client retries, bounded output tokens, and `store=False`. API credentials are
loaded only by the Python server and are never returned to the browser.

## Redis State and Retention

With the default namespace, the workbench owns or uses these names:

| Redis name | Type | Retention |
| ---------- | ---- | --------- |
| `portfolio:cache:workbench:semantic-cache:{digest}` | JSON cache entry | Up to 3,600 seconds |
| `portfolio:workbench:stm:demo-user:{thread}` | JSON short-term memory | 15 minutes |
| `portfolio:workbench:ltm:demo-user:{digest}` | JSON long-term memory | Persistent until reset |
| `portfolio:workbench:rbac:document:{id}` | JSON RAG document | Persistent until reset |
| `portfolio:workbench:evaluation:run:{run_id}` | JSON run configuration | 60 minutes |
| `portfolio:idx:semantic-cache` | Search index over cache entries | Retained; empty workbench entries expire or reset |
| `portfolio:idx:workbench-rbac` | Search index over RAG documents | Persistent until reset |

The Redis inspector scans only `portfolio:workbench:*` and `portfolio:cache:workbench:*` keys,
caps its key listing at 60, and returns only key names, Redis types, TTLs, and memory sizes. It also
lists document counts for indexes under the configured namespace. Stored prompts, answers,
permissions, document text, and embeddings are never returned by the inspector API.

Confirmed reset uses prefix-scoped `SCAN` plus `UNLINK`, then drops only the workbench RBAC index.
It deletes workbench cache, memory, evaluation, and RAG documents without flushing the database.
The shared semantic-cache index remains available after its workbench entries have been removed.

These retention choices exist to contrast expiring, durable, and explicitly deleted Redis state
inside a demonstration. They are not production retention recommendations and do not implement
consent, legal hold, backup, recovery, tenancy, or data-lifecycle governance.

## Run It

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A local Redis 8 instance with Search and JSON commands available.
- An OpenAI API key when using live mode.

From the repository root:

The commands below use `uv` directly. `make setup`, `make doctor`, and `make workbench` are optional
aliases. `make redis-start` is an optional Homebrew-oriented service launcher; use your normal
Redis service manager if the server is already configured elsewhere.

```bash
# Install the locked environment.
uv sync --locked

# Create local configuration and supply an API key for live mode.
cp .env.example .env

# Optional: start Redis with the repository's Homebrew-oriented wrapper.
make redis-start

# Validate the runtime directly.
uv run portfolio-doctor

# Start the Workbench at http://127.0.0.1:8123.
uv run python workbench/server.py --host 127.0.0.1 --port 8123
```

Stop the server with `Ctrl-C`.

For a repeatable offline review:

```bash
WORKBENCH_MODEL_MODE=demo uv run python workbench/server.py --host 127.0.0.1 --port 8123
```

To use another local port:

```bash
uv run python workbench/server.py --host 127.0.0.1 --port 8124
```

The server detects an occupied port and reports the next-port command instead of exposing a socket
traceback. Host and port can also be supplied directly:

```bash
uv run python workbench/server.py --host 127.0.0.1 --port 8123
```

The safe default binds only to `127.0.0.1`. The server has no authentication or TLS termination;
do not expose it to an untrusted network.

## Configuration

The repository-root `.env` is loaded before the server initializes Redis or the model backend.

| Variable | Required | Workbench behavior |
| -------- | -------- | ------------------ |
| `WORKBENCH_MODEL_MODE` | No | `live` by default; accepts only `live` or `demo` |
| `OPENAI_API_KEY` | Live only | Required at startup in live mode; never sent to the browser |
| `OPENAI_MODEL` | No | Defaults to `gpt-5.6-luna` and partitions cache entries by model |
| `OPENAI_EMBEDDING_MODEL` | No | Defaults to `text-embedding-3-small`; used for live cache embeddings |
| `REDIS_URL` | No | Takes precedence over individual Redis connection fields |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | No | Default to `localhost`, `6379`, and `0` |
| `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL` | No | Optional Redis authentication and TLS settings |
| `REDIS_NAMESPACE` | No | Defaults to `portfolio` and scopes workbench keys and indexes |
| `CACHE_TTL_SECONDS` | No | Cache-entry TTL, capped at one hour by the workbench |
| `CACHE_LLM_INPUT_COST_PER_MILLION` | No | Input-token rate used for cost estimates |
| `CACHE_LLM_OUTPUT_COST_PER_MILLION` | No | Output-token rate used for cost estimates |
| `CACHE_EMBEDDING_COST_PER_MILLION` | No | Embedding-token rate used for cost estimates |

The workbench intentionally fixes its semantic-cache distance threshold at `0.28`, STM TTL at 15
minutes, RBAC distance threshold at `0.72`, and embedding dimensions at 512. Consequently,
`CACHE_DISTANCE_THRESHOLD`, `STM_TTL_MINUTES`, and `STM_REFRESH_TTL_ON_READ` configure the
standalone examples but do not change this interface's comparison scenarios.

## HTTP API

| Method | Route | Success | Purpose |
| ------ | ----- | ------- | ------- |
| `GET` | `/` and static assets | `200` | Serve the browser application |
| `GET` | `/ready` | `200` or `503` | Return Redis-aware readiness and sanitized backend configuration |
| `GET` | `/api/status` | `200` | Return Redis, model-mode, model-name, and embedding-mode status |
| `POST` | `/api/runs` | `202` | Validate a request and start a background run |
| `GET` | `/api/runs/{run_id}` | `200` | Retrieve the current or terminal in-memory run snapshot |
| `GET` | `/api/runs/{run_id}/events` | `200` | Replay and stream ordered recorder events with SSE |
| `GET` | `/api/redis` | `200` | Return sanitized key and index metadata |
| `DELETE` | `/api/workbench` | `200` | Perform a confirmed namespace-scoped reset |

Example run request:

```bash
curl -sS http://127.0.0.1:8123/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"demo":"cache","scenario":"auto","prompt":"How does Redis caching reduce latency?"}'
```

The returned `run_id` can be used with the snapshot or SSE routes. Request bodies must be JSON
objects between 1 byte and 16 KiB. Prompts are limited to 2,000 characters; roles, scenarios, and
thread identifiers are validated and normalized before execution.

## Security and Privacy Boundaries

- The server binds to localhost unless explicitly overridden.
- Static paths are resolved beneath the owned static directory to prevent traversal.
- Every response receives `no-store`, MIME-sniffing protection, no-referrer behavior, a restrictive
  same-origin Content Security Policy, denied camera/microphone/geolocation permissions, and
  frame-embedding protection.
- Request bodies and query strings are not written to server access logs.
- The frontend inserts dynamic values with DOM text nodes rather than HTML interpolation.
- Sensitive prompts are withheld from lifecycle events and bypass workbench persistence.
- OpenAI requests use `store=False`; provider credentials remain server-side.
- The inspector exposes metadata rather than stored values.
- Reset requires the exact confirmation payload `{"confirm":"reset"}` and remains namespace-scoped.

These are defense-in-depth controls that make a local Redis primitive safer to demonstrate; they
are not a production security posture. They do not provide user
authentication, authorization for the HTTP API, rate limiting, TLS termination, or a complete
data-loss-prevention system.

## Interface Accessibility

The interface includes:

- A keyboard-accessible skip link.
- Tab semantics with Arrow Left/Right, Home, and End navigation.
- Visible focus indicators for controls, links, and scrollable regions.
- Status and event-stream live regions.
- Labeled forms, tables, source lists, charts, and modal reset controls.
- Responsive layouts for desktop, tablet, and narrow mobile widths.
- Horizontally scrollable data tables.
- Reduced animation when `prefers-reduced-motion` is enabled.

These implementation features improve keyboard, screen-reader, and motion accessibility, but they
are not a formal WCAG conformance claim. Re-run keyboard, screen-reader, zoom, contrast, and
responsive checks after material UI changes.

## Project Structure

```text
workbench/
├── server.py              # Local HTTP, JSON, and SSE server
├── static/
│   ├── index.html         # Semantic application structure
│   ├── styles.css         # Responsive design and interaction states
│   ├── app.js             # Browser state, streams, rendering, and reset
│   └── favicon.svg        # Local icon asset
└── README.md              # Workbench architecture and operating guide

src/redis_ai_portfolio/
├── workbench.py           # Demonstration engine and live/demo backends
├── semantic_cache.py      # Cache-aside storage, policies, guards, and metrics
├── config.py              # Typed environment configuration
└── redis.py               # Bounded shared Redis client

tests/test_workbench.py    # Unit and local-Redis behavior coverage
```

## Verification

Run the complete quality gate directly from the repository root:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.

The workbench tests cover replayable run events, deterministic local embeddings, live OpenAI
request construction and usage accounting, explicit live-mode configuration failure, browser form
value capture, all four real-Redis demonstration paths, sanitized inspection, and sensitive-prompt
non-retention. Redis integration tests skip with a stated reason when Search or JSON is unavailable.

The bounded OpenAI + Redis cache flow is also available as an explicit live smoke test. See the
repository [test strategy](../TESTING.md) for the command, cost ceiling, and service gates.

For a manual interface review, exercise all four tabs in live and demo modes, navigate tabs without
a mouse, interrupt SSE to confirm polling recovery, resize through the responsive breakpoints,
inspect the browser console, and confirm that reset removes only workbench-owned state.

## Scope and Limitations

- The workbench combines generic working primitives to explain Redis AI capabilities. It is not
  intended or suitable for production deployment or use as a hosted multi-user application.
- Roles and `demo-user` illustrate retrieval partitioning; they are not authenticated identities.
- Run records are process-local and disappear when the server stops.
- RBAC and evaluation use four sanitized documents and deterministic local embeddings, so their
  rankings are explanatory rather than benchmark results.
- The memory tab uses direct RedisJSON reads and token-overlap LTM selection, not the full agentic
  memory implementation.
- Latency, token, cost, hit-rate, and precision values describe the current local run and configured
  estimates; they are not production service-level objectives.
- Live mode requires network access, may incur provider charges, and exposes provider failures
  instead of substituting simulated output.

## License

This project is available under the repository's [MIT License](../LICENSE).
