# Native Redis Vector Search with redis-py

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Redis](https://img.shields.io/badge/Redis-8%20Search-DC382D)
![Client](https://img.shields.io/badge/client-redis--py%207.3%2B-B71C1C)
![Embeddings](https://img.shields.io/badge/embeddings-local%20MiniLM-00796B)
![Index](https://img.shields.io/badge/vector%20index-FLAT%20cosine-5D4037)
![License](https://img.shields.io/badge/license-MIT-455A64)

A low-level Redis Search demonstration that turns movie descriptions into local
Sentence Transformer embeddings, stores the resulting binary vectors in Redis Hashes, and
queries them through the native `redis-py` Search API.

The example deliberately avoids orchestration and vector-database wrappers. Index construction,
`FLOAT32` encoding, Dialect 2 query expressions, parameter binding, KNN ranking, metadata
pre-filters, vector ranges, BM25 scoring, aggregation, and scoped teardown remain visible in one
runnable script.

This is a generic, demonstrational working primitive intended to showcase native Redis vector,
metadata, text, and aggregation queries. It is not intended or suitable for production use: the
small dataset, local model, fixed schema, illustrative thresholds, and destructive demo lifecycle
favor visibility of Redis mechanics over durability, scale, relevance guarantees, or operations.

## Architecture Overview

| Component                                                    | Responsibility                                                                                           |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `resources/movies.json`                                      | Supplies 20 demonstration movies with ID, title, genre, rating, and description                          |
| `SentenceTransformer`                                        | Runs `sentence-transformers/all-MiniLM-L6-v2` locally and produces 384-dimensional embeddings            |
| `embed_text()`                                               | Encodes text with `precision="float32"` and returns the raw vector bytes Redis expects                   |
| `load_docs()`                                                | Pipelines all enriched movie records into namespaced Redis Hashes without a transaction                  |
| Redis Search schema                                          | Indexes one vector, one numeric field, one exact-match tag, and two full-text fields                     |
| `Query` examples                                             | Exercise KNN, metadata-filtered KNN, vector ranges, text matching, fuzzy matching, and weighted branches |
| `AggregateRequest`                                           | Groups indexed movies by genre and calculates their average rating                                       |
| [`redispy_sequence_diagram.md`](redispy_sequence_diagram.md) | Traces model initialization, ingestion, index replacement, every query family, and failure-safe cleanup  |

All executable behavior lives in [`Redispy.py`](./Redispy.py). Importing it loads shared
configuration and definitions but does not connect to Redis, initialize the embedding model, or
mutate data.

### End-to-End Flow Sequence Diagram

For the complete interaction among the runner, local model, Redis client, Hash storage, and
Search engine, see the
[native Redis vector-search sequence diagram](redispy_sequence_diagram.md).

## What it demonstrates

- Direct `redis-py` construction of a Redis Search index over Hash documents.
- Exact `FLAT` vector search with 384-dimensional `FLOAT32` embeddings and cosine distance.
- Binary query-vector binding through `query_params` instead of string interpolation.
- Top-K semantic retrieval with a computed distance alias.
- Filter-then-vector search using TAG, NUMERIC, and TEXT predicates.
- `VECTOR_RANGE` queries with explicit cosine-distance limits.
- Full-text prefix, fuzzy, Boolean, BM25, and weighted-query syntax.
- `FT.AGGREGATE` grouping and average reduction.
- Batched Hash ingestion through a non-transactional Redis pipeline.
- Lowercase, colon-separated index and key names from the shared namespace configuration.
- Repeatable execution by replacing and deleting only this example's index and documents.
- Bounded Redis connections, limited retries, and deterministic pool closure.

## Why raw redis-py

Higher-level libraries can infer schemas, serialize vectors, and build query objects. This
example keeps those responsibilities explicit so each Python object can be related directly to
the Redis command it emits:

| redis-py surface                                   | Redis operation       | Purpose                                       |
| -------------------------------------------------- | --------------------- | --------------------------------------------- |
| `client.ft(name).create_index()`                   | `FT.CREATE`           | Create the Hash index and field schema        |
| `pipeline.hset()`                                  | `HSET`                | Write source fields and raw embedding bytes   |
| `client.ft(name).search(Query(...))`               | `FT.SEARCH`           | Run text, metadata, KNN, and range queries    |
| `client.ft(name).aggregate(AggregateRequest(...))` | `FT.AGGREGATE`        | Group and reduce indexed records              |
| `dropindex(delete_documents=True)`                 | `FT.DROPINDEX ... DD` | Remove the owned index and matching documents |

This makes the example suitable for learning Redis Search query mechanics or debugging the
wire-level behavior hidden by a vector-store abstraction.

## Index and document model

| Setting                 | Value                                                                     |
| ----------------------- | ------------------------------------------------------------------------- |
| Search index            | `{REDIS_NAMESPACE}:idx:movies:redispy`                                    |
| Hash key prefix         | `{REDIS_NAMESPACE}:movie:redispy:`                                        |
| Default concrete prefix | `portfolio:movie:redispy:`                                                |
| Dataset size            | 20 movies                                                                 |
| Indexed genres          | `action`, `comedy`                                                        |
| Embedding model         | `sentence-transformers/all-MiniLM-L6-v2`                                  |
| Vector dimensions       | 384                                                                       |
| Vector representation   | Raw `FLOAT32` bytes, 1,536 bytes per embedding                            |
| Vector algorithm        | `FLAT` exact search                                                       |
| Distance metric         | Cosine                                                                    |
| Initial vector capacity | Dataset length at index creation                                          |
| Storage                 | Redis Hash                                                                |
| Retention               | Index and owned Hashes are deleted on ordinary success or handled failure |

The index schema is:

| Field         | Redis type    | Access pattern                                   |
| ------------- | ------------- | ------------------------------------------------ |
| `vector`      | `VECTOR FLAT` | KNN and vector-range retrieval                   |
| `rating`      | `NUMERIC`     | Minimum-rating filters and aggregation           |
| `genre`       | `TAG`         | Exact category filtering and grouping            |
| `title`       | `TEXT`        | Full-text projection and potential search        |
| `description` | `TEXT`        | Term, prefix, fuzzy, BM25, and weighted matching |

The source `id` is stored in each Hash but is not part of the Search schema. Redis document keys
use the dataset's list position (`...:0` through `...:19`), while the source ID remains available
as stored data.

## Binary vector contract

Redis does not receive a Python list for this Hash vector field. Both indexed and query vectors
must use the same byte layout:

```text
all-MiniLM-L6-v2
        ↓
384 numeric values
        ↓
NumPy float32 array
        ↓
.tobytes()
        ↓
1,536-byte Redis Hash field
```

`embed_text()` requests `precision="float32"` and `convert_to_numpy=True`, then calls
`.tobytes()`. Query vectors follow the identical path before being bound to `$vec`:

```python
client.ft(INDEX_NAME).search(
    query,
    query_params={"vec": embedded_query},
)
```

The shared Redis client intentionally leaves `decode_responses=False`, preserving binary vector
payloads. Changing the model, dimensions, numeric type, or byte representation requires a
matching schema change and a complete rebuild of this example's index.

## Query catalogue

The script executes the following examples in order:

| Query                | Redis Search concept                               | Effective behavior                                       |
| -------------------- | -------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| Basic KNN            | `(*)=>[KNN 3 ... AS dist]`                         | Returns the three closest movies to `High tech movies`   |
| Genre-filtered KNN   | `@genre:{action}` before KNN                       | Searches only action movies                              |
| Genre and rating KNN | TAG plus `@rating:[7 inf]`                         | Searches action movies rated at least 7                  |
| Text-filtered KNN    | `@description:(criminal mastermind)`               | Requires both description terms before vector ranking    |
| Prefix-filtered KNN  | `@description:(crim*)`                             | Narrows candidates to matching description prefixes      |
| Fuzzy-filtered KNN   | `@description:%hero%`                              | Allows Levenshtein distance 1 for the text term          |
| Vector range         | `VECTOR_RANGE` with radius `0.8`                   | Returns all movies within the configured cosine distance |
| Boolean range        | Rating at least 9 OR vector distance at most `0.7` | Demonstrates union of structured and semantic conditions |
| BM25                 | `criminal                                          | mastermind`, `BM25STD`                                   | Ranks movies matching either token and returns scores |
| Aggregation          | `GROUPBY @genre`, `AVG @rating`                    | Calculates the average rating per indexed genre          |
| Weighted text        | Action weight 1 OR fuzzy superhero weight 10       | Demonstrates query-clause boosts without vector scoring  |

For KNN queries, `dist` is a Redis-computed cosine-distance field and results are sorted
ascending: lower values are closer. For cosine distance:

```text
cosine similarity = 1 - cosine distance
```

The example's range values are broad teaching defaults, not calibrated relevance thresholds.

## What “hybrid” means here

The KNN examples use **filter-then-vector retrieval**:

```text
TAG / NUMERIC / TEXT predicate
              ↓
eligible Redis documents
              ↓
KNN vector ranking
```

This is hybrid in the common sense of combining structured or lexical constraints with vector
search. It is not Redis 8.4's `FT.HYBRID` command, which runs separate lexical and vector ranking
legs and combines their scores with a fusion strategy such as RRF or LINEAR.

The final weighted query is also not vector search: it combines an exact genre branch and a
fuzzy description branch using text-query weights. Keeping these mechanisms separate makes the
result semantics easier to reason about.

## Lifecycle and cleanup

The demonstration is intentionally ephemeral:

1. It checks whether the namespaced index already exists.
2. If present, it drops that index and its indexed documents.
3. It creates a fresh schema and pipelines the 20 enriched Hash records.
4. It executes the complete query catalogue.
5. On success, it drops the index with `delete_documents=True`.
6. On failure, `main()` attempts the same scoped drop before re-raising the exception.
7. The Redis connection pool closes in `finally` on every path.

No `FLUSHDB`, broad key deletion, or unrelated index mutation occurs. Concurrent runs using the
same `REDIS_NAMESPACE` will still interfere because they intentionally own the same index and
prefix.

This lifecycle is safe for an isolated working demonstration, not a production index-management
pattern; it intentionally omits online migration, aliases, concurrency control, recovery, backup,
and continuous serving.

## Run it

Prerequisites:

- Python 3.13 or later.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- A local Redis 8 instance with Search available.
- The included [`resources/movies.json`](../../resources/movies.json) dataset.
- Network access on first use if the Sentence Transformer model is not already cached locally.

No OpenAI API key is required; every embedding is generated locally.

From the repository root:

The commands below use `uv` directly. `make setup`, `make doctor`, and `make verify` are optional
aliases. `make redis-start` is an optional Homebrew-oriented launcher for the already-installed
Redis server; you may use your normal service manager instead.

```bash
# Install the locked environment.
uv sync --locked

# Create local Redis configuration if required.
cp .env.example .env

# Optional: start Redis with the repository's Homebrew-oriented wrapper.
make redis-start

# Validate the runtime directly.
uv run portfolio-doctor

# Run the native vector-search demonstration.
uv run python vector_search/1_redispy/Redispy.py
```

Run the command from the repository root because the script resolves
`resources/movies.json` relative to the current working directory.

The shared `.env` settings used by this example are:

| Variable                                        | Required | Default / behavior                                       |
| ----------------------------------------------- | -------- | -------------------------------------------------------- |
| `REDIS_URL`                                     | No       | Takes precedence over individual Redis connection fields |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`          | No       | `localhost`, `6379`, `0`                                 |
| `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL` | No       | Optional authentication and TLS settings                 |
| `REDIS_NAMESPACE`                               | No       | `portfolio`                                              |

`OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` do not affect this script.

## Expected output

Exact ordering reflects the local model and included dataset, but the console structure is
deterministic:

```text
Connected to Redis.

Embedded 20 movies.
Index created.
Index loaded with 20 documents.

--- Basic KNN (top 3) ---
  Top 3 results: [...]

--- Hybrid KNN: action genre only ---
  Top ... results: [...]

--- Range query: cosine distance <= 0.8 for 'Family friendly fantasy movies' ---
  Top ... results: [...]

--- BM25: 'Criminal mastermind' (token OR, scorer=BM25STD) ---
  [...]

--- Aggregation: avg rating per genre ---
  [...]

Demo index and keys removed. Done.
```

If an owned index is left behind by an interrupted prior process, the next run reports that it
is dropping and recreating it before loading the dataset.

## Scope and limitations

The successful queries demonstrate Redis command behavior only. They do not qualify this primitive
for production or establish retrieval quality, performance, capacity, or operational safety.

1. The 20-record, two-genre dataset demonstrates query mechanics; it cannot establish retrieval
   quality, scalability, or representative latency.
2. `FLAT` performs exact exhaustive comparison and is appropriate for this tiny corpus. Larger,
   latency-sensitive collections should benchmark HNSW with measured recall and memory usage.
3. The `0.7` and `0.8` range radii are illustrative. Calibrate distance thresholds against
   labeled queries and costly false-positive examples.
4. Metadata predicates are hard-coded. Applications accepting user filters must construct and
   escape TAG/TEXT values safely rather than interpolate arbitrary query syntax.
5. The code relies on NumPy's native `float32` byte order. Cross-platform binary formats should
   specify endianness explicitly when portability beyond common little-endian systems matters.
6. The OR range example can include rating-qualified records that do not have a yielded vector
   distance. Treat its `dist` ordering as a syntax demonstration, not a production ranking
   contract.
7. Prefix, fuzzy, and wide text searches can become expensive on large corpora. Use
   representative `FT.PROFILE` measurements before adopting them in a latency-sensitive path.
8. The script recreates the index on every run and deletes it afterward. It does not demonstrate
   durable serving, incremental ingestion, schema migration, aliases, pagination, or concurrent
   writers.
9. There is no dedicated relevance testset or benchmark. Printed top results are observational
   examples, not correctness assertions.
10. A forced process termination can bypass Python cleanup and leave the owned index behind. The
    next normal run detects and removes that stale index before rebuilding.

## Test it

Validate imports and syntax without starting the embedding model or contacting Redis:

```bash
uv run python -m compileall -q vector_search/1_redispy
```

Run the dedicated integration test against real Redis Search. It substitutes only a fixed
384-dimensional embedding and executes the complete Hash ingestion, KNN, filtered range, BM25,
aggregation, weighted-query, and cleanup path:

```bash
uv run python -m unittest \
  tests.test_vector_search_redis.VectorSearchRedisIntegrationTests.test_redispy_example_executes_real_search_and_aggregation_queries -v
```

Run the repository-wide quality gate directly with the local Redis service available:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src RAG agentic evaluation llm_message_history semantic_cache vector_search workbench
```

`make verify` is the optional convenience alias for these commands.

The live command is the end-to-end integration check. It loads the local model, creates and
queries the Redis index, and verifies scoped teardown by completing without error.

See the repository [test strategy](../../TESTING.md) for why model downloads are excluded from the
real-Redis test while all storage and query boundaries remain real.

## License

This project is available under the repository's [MIT License](../../LICENSE).
