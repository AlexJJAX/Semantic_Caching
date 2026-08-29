# Search and evaluation

## Vector-search demonstrations

All three demos use `resources/movies.json`, local HuggingFace embeddings, Redis hashes, and RediSearch, but they are intentionally different examples.

| Script | Index and vectors | Query style | Cleanup |
|---|---|---|---|
| [`Redispy.py`](../vector_search/1_redispy/Redispy.py) | One 384-d `all-MiniLM-L6-v2` vector; HNSW; float32 | Raw Dialect 2 KNN, hybrid filters, text/BM25, radius, aggregation, weighted branches | Drops only its namespaced index and documents |
| [`Redisvl.py`](../vector_search/2_redisvl/Redisvl.py) | One 384-d vector; FLAT; float32; Redis-backed embedding cache | RedisVL `VectorQuery`, `RangeQuery`, `TextQuery`, and hybrid query; alpha `0.7` | Deletes index |
| [`Multivector_search.py`](../vector_search/3_multivector_search/Multivector_search.py) | 384-d general, 768-d movie-specific, 384-d genre-aware vectors; HNSW | Weighted multi-vector blend: `0.3 / 0.5 / 0.2` | Deletes index |

The multi-vector script's ingestion is sequential even though ranking combines multiple fields in Redis. One vector uses float64 while the others use float32. All fields use cosine distance because the weighted score normalization depends on comparable metrics.

Run a demo from the repository root:

```bash
uv run python vector_search/1_redispy/Redispy.py
uv run python vector_search/2_redisvl/Redisvl.py
uv run python vector_search/3_multivector_search/Multivector_search.py
```

All demos use the shared `REDIS_NAMESPACE` (default `portfolio`) to derive their index and key prefixes. The native demo drops only its own index and documents, so it no longer clears the entire Redis database; an isolated development Redis instance is still recommended.

## Ragas evaluation pipeline

The evaluation target is a Redis-backed RAG chain over [`resources/nke-10k-2023.pdf`](../resources/nke-10k-2023.pdf).

### Generate a testset

[`evaluation/generate_testset.py`](../evaluation/generate_testset.py) splits the PDF with chunk size `512` and overlap `100`, submits one `/v1/embeddings` request per chunk through OpenAI Batch, and records a versioned, reproducible manifest containing the source hash, exact chunks, metadata, models, endpoint, and generation settings. After the asynchronous batch succeeds, collection reconstructs a text-to-vector cache, falls back to live embeddings for missing entries, generates synthetic questions, and writes CSV. `--partial` performs one status check and collects available output immediately.

```bash
uv run python evaluation/generate_testset.py --submit \
  --pdf resources/nke-10k-2023.pdf --state evaluation/.batch_state.json
uv run python evaluation/generate_testset.py --collect \
  --state evaluation/.batch_state.json --out evaluation/new_testset.csv --size 10
```

### Score the RAG chain

[`evaluation/06_ragas_evaluation.py`](../evaluation/06_ragas_evaluation.py) recreates the evaluation index under the configured namespace, runs an LCEL retriever/generator chain that retrieves once per question, and passes the identical raw documents to both generation and Ragas scoring. Retrieval applies a relevance threshold; prompts delimit passages as untrusted evidence and include numbered source/page citations. It evaluates:

- Faithfulness
- Answer relevancy
- Context recall
- Context precision

```bash
uv run python evaluation/06_ragas_evaluation.py
```

Metrics are written to `resources/metrics_512_100.csv`; the namespaced temporary Redis index/documents and vector-store pool are cleaned in teardown, including failure paths. The Ragas implementation imports compatibility-sensitive private metric modules, so dependency upgrades require a verification run.

## What is not currently measured

There is no common recall/latency/quality harness for the three vector demos and no project-owned pytest suite. Printed rankings and screenshots are demonstrations; Ragas scores apply only to the evaluation RAG chain and are API/LLM-dependent.
