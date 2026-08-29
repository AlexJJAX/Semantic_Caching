# RedisVL Movie Search — End-to-End Sequence

This sequence shows exact embedding-cache reuse, local vector generation, schema-first Redis
ingestion, typed retrieval, aggregate hybrid scoring, and the separate lifecycles of movie
documents and cached embeddings.

```mermaid
sequenceDiagram
    autonumber

    actor Operator
    participant App as Redisvl.py<br/>run_demo()
    participant Data as Pandas + movies.json
    participant Cache as RedisVL EmbeddingsCache<br/>600-second sliding TTL
    participant Model as Local HFTextVectorizer<br/>all-MiniLM-L6-v2
    participant Index as RedisVL SearchIndex<br/>Query + Result Layer
    participant Redis as Redis Hash + Search

    Note over Operator,Redis: Initialize data, cache, and local model

    Operator->>App: Run vector_search/2_redisvl/Redisvl.py
    App->>Redis: Create bounded client and PING
    Redis-->>App: PONG
    App->>Data: Read resources/movies.json
    Data-->>App: 20-row DataFrame
    App->>Cache: Create namespaced exact cache with TTL 600
    App->>Model: Initialize vectorizer with cache and float32 dtype
    Model->>Model: Encode dimension-check text
    Model-->>App: Local 384-dimensional vectorizer ready

    Note over App,Cache: Batch description embedding

    App->>Model: embed_many(20 descriptions, as_buffer=true)
    Model->>Cache: Batch exact lookup by content and model
    Cache->>Redis: Pipeline HGETALL for deterministic cache keys
    Redis-->>Cache: Cached entries and misses
    loop Each cache hit
        Cache->>Redis: Refresh TTL to 600 seconds
        Redis-->>Cache: Expiry updated
    end
    Cache-->>Model: Ordered hit/miss map

    alt Every description is cached
        Model->>Model: Reuse all cached vectors
    else Some descriptions are missing
        Model->>Model: Encode only missing descriptions locally
        Model->>Cache: Batch-store new vectors and TTLs
        Cache->>Redis: Pipeline HSET and EXPIRE
        Redis-->>Cache: Cache writes complete
    end

    Model-->>App: Twenty Redis-ready FLOAT32 buffers
    App->>Data: Add vector column

    Note over App,Redis: Recreate and populate the movie index

    App->>Index: Build Hash IndexSchema
    App->>Index: create(overwrite=true, drop=true)
    Index->>Redis: Inspect existing index

    alt Movie index exists
        Index->>Redis: Drop old index and indexed Hashes
        Redis-->>Index: Old movie state removed
    else Movie index is absent
        Redis-->>Index: Clean namespace
    end

    Index->>Redis: Create TEXT, TAG, NUMERIC,<br/>and FLAT VECTOR schema
    Redis-->>Index: Search index ready
    App->>Index: load(DataFrame records)
    Index->>Index: Generate one ULID key per row
    Index->>Redis: Pipeline Hash writes under movie prefix
    Redis-->>Index: 20 indexed records

    Note over App,Redis: KNN retrieval for High tech and action packed movie

    App->>Model: embed(query)
    Model->>Cache: Exact query-embedding lookup
    Cache->>Redis: HGETALL cache key

    alt Query embedding cache hit
        Redis-->>Cache: Stored vector
        Cache->>Redis: Refresh TTL to 600 seconds
        Cache-->>Model: Cached embedding
    else Query embedding cache miss
        Redis-->>Cache: No entry
        Model->>Model: Encode query locally
        Model->>Cache: Store embedding with TTL 600
        Cache->>Redis: HSET and EXPIRE
    end

    Model-->>App: Query vector
    App->>Index: VectorQuery, top 3
    Index->>Redis: FT.SEARCH KNN with DIALECT 2
    Redis-->>Index: Closest movies and vector_distance
    Index-->>Operator: Print normalized result table

    loop Action TAG; action plus rating;<br/>description terms; prefix; fuzzy term
        App->>Index: VectorQuery with typed filter expression
        Index->>Redis: Pre-filtered FT.SEARCH KNN
        Redis-->>Index: Filter-qualified nearest movies
        Index-->>Operator: Print result table
    end

    Note over App,Redis: Vector ranges for Family friendly fantasy movies

    App->>Model: embed(range query)
    Model->>Cache: Exact lookup, compute on miss, refresh or set TTL
    Cache->>Redis: Read or write cached query vector
    Redis-->>Model: Cached state acknowledged
    Model-->>App: Range query vector
    App->>Index: RangeQuery with distance 0.8
    Index->>Redis: FT.SEARCH VECTOR_RANGE
    Redis-->>Index: Up to ten qualified records
    Index-->>Operator: Print distance-ranked table
    App->>Index: RangeQuery plus rating at least 8
    Index->>Redis: Filtered VECTOR_RANGE
    Redis-->>Index: Distance-and-rating-qualified records
    Index-->>Operator: Print result table

    Note over App,Redis: Lexical and aggregate hybrid ranking

    App->>Index: TextQuery with BM25STD, 20 results
    Index->>Redis: FT.SEARCH over OR-token description query
    Redis-->>Index: BM25-scored records
    Index-->>Operator: Print first four

    App->>Model: embed(the same lexical query)
    Model->>Cache: Exact lookup, compute on miss
    Cache->>Redis: Read or write query embedding
    Redis-->>Model: Embedding cache result
    Model-->>App: Hybrid query vector
    App->>Index: AggregateHybridQuery with alpha 0.7
    Index->>Redis: FT.AGGREGATE with optional text,<br/>KNN 20, APPLY scores, SORTBY hybrid
    Redis-->>Index: text_score, vector_similarity,<br/>hybrid_score, and fields
    Index-->>Operator: Print first four blended results

    Note over App,Redis: Movie state is ephemeral; embedding cache is reusable

    alt Query catalogue completes
        App->>Index: delete()
        Index->>Redis: Drop movie index and Hashes
        Redis-->>Index: Movie state removed
        Index-->>Operator: Print completion
    else Any operation raises
        App->>Redis: Attempt scoped FT.DROPINDEX with document deletion
        Redis-->>App: Removed or index-not-found response
        App-->>Operator: Re-raise original error
    end

    Note over Cache,Redis: Cache keys remain until 600 seconds<br/>after their most recent successful access
    App->>Redis: Close shared connection pool in finally
    App-->>Operator: Process exits
```
