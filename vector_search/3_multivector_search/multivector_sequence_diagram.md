# RedisVL Multi-Vector Retrieval — End-to-End Sequence

This sequence shows the script's sequential multi-model embedding work, exact Redis cache reuse,
three-field HNSW ingestion, single-request aggregate score fusion, and the separate lifecycles of
movie documents and cached embeddings.

```mermaid
sequenceDiagram
    autonumber

    actor Operator
    participant App as Multivector_search.py<br/>run_demo()
    participant Data as resources/movies.json
    participant Models as Three local HFTextVectorizers
    participant Caches as Three RedisVL EmbeddingsCaches<br/>600-second sliding TTL
    participant Index as RedisVL SearchIndex<br/>Schema + Query Layer
    participant Redis as Redis Hash + Search

    Note over Operator,Redis: Initialize the connection, dataset, caches, and models

    Operator->>App: Run vector_search/3_multivector_search/Multivector_search.py
    App->>Redis: Create bounded client and PING
    Redis-->>App: PONG
    App->>Data: Read movie dataset
    Data-->>App: 20 movie records

    loop General, description-rich, and genre-prefixed views
        App->>Caches: Create isolated exact cache with TTL 600
        App->>Models: Initialize view-specific Sentence Transformer and dtype
        Models->>Models: Encode dimension-check text
        Models-->>App: Model ready with 384, 768, or 384 dimensions
    end

    Note over App,Redis: Generate three representations per movie, sequentially

    loop Each of 20 movies
        loop General description, MPNet description, and genre plus description
            App->>Models: embed(view-specific content, as_buffer=true)
            Models->>Caches: Exact lookup by serialized content and model
            Caches->>Redis: HGETALL deterministic cache key

            alt Embedding cache hit
                Redis-->>Caches: Stored numeric embedding
                Caches->>Redis: Refresh TTL to 600 seconds
                Redis-->>Caches: Expiry updated
                Caches-->>Models: Cached embedding
            else Embedding cache miss or cache read failure
                Redis-->>Caches: Missing entry or error
                Models->>Models: Encode content locally
                Models->>Caches: Store numeric embedding with model identity
                Caches->>Redis: HSET and EXPIRE 600
                Redis-->>Caches: Cache write acknowledged
            end

            Models-->>App: Matching FLOAT64 or FLOAT32 buffer
        end
        App->>App: Attach three vector fields to the movie record
    end

    Note over App,Redis: Replace and populate only the owned movie index

    App->>Index: Build Hash schema with metadata and three HNSW vectors
    App->>Index: Create SearchIndex with load validation
    App->>Index: create(overwrite=true, drop=true)
    Index->>Redis: Inspect namespaced index

    alt Index already exists
        Index->>Redis: Drop old index and indexed movie Hashes
        Redis-->>Index: Previous movie state removed
    else Index does not exist
        Redis-->>Index: Clean namespace
    end

    Index->>Redis: FT.CREATE TEXT, TAG, NUMERIC,<br/>and three cosine HNSW fields
    Redis-->>Index: Index ready
    App->>Index: load(20 enriched records)
    Index->>Index: Validate records and generate ULID keys
    Index->>Redis: Pipeline Hash writes under movie prefix
    Redis-->>Index: 20 indexed movie documents

    Note over App,Redis: Encode one query in all three vector spaces

    loop Each model and matching datatype, sequentially
        App->>Models: embed(action movie with superheroes and explosions)
        Models->>Caches: Exact query-embedding lookup
        Caches->>Redis: HGETALL content-and-model digest

        alt Query embedding cache hit
            Redis-->>Caches: Stored vector
            Caches->>Redis: Refresh TTL to 600 seconds
            Caches-->>Models: Cached query embedding
        else Query embedding cache miss or cache read failure
            Redis-->>Caches: Missing entry or error
            Models->>Models: Encode query locally
            Models->>Caches: Store query embedding with TTL 600
            Caches->>Redis: HSET and EXPIRE
        end

        Models-->>App: Redis-ready query buffer
    end

    Note over App,Redis: One aggregate request performs three-field retrieval and fusion

    App->>Index: MultiVectorQuery with weights 0.3, 0.5, 0.2<br/>and five requested results
    Index->>Redis: FT.AGGREGATE with three VECTOR_RANGE 2.0 clauses joined by AND
    Redis->>Redis: Yield distance_0, distance_1, and distance_2
    Redis->>Redis: Apply score_i = (2 - distance_i) / 2
    Redis->>Redis: Apply weighted combined_score
    Redis->>Redis: Sort combined_score descending, maximum five
    Redis-->>Index: Titles, descriptions, genres, ratings,<br/>distances, scores, and combined scores
    Index-->>App: Normalized result dictionaries
    App-->>Operator: Print top-five ranked movies

    Note over App,Redis: Movie data is ephemeral; exact embedding caches are reusable

    alt Query and output complete
        App->>Index: delete()
        Index->>Redis: Drop movie index and indexed Hashes
        Redis-->>Index: Movie state removed
        Index-->>Operator: Print completion
    else Any operation raises
        App->>Redis: Attempt scoped FT.DROPINDEX with document deletion
        Redis-->>App: Removed or index-not-found response
        App-->>Operator: Re-raise original error
    end

    Note over Caches,Redis: Cache keys remain until 600 seconds<br/>after their latest successful access
    App->>Redis: Close shared connection pool in finally
    App-->>Operator: Process exits
```
