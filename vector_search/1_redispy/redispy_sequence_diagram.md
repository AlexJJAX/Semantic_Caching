# Native redis-py Vector Search — End-to-End Sequence

This sequence shows the complete ephemeral lifecycle: local embedding, Hash ingestion, native
Redis Search queries, aggregation, scoped index deletion, and connection cleanup.

```mermaid
sequenceDiagram
    autonumber

    actor Operator
    participant App as Redispy.py<br/>run_demo()
    participant Data as resources/movies.json
    participant Model as Local Sentence Transformer<br/>all-MiniLM-L6-v2
    participant Client as redis-py<br/>Query Builders + Pipeline
    participant Redis as Redis Hash + Search<br/>Namespaced Movie Index

    Note over Operator,Redis: Initialization and local embedding

    Operator->>App: Run vector_search/1_redispy/Redispy.py
    App->>Client: Create bounded Redis client and PING
    Client->>Redis: PING
    Redis-->>Client: PONG
    Client-->>App: Connection ready
    App->>Data: Load movie records
    Data-->>App: 20 movies with metadata and descriptions
    App->>Model: Initialize all-MiniLM-L6-v2
    Model-->>App: Local 384-dimensional encoder ready

    loop Each movie description
        App->>Model: encode(precision=float32)
        Model-->>App: 384-value NumPy vector
        App->>App: Convert vector to 1,536 raw bytes
    end

    Note over App,Redis: Replace only the owned index and Hashes

    App->>Client: FT.INFO for namespaced index
    Client->>Redis: Inspect index

    alt Index already exists
        Redis-->>Client: Existing index metadata
        Client->>Redis: FT.DROPINDEX with document deletion
        Redis-->>Client: Old index and owned Hashes removed
    else Index does not exist
        Redis-->>Client: ResponseError
        Client-->>App: Continue with a clean namespace
    end

    App->>Client: Define Hash schema and key prefix
    Client->>Redis: FT.CREATE with VECTOR, NUMERIC,<br/>TAG, and TEXT fields
    Redis-->>Client: Index created
    App->>Client: Queue 20 HSET operations<br/>in a non-transactional pipeline
    Client->>Redis: Execute pipeline
    Redis-->>Client: Hash writes indexed
    App->>Client: FT.SEARCH all documents
    Client->>Redis: Count indexed records
    Redis-->>Client: 20 documents
    Client-->>App: Index load confirmation

    Note over App,Redis: KNN queries for High tech movies

    App->>Model: Embed High tech movies
    Model-->>App: FLOAT32 query bytes

    App->>Client: Basic KNN 3, bind vec, sort by dist
    Client->>Redis: FT.SEARCH with DIALECT 2
    Redis-->>Client: Three nearest movies
    Client-->>Operator: Print title, genre, and rating

    loop Action TAG; action plus rating;<br/>description terms; prefix; fuzzy term
        App->>Client: Build pre-filtered KNN query
        Client->>Redis: FT.SEARCH with bound vec and DIALECT 2
        Redis-->>Client: Filter-qualified nearest movies
        Client-->>Operator: Print projected results
    end

    Note over App,Redis: Range retrieval for Family friendly fantasy movies

    App->>Model: Embed Family friendly fantasy movies
    Model-->>App: FLOAT32 query bytes
    App->>Client: VECTOR_RANGE radius 0.8
    Client->>Redis: FT.SEARCH with yielded dist
    Redis-->>Client: All distance-qualified movies
    Client-->>Operator: Print results
    App->>Client: Rating at least 9 OR VECTOR_RANGE radius 0.7
    Client->>Redis: FT.SEARCH Boolean range query
    Redis-->>Client: Union of structured and semantic matches
    Client-->>Operator: Print results

    Note over App,Redis: Lexical scoring, aggregation, and boosts

    App->>Client: BM25STD query for criminal OR mastermind
    Client->>Redis: FT.SEARCH with scores
    Redis-->>Client: BM25-ranked documents
    Client-->>Operator: Print titles and scores

    App->>Client: Group by genre and average rating
    Client->>Redis: FT.AGGREGATE with DIALECT 2
    Redis-->>Client: Average rating rows
    Client-->>Operator: Print aggregation

    App->>Client: Action weight 1 OR fuzzy superhero weight 10
    Client->>Redis: Weighted FT.SEARCH with DIALECT 2
    Redis-->>Client: Text-ranked top three
    Client-->>Operator: Print titles and genres

    Note over App,Redis: Success and failure use scoped cleanup

    alt Query catalogue completes
        App->>Client: dropindex(delete_documents=true)
        Client->>Redis: FT.DROPINDEX with document deletion
        Redis-->>Client: Owned index and Hashes removed
        Client-->>Operator: Print completion
    else Any operation raises
        App->>Client: Attempt the same owned-index drop
        Client->>Redis: FT.DROPINDEX with document deletion
        Redis-->>Client: Removed or index-not-found response
        App-->>Operator: Re-raise original error
    end

    App->>Client: Close connection pool in finally
    Client-->>Operator: Process exits
```
