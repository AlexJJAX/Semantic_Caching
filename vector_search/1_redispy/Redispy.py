"""
Vector Search with redis-py

Demonstrates how to use the redis-py client to store, index, and query
movie data using Redis Open Source's vector search capabilities. Covers:
  1. Connecting to Redis and loading movie data from JSON
  2. Embedding movie descriptions with SentenceTransformer (v5)
  3. Defining a FLAT vector index schema with a key prefix
  4. Basic vector search (KNN)
  5. Hybrid filter vector searches (genre, rating, full-text, wildcard, fuzzy)
  6. Range queries with distance thresholds
  7. Full-text BM25 search
  8. Aggregations (average rating per genre)
  9. Weighted / boosted searches

Resources: place resources/movies.json in the working directory.
Redis must be running (default: localhost:6379).
"""

# --- Stdlib ---
import json

# --- Third-party ---
import numpy as np
import redis.commands.search.reducers as reducers
from redis import Redis
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError
from sentence_transformers import SentenceTransformer

from redis_ai_portfolio.config import get_settings
from redis_ai_portfolio.redis import create_redis_client

# --- Redis Connection ---

SETTINGS = get_settings()
REDIS_URL = SETTINGS.redis_url


# --- Constants ---

INDEX_NAME = SETTINGS.redis_name("idx", "movies", "redispy")
KEY_PREFIX = f"{SETTINGS.redis_name('movie', 'redispy')}:"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM    = 384               # all-MiniLM-L6-v2 produces 384-dimensional embeddings


# --- Helpers ---

def embed_text(model: SentenceTransformer, text: str) -> bytes:
    """
    Encode text and return raw float32 bytes for storage in a Redis Hash.

    SentenceTransformer v5: use precision='float32' directly instead of
    manually casting with np.array(...).astype(np.float32).
    """
    embedding: np.ndarray = model.encode(text, precision="float32", convert_to_numpy=True)
    return embedding.tobytes()


def print_results(res) -> None:
    """Print the top-N search results as (title, genre, rating) tuples."""
    docs = [(doc.title, doc.genre, doc.rating) for doc in res.docs]
    print(f"  Top {len(docs)} results: {docs}")


def load_docs(client: Redis, data: list[dict]) -> None:
    """
    Store movie dicts as Redis hashes.

    redis-py 7.x: keys should be namespaced strings (e.g. 'movie:0') rather
    than bare integers so they match the IndexDefinition prefix filter.
    """
    pipeline = client.pipeline(transaction=False)
    for i, doc in enumerate(data):
        pipeline.hset(f"{KEY_PREFIX}{i}", mapping=doc)
    pipeline.execute()


def tokenize(phrase: str) -> str:
    """Convert a phrase into a Redis BM25 OR-token query string."""
    return " | ".join(phrase.lower().split())


# --- Main Demo ---

def run_demo(client: Redis) -> None:
    """Run the vector-search queries using an initialized Redis client."""

    # --- Load and embed movies ---
    with open("resources/movies.json", "r") as f:
        movies = json.load(f)

    # SentenceTransformer v5: model initialisation unchanged
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed movie descriptions using SentenceTransformer v5's precision param
    movie_data = [
        {**movie, "vector": embed_text(model, movie["description"])}
        for movie in movies
    ]
    print(f"Embedded {len(movie_data)} movies.")

    # --- Define index schema ---
    # FLAT is exact and is the simplest choice for this small demonstration dataset.
    schema = (
        VectorField(
            "vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": len(movie_data),
            },
        ),
        NumericField("rating"),
        TagField("genre"),
        TextField("title"),
        TextField("description"),
    )

    # Use a key prefix so the index only covers "movie:*" hashes
    # (avoids accidentally indexing unrelated keys in the same Redis DB).
    definition = IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH)

    try:
        client.ft(INDEX_NAME).info()
        print("Index already exists — dropping and re-creating.")
        client.ft(INDEX_NAME).dropindex(delete_documents=True)
    except ResponseError:
        pass  # Index did not exist yet

    client.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
    print("Index created.")

    # --- Populate index via pipeline ---
    load_docs(client, movie_data)
    res = client.ft(INDEX_NAME).search("*")
    print(f"Index loaded with {res.total} documents.\n")

    # -----------------------------------------------------------------------
    # Vector searches — dialect=2 is required for KNN and range queries.
    # query_params keys must match the $ placeholders in the query string.
    # -----------------------------------------------------------------------

    user_query     = "High tech movies"
    embedded_query = embed_text(model, user_query)

    # --- Basic KNN vector search ---
    print("--- Basic KNN (top 3) ---")
    q = Query("(*)=>[KNN 3 @vector $vec AS dist]").sort_by("dist").dialect(2)
    res = client.ft(INDEX_NAME).search(q, query_params={"vec": embedded_query})
    print_results(res)

    # --- Hybrid: filter by genre tag ---
    print("\n--- Hybrid KNN: action genre only ---")
    # Tag field syntax: @<field>:{ <tag> | <tag> }
    q = Query("(@genre:{action})=>[KNN 3 @vector $vec AS dist]").sort_by("dist").dialect(2)
    res = client.ft(INDEX_NAME).search(q, query_params={"vec": embedded_query})
    print_results(res)

    # --- Hybrid: genre AND minimum rating ---
    print("\n--- Hybrid KNN: action genre + rating >= 7 ---")
    q = (
        Query("(@genre:{action} & (@rating:[7 inf]))=>[KNN 3 @vector $vec AS dist]")
        .sort_by("dist")
        .dialect(2)
    )
    res = client.ft(INDEX_NAME).search(q, query_params={"vec": embedded_query})
    print_results(res)

    # --- Hybrid: full-text phrase match in description ---
    print("\n--- Hybrid KNN: description contains 'criminal mastermind' ---")
    q = (
        Query("(@description:(criminal mastermind))=>[KNN 3 @vector $vec AS dist]")
        .sort_by("dist")
        .dialect(2)
    )
    res = client.ft(INDEX_NAME).search(q, query_params={"vec": embedded_query})
    print_results(res)

    # --- Hybrid: wildcard prefix match ---
    print("\n--- Hybrid KNN: description wildcard 'crim*' ---")
    q = Query("(@description:(crim*))=>[KNN 3 @vector $vec AS dist]").sort_by("dist").dialect(2)
    res = client.ft(INDEX_NAME).search(q, query_params={"vec": embedded_query})
    print_results(res)

    # --- Hybrid: fuzzy Levenshtein match ---
    print("\n--- Hybrid KNN: description fuzzy '%hero%' (Levenshtein distance 1) ---")
    # One leading/trailing % = Levenshtein distance of 1.
    # See: https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/query_syntax/
    q = Query("(@description:%hero%)=>[KNN 3 @vector $vec AS dist]").sort_by("dist").dialect(2)
    res = client.ft(INDEX_NAME).search(q, query_params={"vec": embedded_query})
    print_results(res)

    # --- Range query: cosine distance threshold ---
    print("\n--- Range query: cosine distance <= 0.8 for 'Family friendly fantasy movies' ---")
    user_query     = "Family friendly fantasy movies"
    embedded_query = embed_text(model, user_query)

    q = (
        Query("@vector:[VECTOR_RANGE $radius $vec]=>{$YIELD_DISTANCE_AS: dist}")
        .sort_by("dist")
        .return_fields("title", "rating", "genre", "dist")
        .dialect(2)
    )
    res = client.ft(INDEX_NAME).search(q, query_params={"radius": 0.8, "vec": embedded_query})
    print_results(res)

    # --- Range query: OR with high-rating condition ---
    print("\n--- Range query: within 0.7 OR rating >= 9 ---")
    q = (
        Query("@rating:[9 +inf] | @vector:[VECTOR_RANGE $radius $vec]=>{$YIELD_DISTANCE_AS: dist}")
        .sort_by("dist")
        .return_fields("title", "rating", "genre", "dist")
        .dialect(2)
    )
    res = client.ft(INDEX_NAME).search(q, query_params={"radius": 0.7, "vec": embedded_query})
    print_results(res)

    # --- Full-text BM25 search ---
    print("\n--- BM25: 'Criminal mastermind' (token OR, scorer=BM25STD) ---")
    bm25_phrase = "Criminal mastermind"
    bm25_q = (
        Query(tokenize(bm25_phrase))
        .scorer("BM25STD")
        .with_scores()
        .return_fields("title", "genre", "rating", "description")
        .paging(0, 10)
    )
    res = client.ft(INDEX_NAME).search(bm25_q)
    print(f"  {[(d.title, d.score) for d in res.docs]}")

    # --- Aggregation: average rating per genre ---
    print("\n--- Aggregation: avg rating per genre ---")
    # AggregateRequest.dialect() is still required in redis-py 7.x
    agg_req = (
        AggregateRequest("*")
        .group_by(["@genre"], reducers.avg("rating").alias("avg_rating"))
        .dialect(2)
    )
    agg_res = client.ft(INDEX_NAME).aggregate(agg_req)
    print(f"  {agg_res.rows}")

    # --- Weighted / boosted search ---
    print("\n--- Weighted: action (weight 1) vs fuzzy '%superhero%' description (weight 10) ---")
    # Non-action movies with superhero descriptions can outrank pure-action results.
    q = (
        Query(
            "((@genre:{action}=>{$weight: 1}) | (@description:(%superhero%)=>{$weight: 10}))"
        )
        .return_fields("title", "genre", "rating", "description")
        .paging(0, 3)
        .dialect(2)
    )
    res = client.ft(INDEX_NAME).search(q)
    print(f"  {[(d.title, d.genre) for d in res.docs]}")

    # --- Cleanup: delete only this demo's index and owned keys. ---
    client.ft(INDEX_NAME).dropindex(delete_documents=True)
    print("\nDemo index and keys removed. Done.")


def main() -> None:
    """Connect, run the demo, and clean up its namespace even after failures."""
    client = create_redis_client(REDIS_URL)
    try:
        client.ping()
        print("Connected to Redis.\n")
        run_demo(client)
    except Exception:
        try:
            client.ft(INDEX_NAME).dropindex(delete_documents=True)
        except ResponseError:
            pass
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
