"""
Vector Search with RedisVL

Demonstrates how to use the redisvl library to store, index, and query
movie data with Redis Open Source. Covers:
  1. Loading movie data from JSON into a Pandas DataFrame
  2. Embedding descriptions with HFTextVectorizer (backed by EmbeddingsCache)
  3. Defining an IndexSchema and creating a SearchIndex
  4. Standard KNN vector search
  5. Filtered vector search (tag, numeric, full-text, wildcard, fuzzy)
  6. Range queries with distance thresholds
  7. Full-text BM25 search via TextQuery
  8. Hybrid vector + BM25 search via AggregateHybridQuery

Dependencies: redisvl>=0.16.0, sentence-transformers, pandas, python-dotenv
Resources: place resources/movies.json in the working directory.
Redis must be running (default: localhost:6379).
"""

# --- Stdlib ---
import os
import warnings

# --- Third-party ---
import pandas as pd
from redis import Redis
from redis.exceptions import ResponseError
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.index import SearchIndex
from redisvl.query import AggregateHybridQuery, RangeQuery, TextQuery, VectorQuery
from redisvl.query.filter import Num, Tag, Text
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import HFTextVectorizer

from redis_ai_portfolio.config import get_settings
from redis_ai_portfolio.redis import create_redis_client

# Suppress noisy third-party warnings (tokenizer parallelism etc.)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Redis Connection ---

SETTINGS = get_settings()
REDIS_URL = SETTINGS.redis_url


# --- Constants ---

INDEX_NAME = SETTINGS.redis_name("idx", "movies", "redisvl")
KEY_PREFIX = SETTINGS.redis_name("movie", "redisvl")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM      = 384   # all-MiniLM-L6-v2 output dimensionality


# --- Main Demo ---

def run_demo(client: Redis) -> None:
    """Run the RedisVL vector-search queries with an initialized client."""

    # --- Load movie dataset ---
    df = pd.read_json("resources/movies.json")
    print(f"Loaded {len(df)} movie entries.")

    # HFTextVectorizer wraps SentenceTransformer with an optional Redis EmbeddingsCache
    # to avoid re-computing the same embeddings across runs (TTL = 600 s).
    # redisvl 0.16: pass dtype at construction so all embed/embed_many calls
    # return float32 by default without having to specify it per call.
    hf = HFTextVectorizer(
        model=EMBEDDING_MODEL,
        dtype="float32",
        cache=EmbeddingsCache(
            name=SETTINGS.redis_name("cache", "embeddings", "redisvl"),
            ttl=600,
            redis_client=client,
        ),
    )

    # redisvl 0.16: the 'texts' kwarg is deprecated — use 'contents' instead.
    # as_buffer=True returns raw float32 bytes, ready for Redis Hash storage.
    df["vector"] = hf.embed_many(contents=df["description"].tolist(), as_buffer=True)
    print(f"Embedded {len(df)} descriptions.\n")

    # --- Define index schema ---
    schema = IndexSchema.from_dict({
        "index": {
            "name": INDEX_NAME,
            "prefix": KEY_PREFIX,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title",       "type": "text"},
            {"name": "description", "type": "text"},
            {
                "name": "genre",
                "type": "tag",
                "attrs": {"sortable": True},
            },
            {
                "name": "rating",
                "type": "numeric",
                "attrs": {"sortable": True},
            },
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "dims": VECTOR_DIM,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    })

    # overwrite=True / drop=True ensures a clean slate inside this demo's namespace.
    # The shared client supplies bounded timeouts and owns the connection pool.
    index = SearchIndex(schema, redis_client=client)
    index.create(overwrite=True, drop=True)
    print(f"Index '{INDEX_NAME}' created.")

    # --- Populate index ---
    index.load(df.to_dict(orient="records"))
    print(f"Loaded {len(df)} records into index.\n")

    # -----------------------------------------------------------------------
    # Search demos
    # -----------------------------------------------------------------------

    user_query = "High tech and action packed movie"
    # redisvl 0.16: the 'text' kwarg in embed() is deprecated — use positional 'content'.
    embedded_query = hf.embed(user_query)

    # --- Standard KNN vector search ---
    print("--- Standard KNN vector search (top 3) ---")
    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        num_results=3,
        return_fields=["title", "genre"],
        return_score=True,
    )
    result = index.query(vec_query)
    print(pd.DataFrame(result).to_string(index=False))

    # --- Filtered vector search: genre tag ---
    print("\n--- Filtered KNN: action genre only ---")
    tag_filter = Tag("genre") == "action"
    vec_query.set_filter(tag_filter)
    result = index.query(vec_query)
    print(pd.DataFrame(result).to_string(index=False))

    # --- Filtered vector search: genre tag AND numeric rating ---
    print("\n--- Filtered KNN: action genre + rating >= 7 ---")
    tag_filter     = Tag("genre") == "action"
    num_filter     = Num("rating") >= 7
    combined_filter = tag_filter & num_filter

    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        num_results=3,
        return_fields=["title", "rating", "genre"],
        return_score=True,
        filter_expression=combined_filter,
    )
    result = index.query(vec_query)
    print(pd.DataFrame(result).to_string(index=False))

    # --- Filtered vector search: full-text phrase match ---
    print("\n--- Filtered KNN: description contains 'criminal mastermind' ---")
    text_filter = Text("description") % "criminal mastermind"
    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        num_results=3,
        return_fields=["title", "rating", "genre", "description"],
        return_score=True,
        filter_expression=text_filter,
    )
    result = index.query(vec_query)
    print(pd.DataFrame(result)[["title", "genre", "vector_distance"]].to_string(index=False))

    # --- Filtered vector search: wildcard prefix ---
    print("\n--- Filtered KNN: description wildcard 'crim*' ---")
    text_filter = Text("description") % "crim*"
    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        num_results=3,
        return_fields=["title", "rating", "genre", "description"],
        return_score=True,
        filter_expression=text_filter,
    )
    result = index.query(vec_query)
    print(pd.DataFrame(result)[["title", "genre", "vector_distance"]].to_string(index=False))

    # --- Filtered vector search: fuzzy Levenshtein match ---
    print("\n--- Filtered KNN: description fuzzy '%hero%' (Levenshtein distance 1) ---")
    # One leading/trailing % = Levenshtein distance of 1.
    # See: https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/query_syntax/
    text_filter = Text("description") % "%hero%"
    vec_query = VectorQuery(
        vector=embedded_query,
        vector_field_name="vector",
        num_results=3,
        return_fields=["title", "rating", "genre", "description"],
        return_score=True,
        filter_expression=text_filter,
    )
    result = index.query(vec_query)
    print(pd.DataFrame(result)[["title", "genre", "vector_distance"]].to_string(index=False))

    # --- Range query: distance threshold ---
    print("\n--- Range query: cosine distance <= 0.8 for 'Family friendly fantasy movies' ---")
    user_query     = "Family friendly fantasy movies"
    embedded_query = hf.embed(user_query)

    range_query = RangeQuery(
        vector=embedded_query,
        vector_field_name="vector",
        return_fields=["title", "rating", "genre"],
        return_score=True,
        distance_threshold=0.8,  # return all items within cosine distance 0.8
    )
    result = index.query(range_query)
    print(pd.DataFrame(result).to_string(index=False))

    # --- Range query: distance threshold AND numeric filter ---
    print("\n--- Range query: distance <= 0.8 AND rating >= 8 ---")
    range_query = RangeQuery(
        vector=embedded_query,
        vector_field_name="vector",
        return_fields=["title", "rating", "genre"],
        distance_threshold=0.8,
    )
    range_query.set_filter(Num("rating") >= 8)
    result = index.query(range_query)
    print(pd.DataFrame(result).to_string(index=False))

    # --- Full-text BM25 search ---
    print("\n--- Full-text BM25 search ---")
    user_query = "High tech, action packed, superheros fight scenes"
    text_query = TextQuery(
        text=user_query,
        text_field_name="description",
        text_scorer="BM25STD",
        num_results=20,
        return_fields=["title", "description"],
        # redisvl 0.16 defaults to NLTK English stopwords; set to None to disable
        # client-side filtering (Redis handles stopwords server-side via index config).
        stopwords=None,
    )
    result = index.query(text_query)[:4]
    print(pd.DataFrame(result)[["title", "score"]].to_string(index=False))

    # --- Hybrid vector + BM25 search ---
    print("\n--- Hybrid search (alpha=0.7 → 70% vector / 30% BM25) ---")
    # AggregateHybridQuery blends vector similarity and BM25 scores via a
    # configurable alpha weight. alpha=1.0 = pure vector, alpha=0.0 = pure BM25.
    # redisvl 0.16: text_scorer default changed to 'BM25STD' (more accurate than
    # plain 'BM25'). Use 'BM25STD' explicitly for clarity and future-proofing.
    embedded_query = hf.embed(user_query)
    hybrid_query = AggregateHybridQuery(
        text=user_query,
        text_field_name="description",
        text_scorer="BM25STD",
        vector=embedded_query,
        vector_field_name="vector",
        alpha=0.7,
        num_results=20,
        return_fields=["title", "description"],
        stopwords=None,  # disable NLTK client-side stopword filtering
    )
    result = index.query(hybrid_query)[:4]
    print(
        pd.DataFrame(result)[["title", "vector_similarity", "text_score", "hybrid_score"]]
        .to_string(index=False)
    )

    # --- Cleanup ---
    index.delete()
    print("\nIndex deleted. Done.")


def main() -> None:
    """Connect, run the demo, and clean up its namespace even after failures."""
    client = create_redis_client(REDIS_URL)
    try:
        client.ping()
        print("Connected to Redis.\n")
        run_demo(client)
    except Exception:
        try:
            client.execute_command("FT.DROPINDEX", INDEX_NAME, "DD")
        except ResponseError:
            pass
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
