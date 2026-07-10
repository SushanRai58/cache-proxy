"""RedisVL-backed storage and nearest-neighbor lookup for cached prompt/response pairs."""

from __future__ import annotations

import os

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

DEFAULT_REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "semantic_cache"
KEY_PREFIX = "cache"
VECTOR_FIELD = "vector"


def build_schema(dims: int) -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": INDEX_NAME,
                "prefix": KEY_PREFIX,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {
                    "name": VECTOR_FIELD,
                    "type": "vector",
                    "attrs": {
                        "dims": dims,
                        # flat = exact brute-force search. At demo scale there's no
                        # approximate-search benefit from HNSW, and flat guarantees
                        # exact results, which matters when validating the math.
                        "algorithm": "flat",
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


class SemanticCacheStore:
    """Thin wrapper around a RedisVL SearchIndex for the prompt-cache demo."""

    def __init__(self, dims: int, redis_url: str | None = None) -> None:
        redis_url = redis_url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
        self.index = SearchIndex(build_schema(dims), redis_url=redis_url)

    def reset(self) -> None:
        # overwrite=True drops and recreates the index on each run, so the demo
        # script is deterministic and doesn't accumulate duplicate entries across
        # runs. A long-lived proxy would instead create the index once and just
        # keep appending/querying across its process lifetime.
        self.index.create(overwrite=True)

    def store(self, prompt: str, response: str, vector: np.ndarray) -> None:
        self.index.load(
            [
                {
                    "prompt": prompt,
                    "response": response,
                    VECTOR_FIELD: vector.astype(np.float32).tobytes(),
                }
            ]
        )

    def query_nearest(self, vector: np.ndarray, k: int = 1) -> list[dict]:
        query = VectorQuery(
            vector=vector.astype(np.float32).tobytes(),
            vector_field_name=VECTOR_FIELD,
            return_fields=["prompt", "response", "vector_distance"],
            num_results=k,
        )
        return self.index.query(query)
