"""RedisVL-backed storage and nearest-neighbor lookup for cached prompt/response pairs."""

from __future__ import annotations

import os
from enum import Enum

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

from app.cache.key_builder import build_cache_key

DEFAULT_REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "semantic_cache"
VECTOR_FIELD = "vector"
CONFIG_HASH_FIELD = "config_hash"
DEFAULT_TOP_K = 5


class CacheStatus(Enum):
    HIT = "HIT"
    MISS_NO_MATCH = "MISS_NO_MATCH"
    MISS_CONFIG_MISMATCH = "MISS_CONFIG_MISMATCH"


def build_schema(dims: int, index_name: str = INDEX_NAME) -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": index_name,
                # Trailing ":" matters: Redis's PREFIX matching is a raw
                # string-prefix check, not delimiter-aware. Without the ":",
                # an index named "semantic_cache" would also match keys
                # belonging to "semantic_cache_test", since that whole string
                # literally starts with "semantic_cache". The ":" guarantees
                # a shorter index name can never accidentally swallow a
                # longer sibling's keys.
                "prefix": f"{index_name}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                # tag = exact match, not tokenized. A "text" field would let
                # Redis fuzzy/partial-match the hash, which is meaningless
                # for something that's only ever compared for equality.
                # Bonus: since the hash is pure hex, it can never contain a
                # tag-field separator character, so it needs no escaping.
                {"name": CONFIG_HASH_FIELD, "type": "tag"},
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
    """Thin wrapper around a RedisVL SearchIndex for the semantic-cache demo."""

    def __init__(self, dims: int, redis_url: str | None = None, index_name: str = INDEX_NAME) -> None:
        redis_url = redis_url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
        self.index = SearchIndex(build_schema(dims, index_name), redis_url=redis_url)

    def reset(self) -> None:
        # overwrite=True recreates the index *definition*, but by itself
        # leaves existing data keys under this prefix untouched -- RediSearch
        # will happily pick up old keys (even ones written under a prior
        # schema, missing fields the current schema expects) as soon as a
        # matching-prefix index exists again. drop=True additionally deletes
        # those keys, which is what actually makes each run start clean.
        # A long-lived proxy would do neither -- create the index once and
        # keep appending/querying across its process lifetime.
        self.index.create(overwrite=True, drop=True)

    def store(
        self,
        prompt: str,
        response: str,
        vector: np.ndarray,
        *,
        system_prompt: str,
        model: str,
        temperature: float,
    ) -> None:
        config_hash = build_cache_key(system_prompt, model, temperature)
        self.index.load(
            [
                {
                    "prompt": prompt,
                    "response": response,
                    CONFIG_HASH_FIELD: config_hash,
                    VECTOR_FIELD: vector.astype(np.float32).tobytes(),
                }
            ]
        )

    def query(
        self,
        vector: np.ndarray,
        *,
        system_prompt: str,
        model: str,
        temperature: float,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        query_config_hash = build_cache_key(system_prompt, model, temperature)

        # Deliberately NOT filtered by config_hash server-side. A Redis-side
        # filter (FT.SEARCH ... @config_hash:{...}) would make "no
        # semantically similar prompt exists at all" and "one exists, but
        # under a different config" both come back as an empty result set --
        # there'd be no way to tell those two cases apart downstream. So the
        # search deliberately over-fetches (top_k=5, not 1) across ALL
        # configs, and classify_results() below reasons about config match
        # in Python where both cases stay observable.
        vector_query = VectorQuery(
            vector=vector.astype(np.float32).tobytes(),
            vector_field_name=VECTOR_FIELD,
            return_fields=["prompt", "response", CONFIG_HASH_FIELD, "vector_distance"],
            num_results=top_k,
        )
        raw_results = self.index.query(vector_query)

        annotated = []
        for result in raw_results:
            result = dict(result)
            # Redis's COSINE distance metric returns 1 - similarity, not
            # similarity -- convert once here so every caller downstream
            # (classify_results, the CLI, tests) works in similarity terms.
            result["similarity"] = 1 - float(result["vector_distance"])
            result["config_match"] = result[CONFIG_HASH_FIELD] == query_config_hash
            annotated.append(result)
        return annotated


def classify_results(results: list[dict], threshold: float) -> CacheStatus:
    """Turn query()'s annotated results into a HIT / MISS_* verdict.

    Redis's KNN search already returns nearest-first, so scanning the list
    in order is scanning in similarity order -- no re-sort needed. Scanning
    (rather than only looking at results[0]) is what lets a same-config
    match at rank 2+ still register as a HIT even when a different-config
    entry happens to be marginally more similar and outranks it.
    """
    any_similar_entry = False
    for result in results:
        if result["similarity"] >= threshold:
            any_similar_entry = True
            if result["config_match"]:
                return CacheStatus.HIT
    return CacheStatus.MISS_CONFIG_MISMATCH if any_similar_entry else CacheStatus.MISS_NO_MATCH
