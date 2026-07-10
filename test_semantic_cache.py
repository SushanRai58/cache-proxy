"""
Manual CLI demo for the Phase 1 semantic-cache core.

Usage:
    python test_semantic_cache.py "What's the capital of France?" "What is the capital of France?"

Embeds two prompts locally, prints their direct cosine similarity, stores the
first prompt (+ a dummy response) in Redis via RedisVL, then looks up the
second prompt's nearest neighbor and reports whether it clears the 0.95
cache-hit similarity threshold.
"""

from __future__ import annotations

import argparse
import sys

from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.vector_store import SemanticCacheStore
from app.embeddings.local_embedder import LocalEmbedder

CACHE_HIT_THRESHOLD = 0.95
DUMMY_RESPONSE = "[dummy cached response for prompt 1]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic cache Phase 1 demo")
    parser.add_argument("prompt1", help="First prompt (the one that gets cached)")
    parser.add_argument("prompt2", help="Second prompt (the one that queries the cache)")
    args = parser.parse_args()

    print(f"Loading embedding model ({LocalEmbedder.__module__})...")
    embedder = LocalEmbedder()

    vec1, vec2 = embedder.embed_batch([args.prompt1, args.prompt2])

    direct_similarity = embedder.cosine_similarity(vec1, vec2)
    print("\n--- Direct similarity (no Redis involved) ---")
    print(f"prompt1: {args.prompt1!r}")
    print(f"prompt2: {args.prompt2!r}")
    print(f"cosine similarity: {direct_similarity:.4f}")

    try:
        store = SemanticCacheStore(dims=embedder.dimensions)
        store.reset()
    except RedisConnectionError:
        print(
            "\nCould not connect to Redis. Make sure Redis Stack is running "
            "locally on port 6379 (Redis Stack, not plain redis-server -- "
            "RedisVL needs the search module).",
            file=sys.stderr,
        )
        return 1

    store.store(args.prompt1, DUMMY_RESPONSE, vec1)

    results = store.query_nearest(vec2, k=1)
    print("\n--- Redis nearest-neighbor lookup for prompt2 ---")
    if not results:
        print("No results returned from Redis (index is empty).")
        return 1

    top = results[0]
    distance = float(top["vector_distance"])
    # Redis's COSINE distance metric returns 1 - cosine_similarity, not
    # similarity itself, so it has to be converted before comparing to the
    # 0.95 similarity threshold.
    redis_similarity = 1 - distance
    is_hit = redis_similarity >= CACHE_HIT_THRESHOLD

    print(f"matched prompt: {top['prompt']!r}")
    print(f"matched response: {top['response']!r}")
    print(f"vector_distance: {distance:.4f}")
    print(f"derived similarity (1 - distance): {redis_similarity:.4f}")
    print(f"threshold: {CACHE_HIT_THRESHOLD}")
    print(f"\nRESULT: {'CACHE HIT' if is_hit else 'CACHE MISS'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
