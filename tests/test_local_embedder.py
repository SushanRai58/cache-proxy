"""
Manual CLI demo for the semantic-cache core (Phase 1 embedding/similarity +
Phase 2 config-aware cache keys).

Usage:
    python tests/test_local_embedder.py "prompt1" "prompt2"
    python tests/test_local_embedder.py "prompt1" "prompt2" --model gpt-4 --temperature 0.7
    python tests/test_local_embedder.py "prompt1" "prompt2" --model gpt-4 --query-model gpt-3.5-turbo

Embeds two prompts locally, prints their direct cosine similarity, stores the
first prompt (+ a dummy response) in Redis via RedisVL under a given
system_prompt/model/temperature config, then looks up the second prompt's
top-5 nearest neighbors under a (possibly different) config and reports a
CacheStatus: HIT, MISS_NO_MATCH, or MISS_CONFIG_MISMATCH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run directly (python tests/test_local_embedder.py), not via `python -m`, so
# Python only puts this script's own directory on sys.path by default. Add
# the repo root explicitly or `from app...` below fails with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.key_builder import build_cache_key
from app.cache.vector_store import SemanticCacheStore, classify_results
from app.embeddings.local_embedder import LocalEmbedder

CACHE_HIT_THRESHOLD = 0.95
DUMMY_RESPONSE = "[dummy cached response for prompt 1]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic cache demo (Phase 1 + Phase 2)")
    parser.add_argument("prompt1", help="First prompt (the one that gets cached)")
    parser.add_argument("prompt2", help="Second prompt (the one that queries the cache)")

    parser.add_argument("--system-prompt", default="", help="System prompt used when storing prompt1 (default: '')")
    parser.add_argument("--model", default="test-model", help="Model used when storing prompt1 (default: test-model)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature used when storing prompt1 (default: 0.0)")

    parser.add_argument("--query-system-prompt", default=None, help="System prompt for the query; defaults to --system-prompt")
    parser.add_argument("--query-model", default=None, help="Model for the query; defaults to --model")
    parser.add_argument("--query-temperature", type=float, default=None, help="Temperature for the query; defaults to --temperature")

    args = parser.parse_args()

    query_system_prompt = args.system_prompt if args.query_system_prompt is None else args.query_system_prompt
    query_model = args.model if args.query_model is None else args.query_model
    query_temperature = args.temperature if args.query_temperature is None else args.query_temperature

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

    store.store(
        args.prompt1,
        DUMMY_RESPONSE,
        vec1,
        system_prompt=args.system_prompt,
        model=args.model,
        temperature=args.temperature,
    )

    results = store.query(
        vec2,
        system_prompt=query_system_prompt,
        model=query_model,
        temperature=query_temperature,
        top_k=5,
    )

    print("\n--- Redis nearest-neighbor lookup for prompt2 (top-5, config-annotated) ---")
    if not results:
        print("No results returned from Redis (index is empty).")
        return 1

    for i, r in enumerate(results, start=1):
        print(
            f"  [{i}] similarity={r['similarity']:.4f} config_match={r['config_match']} "
            f"prompt={r['prompt']!r} response={r['response']!r}"
        )

    status = classify_results(results, CACHE_HIT_THRESHOLD)
    print(f"\nthreshold: {CACHE_HIT_THRESHOLD}")
    print(f"RESULT: {status.value}")

    store_hash = build_cache_key(args.system_prompt, args.model, args.temperature)
    query_hash = build_cache_key(query_system_prompt, query_model, query_temperature)
    if store_hash != query_hash:
        print(f"store config_hash: {store_hash}")
        print(f"query config_hash: {query_hash}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
