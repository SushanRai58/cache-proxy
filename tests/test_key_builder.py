"""
Standalone smoke test (print+assert style, same as test_local_embedder.py --
run directly, not via pytest) for Phase 2 cache-key correctness.

Usage:
    python tests/test_key_builder.py

Requires Redis Stack running locally on port 6379. Uses a dedicated
"semantic_cache_test" index so it never touches the CLI demo's real
"semantic_cache" index.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.key_builder import build_cache_key
from app.cache.vector_store import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheStatus,
    SemanticCacheStore,
    classify_results,
)
from app.embeddings.local_embedder import LocalEmbedder

TEST_INDEX_NAME = "semantic_cache_test"
THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD


def check(label: str, condition: bool) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, f"FAILED: {label}"


def test_build_cache_key_sanity() -> None:
    print("\n=== build_cache_key sanity checks ===")
    base = build_cache_key("You are helpful.", "gpt-4", 0.7)
    check("same inputs -> same hash", base == build_cache_key("You are helpful.", "gpt-4", 0.7))
    check("different system_prompt -> different hash", base != build_cache_key("You are a pirate.", "gpt-4", 0.7))
    check("different model -> different hash", base != build_cache_key("You are helpful.", "gpt-3.5-turbo", 0.7))
    check("different temperature -> different hash", base != build_cache_key("You are helpful.", "gpt-4", 0.9))


def test_config_aware_classification(embedder: LocalEmbedder) -> None:
    print("\n=== Config-aware classification (single base entry) ===")
    store = SemanticCacheStore(dims=embedder.dimensions, index_name=TEST_INDEX_NAME)
    store.reset()

    base_prompt = "How do I reverse a linked list in Python?"
    base_response = "[dummy response]"
    base_config = {"system_prompt": "You are a coding assistant.", "model": "gpt-4", "temperature": 0.7}
    store.store(base_prompt, base_response, embedder.embed(base_prompt), **base_config)

    query_vector = embedder.embed(base_prompt)

    status, matched = classify_results(store.query(query_vector, top_k=5, **base_config), THRESHOLD)
    check("same prompt, same config -> HIT", status is CacheStatus.HIT)
    check("HIT returns the matched entry's response", matched is not None and matched["response"] == base_response)

    cfg = {**base_config, "temperature": 0.2}
    status, matched = classify_results(store.query(query_vector, top_k=5, **cfg), THRESHOLD)
    check("different temperature -> MISS_CONFIG_MISMATCH", status is CacheStatus.MISS_CONFIG_MISMATCH)
    check("MISS returns no matched entry", matched is None)

    cfg = {**base_config, "system_prompt": "You are a poet."}
    status, matched = classify_results(store.query(query_vector, top_k=5, **cfg), THRESHOLD)
    check("different system_prompt -> MISS_CONFIG_MISMATCH", status is CacheStatus.MISS_CONFIG_MISMATCH)

    cfg = {**base_config, "model": "gpt-3.5-turbo"}
    status, matched = classify_results(store.query(query_vector, top_k=5, **cfg), THRESHOLD)
    check("different model -> MISS_CONFIG_MISMATCH", status is CacheStatus.MISS_CONFIG_MISMATCH)

    unrelated_vector = embedder.embed("What's a good sourdough starter recipe?")
    status, matched = classify_results(store.query(unrelated_vector, top_k=5, **base_config), THRESHOLD)
    check("unrelated prompt, same config -> MISS_NO_MATCH", status is CacheStatus.MISS_NO_MATCH)


def test_top_k_prevents_false_mismatch(embedder: LocalEmbedder) -> None:
    print("\n=== top_k=5 prevents the false MISS_CONFIG_MISMATCH top_k=1 would cause ===")
    store = SemanticCacheStore(dims=embedder.dimensions, index_name=TEST_INDEX_NAME)
    store.reset()

    query_prompt = "What is the capital of France?"
    query_config = {"system_prompt": "", "model": "test-model", "temperature": 0.0}
    query_vector = embedder.embed(query_prompt)

    # Same-config entry: a paraphrase, so it's highly similar to the query
    # but not a perfect match -- this pairing (~0.95+ cosine similarity) was
    # already confirmed empirically during the Phase 1 demo.
    same_config_prompt = "What's the capital of France?"
    store.store(same_config_prompt, "[same-config response]", embedder.embed(same_config_prompt), **query_config)

    # Different-config entry: literally identical text to the query, so its
    # similarity is ~1.0 -- guaranteed to outrank the paraphrase above and
    # land in Redis's rank-1 slot ahead of the entry we actually want.
    diff_config = {"system_prompt": "", "model": "other-model", "temperature": 0.0}
    store.store(query_prompt, "[diff-config response]", embedder.embed(query_prompt), **diff_config)

    results = store.query(query_vector, top_k=5, **query_config)
    print("  top-5 results:")
    for i, r in enumerate(results, start=1):
        print(f"    [{i}] similarity={r['similarity']:.4f} config_match={r['config_match']} prompt={r['prompt']!r}")

    check(
        "diff-config entry ranks above same-config entry",
        results[0]["config_match"] is False and results[1]["config_match"] is True,
    )
    check("both entries clear the threshold", all(r["similarity"] >= THRESHOLD for r in results[:2]))

    status, matched = classify_results(results, THRESHOLD)
    check("classifier scans past rank 1 and finds the same-config match -> HIT", status is CacheStatus.HIT)
    check("matched entry is the same-config (rank 2) one, not rank 1", matched is not None and matched["config_match"] is True)

    # Prove this is actually testing what it claims to: feeding the
    # classifier only the rank-1 result (i.e. simulating the old top_k=1
    # behavior) should reproduce the exact bug this test guards against.
    naive_status, naive_matched = classify_results(results[:1], THRESHOLD)
    check(
        "regression check: top_k=1 would have misreported MISS_CONFIG_MISMATCH here",
        naive_status is CacheStatus.MISS_CONFIG_MISMATCH,
    )
    check("...and would have had no matched entry to fall back on", naive_matched is None)


def main() -> int:
    test_build_cache_key_sanity()

    print("\nLoading embedding model...")
    embedder = LocalEmbedder()

    try:
        test_config_aware_classification(embedder)
        test_top_k_prevents_false_mismatch(embedder)
    except RedisConnectionError:
        print(
            "\nCould not connect to Redis. Make sure Redis Stack is running "
            "locally on port 6379.",
            file=sys.stderr,
        )
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
