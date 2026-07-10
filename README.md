# semantic-cache — Phase 1: Embedding + Similarity Core

A semantic caching layer for LLM APIs. Instead of caching on exact prompt text, this
caches on *meaning* — if a new prompt is close enough in embedding space to a
previously-answered prompt, the cached response can be reused instead of calling the
LLM again.

Phase 1 is deliberately narrow: it proves out embedding generation and vector
similarity search end-to-end, with no proxy, no HTTP server, and no real LLM calls.
Later phases would wrap this core in a proxy that intercepts real API calls.

## Repo structure

```
semantic-cache/
├── app/
│   ├── embeddings/
│   │   └── local_embedder.py   # text -> vector, and vector-vector cosine similarity
│   └── cache/
│       └── vector_store.py     # RedisVL schema, storage, nearest-neighbor query
├── test_semantic_cache.py      # CLI demo tying both pieces together
├── requirements.txt
└── README.md
```

`local_embedder.py` only knows about turning text into vectors and comparing vectors.
`vector_store.py` only knows about Redis. Neither depends on the other — the demo
script is what wires them together. That separation is what lets a future proxy phase
import both pieces without dragging in CLI or argument-parsing concerns.

## Prerequisites

- Python 3.9+
- [Redis Stack](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/)
  running locally on port 6379 (plain `redis-server` is **not** enough — RedisVL needs
  the RediSearch module that ships with Redis Stack, not vanilla Redis)

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

The first run of the demo script downloads the `all-MiniLM-L6-v2` model (~90 MB) from
Hugging Face — that needs internet access once. After that it's cached locally
(`~/.cache/huggingface`) and everything runs offline.

## Running the demo

```bash
python test_semantic_cache.py "What's the capital of France?" "What is the capital of France?"
```

This will:
1. Embed both prompts locally (one batched call).
2. Print the direct cosine similarity between them.
3. Recreate the Redis index and store prompt 1 + a dummy response.
4. Run a nearest-neighbor query for prompt 2 against that index.
5. Report `CACHE HIT` or `CACHE MISS` at a 0.95 similarity threshold.

Try it with two unrelated prompts too (e.g. swap the second one for `"How do I bake
sourdough bread?"`) to see a `CACHE MISS`.

## Design choices

**Why `all-MiniLM-L6-v2`.** It's small (~90 MB), fast on CPU, runs fully locally with
no API key, and produces 384-dimensional embeddings — plenty of semantic resolution
for deciding "are these two prompts close enough to reuse a response," without the
size/latency cost of a larger embedding model. For a cache, embedding speed matters
almost as much as embedding quality: a slow embedder would erode the latency win the
cache is supposed to provide.

**Why normalize embeddings.** `local_embedder.py` calls `.encode(..., normalize_embeddings=True)`.
Cosine similarity is scale-invariant, so this isn't required for correctness — but
normalized vectors mean cosine similarity reduces to a plain dot product, which is
cheaper to compute and is the convention most semantic-cache implementations (e.g.
GPTCache) follow.

**Why compute cosine similarity by hand.** `LocalEmbedder.cosine_similarity` is a
three-line numpy formula instead of a call to `sentence_transformers.util.cos_sim`.
The whole point of a "core" module in a learning-oriented project is that the math
behind cache-hit decisions should be visible, not hidden behind a library call.

**Why RedisVL instead of raw `redis-py` + `FT.*` commands.** RedisVL gives you a typed
schema (`IndexSchema`) and query builders (`VectorQuery`) on top of Redis's search
module, instead of hand-building `FT.CREATE` / `FT.SEARCH` command strings. Same
underlying Redis functionality, much less boilerplate and fewer ways to typo a field
name.

**Why `FLAT` instead of `HNSW` for the vector index.** `HNSW` is an approximate
nearest-neighbor algorithm that pays off at scale (millions of vectors) by trading a
little accuracy for a lot of speed. At demo scale (one stored vector), there's nothing
to gain from that trade-off, and `FLAT` (exact brute-force search) guarantees the
result you get back is *the* correct nearest neighbor — useful when you're trying to
confirm the similarity math itself is right, not just plausible.

**Why the index gets dropped and recreated on every run (`store.reset()`).** It keeps
the demo idempotent and deterministic — every run starts from a clean slate instead of
accumulating duplicate cached prompts from previous runs. A real proxy would do the
opposite: create the index once at startup and keep reading/writing to it for the
life of the process.

**The distance-vs-similarity conversion, and why it's called out explicitly.** Redis's
`COSINE` distance metric returns *distance* (`1 - cosine_similarity`, range 0–2, where
0 means identical), not similarity. `VectorQuery` results come back with a
`vector_distance` field. So checking "is this a cache hit at 0.95 similarity" means
computing `1 - vector_distance` and comparing *that* to 0.95 — not comparing
`vector_distance` to 0.95 directly. Getting this backwards (treating distance as if it
were similarity) is a very easy, very silent bug: everything still runs, the numbers
just mean the opposite of what you think, and low-distance (good) matches get
rejected while high-distance (bad) matches get accepted. `test_semantic_cache.py`
prints both the raw distance and the derived similarity so this conversion step is
visible rather than buried.

**Why 0.95 specifically.** It's the threshold given in the requirements, and it's a
deliberately conservative one: at 0.95 cosine similarity, two prompts need to be
almost paraphrases of each other, not just topically related, to trigger a hit. That
trades some cache-hit rate for a much lower risk of serving a stale/wrong cached
response for a prompt that only superficially resembles the cached one — for an LLM
cache, a false hit (serving the wrong answer) is a worse failure mode than a false
miss (an unnecessary but harmless LLM call).

## What's out of scope for Phase 1

No HTTP proxy, no interception of real LLM API calls, no cache eviction/TTL policy, no
handling of multiple stored entries at once (the demo stores exactly one). Those are
Phase 2+ concerns once this core is validated.
