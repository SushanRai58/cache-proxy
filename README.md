# semantic-cache

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
│   ├── main.py                 # FastAPI proxy: POST /v1/chat/completions
│   ├── embeddings/
│   │   └── local_embedder.py   # text -> vector, and vector-vector cosine similarity
│   ├── cache/
│   │   ├── key_builder.py      # hashes (system_prompt, model, temperature) into a config partition key
│   │   └── vector_store.py     # RedisVL schema, storage, config-aware nearest-neighbor query
│   └── providers/
│       ├── groq_client.py      # complete(messages, model, temperature) -> str, via Groq's API
│       ├── ollama_client.py    # same interface, via a local Ollama server
│       └── router.py           # picks a provider by model name
├── tests/
│   ├── test_local_embedder.py  # CLI demo tying the embedder + cache store together
│   └── test_key_builder.py     # standalone checks for config-partitioned cache hits/misses
├── requirements.txt
└── README.md
```

`local_embedder.py` only knows about turning text into vectors and comparing vectors.
`vector_store.py` only knows about Redis, and now also `key_builder.py` for turning a
request's config into a partition key. None of these depend on each other — the
scripts under `tests/` are what wire them together. That separation is what lets a
future proxy phase import the pieces it needs without dragging in CLI or
argument-parsing concerns.

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
python tests/test_local_embedder.py "What's the capital of France?" "What is the capital of France?"
```

This will:
1. Embed both prompts locally (one batched call).
2. Print the direct cosine similarity between them.
3. Recreate the Redis index and store prompt 1 + a dummy response, under a config
   (`--system-prompt` / `--model` / `--temperature`, defaulting to `""` / `test-model` / `0.0`).
4. Run a top-5 nearest-neighbor query for prompt 2 against that index, under a query
   config (`--query-system-prompt` / `--query-model` / `--query-temperature`, each
   defaulting to its store-side counterpart if omitted).
5. Report a `CacheStatus`: `HIT`, `MISS_NO_MATCH`, or `MISS_CONFIG_MISMATCH`.

Try it with two unrelated prompts too (e.g. swap the second one for `"How do I bake
sourdough bread?"`) to see `MISS_NO_MATCH`. Or keep the prompts identical but change
the query-side config to see `MISS_CONFIG_MISMATCH`:

```bash
python tests/test_local_embedder.py "What's the capital of France?" "What is the capital of France?" --model gpt-4 --query-model gpt-3.5-turbo
```

## Running the tests

```bash
python tests/test_key_builder.py
```

A standalone script (not pytest — run it directly) that prints and asserts its way
through: `build_cache_key` hashing sanity checks, the five HIT/MISS_NO_MATCH/
MISS_CONFIG_MISMATCH scenarios from a single stored entry, and a regression check for
the top-5-vs-top-1 ranking bug described below.

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

## Phase 2: cache-key correctness

Phase 1 only ever stored one entry and matched purely on prompt-text similarity. That's
a real bug waiting to happen: the same prompt sent under two different configs (a
different system prompt, model, or temperature) should almost never share a cached
response — the wrong config can mean a completely different answer — but Phase 1's
matcher couldn't tell those cases apart from a genuine cache hit.

**Why the config gets hashed instead of stored/compared as raw fields.** `key_builder.py`
collapses `(system_prompt, model, temperature)` into one 16-hex-char SHA-256-derived
key. That gives a single fixed-width value to store and compare as a Redis `TAG`
field, instead of three separate fields (one of them, `system_prompt`, of unbounded
length) that would all need comparing on every lookup. It's not a security hash —
16 hex chars (64 bits) is plenty of collision resistance for partitioning legitimate
configs, and there's no adversary here trying to force one config to collide with
another.

**Why `json.dumps(...)` instead of an f-string join.** `f"{system_prompt}|{model}"`
breaks the moment `system_prompt` itself contains a `|`, silently merging what should
be two distinct configs into the same partition key. `json.dumps([system_prompt,
model, temperature])` escapes each field, so no field's contents can ever be
misread as a delimiter.

**Why `config_hash` is a `TAG` field, not `TEXT`.** RediSearch `TEXT` fields are
tokenized for fuzzy/partial matching — appropriate for `prompt`/`response`, wrong for
a hash that should only ever be compared for exact equality. `TAG` fields are exact-match
by design. (As a side effect, since the hash is pure hex, it can never contain a
tag-field separator character, so there's nothing to escape either.)

**Why the vector search is NOT filtered server-side by `config_hash`, and why `top_k=5`
instead of `top_k=1` — the two design choices that matter most here.** The naive
approach is to pass `config_hash` as a Redis-side filter on the vector query, so Redis
only ever returns same-config candidates. That fails in a specific, easy-to-miss way:
if a semantically similar prompt exists but was cached under a *different* config, a
server-side filter makes it invisible — the query returns empty, identical to the
case where no similar prompt exists at all. Those are two very different situations
(one means "definitely call the LLM," the other could mean "you're one config
mismatch away from a valid cache entry") and a filtered query can't tell them apart.

So `query()` runs completely unfiltered — across all configs — and asks for the top 5
nearest neighbors, not just the top 1. Each result gets annotated with `config_match`
in Python, and `classify_results()` scans them in similarity order (Redis already
returns nearest-first) looking for the first result that's both above the similarity
threshold *and* config-matched. This is where `top_k=5` earns its keep: if a
different-config entry happens to be marginally more similar to the query than a
same-config entry — both above threshold — a `top_k=1` search would only ever see the
different-config one, and misreport `MISS_CONFIG_MISMATCH` even though a valid hit was
sitting right behind it in rank 2. `tests/test_key_builder.py` builds exactly that
scenario (an identical-text entry under one config outranking a paraphrase under the
query's own config) to prove `top_k=5` + Python-side scanning finds the real hit that
`top_k=1` would have missed.

**Why `classify_results()` returns a `CacheStatus` enum, not a bool.** A bool can only
answer "did it hit," collapsing `MISS_NO_MATCH` (no similar prompt exists — no
sign it's fixable) and `MISS_CONFIG_MISMATCH` (a similar prompt exists, just under the
wrong config — informative, and a real proxy might use this to log config-fragmentation
stats or even decide whether to serve the cached response's *shape* while regenerating
content) into the same "miss" bucket. Keeping them distinct up through the return type
means callers don't have to re-derive the distinction later. It's named
`classify_results`, not `is_cache_hit`, precisely because "is_" implies a bool.

**Why `store()`/`query()` have no default values for `system_prompt`/`model`/
`temperature`, and why they're keyword-only.** Before this change, a caller could
silently keep using the old two-argument shape and never notice their cache was
config-blind. Making the new parameters required breaks any old call site loudly at
the call, rather than letting it keep running with a latent correctness bug. Keyword-only
(`*,` in the signature) additionally rules out accidentally passing `model` where
`system_prompt` was meant, since callers must name each argument explicitly.

## Two bugs found while verifying Phase 2 against real Redis

Both were invisible from reading the code — they only showed up once `tests/test_key_builder.py`
actually ran against a live Redis instance that already had data in it from earlier sessions.
Worth recording since neither is obvious in hindsight:

**`reset()` wasn't actually resetting.** `index.create(overwrite=True)` recreates the RediSearch
*index definition*, but leaves existing data keys untouched. RediSearch then happily
re-indexes any pre-existing key matching the prefix — including ones written under a
now-stale schema, missing fields the current code expects. That surfaced as a `KeyError:
'config_hash'` on a leftover key from before `config_hash` existed at all. Fix:
`index.create(overwrite=True, drop=True)` — `drop=True` is what actually deletes the old keys.

**Index prefixes need a trailing delimiter.** Making each `SemanticCacheStore` use its own
`index_name` as its Redis key prefix (so a dedicated test index wouldn't see the CLI demo's
data, or vice versa) *looked* like enough isolation — but Redis's `PREFIX` matching in
`FT.CREATE` is a raw string-prefix check, not delimiter-aware. An index prefixed `semantic_cache`
also matches keys belonging to an index prefixed `semantic_cache_test`, since the latter's
key names literally start with the former's prefix as a substring. Fix: every prefix now ends
in `":"` (`f"{index_name}:"`), so a shorter name can never be an accidental string-prefix of a
longer sibling's.

## What's out of scope for Phase 2

No HTTP proxy or interception of real LLM API calls yet, no cache eviction/TTL
policy, and no attempt at query-time optimization for large numbers of distinct
configs (a production system with thousands of configs would likely want a smarter
approach than "always fetch top-5 across every config and filter in Python" — but at
demo scale, that cost is negligible, and it's what makes the HIT/MISS_NO_MATCH/
MISS_CONFIG_MISMATCH distinction possible at all). Those remain Phase 3+ concerns.

## Phase 3: FastAPI proxy + provider routing

Phases 1-2 proved the cache logic works in isolation, but nothing actually intercepted
a real LLM call yet. Phase 3 wraps `SemanticCacheStore` and `LocalEmbedder` in an HTTP
proxy that mirrors the OpenAI chat completions API shape: `POST /v1/chat/completions`
with `{"model", "messages", "temperature"}` in, an OpenAI-shaped response out, plus an
extra `cache_status` field showing `HIT`, `MISS_NO_MATCH`, or `MISS_CONFIG_MISMATCH` so
the caching behavior is visible during testing rather than hidden.

**Why the embedder and cache store are loaded once at module scope, not per-request.**
`app/main.py` creates `embedder = LocalEmbedder()` and `cache = SemanticCacheStore(...)`
at import time, not inside the request handler. Loading `all-MiniLM-L6-v2` takes real
time; doing that once at process startup instead of on every request is the difference
between a proxy that adds milliseconds of overhead and one that adds seconds.

**Why `ensure_index()` exists as a separate method from `reset()`.** The demo/test
scripts want a clean slate every run (`reset()`, which drops and recreates). A proxy is
the opposite: it's meant to keep serving cache hits across its whole lifetime, so
restarting it should NOT wipe previously cached responses. `ensure_index()` creates the
index only if it's missing (`overwrite=False`) and otherwise leaves both the index and
its data untouched — this is the "long-lived proxy" behavior the Phase 1/2 READMEs
already said would eventually be needed.

**Why `classify_results()` now returns `(CacheStatus, matched_entry | None)` instead of
just `CacheStatus`.** Phase 2 only ever needed to know *whether* a hit occurred. The
proxy needs the actual cached `response` text to return to the caller on a `HIT` — the
status alone doesn't carry that. Rather than have `main.py` re-scan the results itself
(duplicating the exact same scan-in-similarity-order logic that already lives in
`classify_results`), the function now hands back the winning entry directly. `tests/`
were updated for the new return shape, including new assertions that a `HIT` carries a
non-`None` matched entry and a `MISS` carries `None`.

**Why the cache only ever sees the last user message + system prompt, but the provider
gets the full conversation.** `SemanticCacheStore` embeds and matches on a single piece
of text — there's no notion of multi-turn history in its interface. Rather than build
that in now, the proxy embeds only the latest user message (a single-turn
approximation of "what is this request asking"), while `provider_complete()` still
receives every message in the request, so the actual LLM call keeps full context. The
tradeoff: a cache hit is judged only on the latest turn, so two different conversations
that happen to end in the same question could hit each other's cached answer even if
earlier turns differed. Acceptable for proving the pipeline end-to-end; a real system
would likely hash/embed more of the conversation.

**Why provider routing is "one function per provider, same signature," picked by a
lookup in `router.py` — not, say, a base class with subclasses.** `groq_client.complete()`
and `ollama_client.complete()` both have the exact same shape:
`(messages: list[dict], model: str, temperature: float) -> str`. `main.py` never needs
to know which one it's calling — it just calls `provider_complete(...)` and gets a
string back. A plain function with a shared signature is enough to make that swap
invisible to the caller; a class hierarchy would add structure (an abstract base,
instantiation, `self`) without buying anything a bare function doesn't already give
here. `router.py` itself is a single `if model in OLLAMA_MODELS: ... else: ...` —
routing by exact model name, since that's the one piece of information the request
already carries that unambiguously identifies which backend should serve it.

**`router.OLLAMA_MODELS`.** Now that Ollama is set up locally with `llama3.2:1b`
pulled, `OLLAMA_MODELS = {"llama3.2:1b"}` — a request with `"model": "llama3.2:1b"`
routes to Ollama, anything else still routes to Groq. Confirms the design held up
exactly as intended: adding Ollama support was a one-line change to this set, nothing
else in `main.py` or either provider client needed to change. (Originally set up with
`phi3:mini`, swapped to the smaller `llama3.2:1b` after `phi3:mini`'s ~1.9 GB compute
buffer failed to allocate under low free RAM — see the note on model size tradeoffs
in the setup steps below.)

**Why the Groq client lazily initializes its client object instead of doing it at
import time.** `main.py` imports `router`, which imports `groq_client` — if that import
eagerly constructed a `Groq(api_key=...)` client, simply importing `app.main` would
crash whenever `GROQ_API_KEY` isn't set, even for a request that's served entirely from
cache or would have routed to Ollama. Constructing the client lazily, on first actual
use, means the API key is only required when a request genuinely needs to reach Groq.

**Why `GROQ_API_KEY` is read from the environment, never hardcoded or passed in a
request.** It's a secret tied to your account and billing — putting it in code (or
committing it) risks leaking it into git history or logs. Reading it from the
environment keeps it out of the codebase entirely; you set it once per shell session
before starting the server.

## Running the proxy

```bash
uvicorn app.main:app --reload --port 8000
```

Requires `GROQ_API_KEY` set in the environment first (`export GROQ_API_KEY=...` /
`$env:GROQ_API_KEY = "..."` on PowerShell) and Redis Stack running, same as the earlier
phases.

## Testing the proxy

Three requests that exercise the three `cache_status` outcomes:

1. A prompt with nothing cached yet -> a MISS, routed to Groq.
2. A near-duplicate of it, same config -> `HIT`, served from Redis, no Groq call.
3. A request with `"model": "llama3.2:1b"` -> routed to the local Ollama server
   instead of Groq, per `router.OLLAMA_MODELS`.

See exact commands below.

## What's out of scope for Phase 3

No streaming (the whole response is generated before anything is returned to the
client — that's Phase 7 per the roadmap). Minimal error handling: a malformed request
gets a basic 400, but a Groq/Ollama call failing outright isn't retried or gracefully
degraded — this phase proves the pipeline works end-to-end, not production robustness.
No cache eviction/TTL still. Multi-turn conversations
are cached on a single-turn approximation (see the design-choices note above) rather
than considering the full message history.
