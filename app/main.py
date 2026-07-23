"""FastAPI proxy: OpenAI-compatible /v1/chat/completions with semantic caching in front."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.cache.vector_store import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheStatus,
    SemanticCacheStore,
    classify_results,
)
from app.embeddings.local_embedder import LocalEmbedder
from app.providers.router import complete as provider_complete

app = FastAPI(title="semantic-cache proxy")

# Loaded once at process startup, not per-request: the embedding model takes
# real time to load, and every request needs both of these regardless of
# whether it ends up a cache hit or a provider call.
embedder = LocalEmbedder()
cache = SemanticCacheStore(dims=embedder.dimensions)
cache.ensure_index()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0


def _extract_prompt_and_system(messages: list[ChatMessage]) -> tuple[str, str]:
    system_prompt = ""
    for message in messages:
        if message.role == "system":
            system_prompt = message.content  # last system message wins if there are several

    user_contents = [message.content for message in messages if message.role == "user"]
    if not user_contents:
        raise HTTPException(status_code=400, detail="messages must include at least one user message")
    return user_contents[-1], system_prompt


def _openai_response(content: str, model: str, cache_status: CacheStatus) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "cache_status": cache_status.value,
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict:
    prompt, system_prompt = _extract_prompt_and_system(request.messages)
    vector = embedder.embed(prompt)

    results = cache.query(
        vector,
        system_prompt=system_prompt,
        model=request.model,
        temperature=request.temperature,
        top_k=5,
    )
    status, matched = classify_results(results, DEFAULT_SIMILARITY_THRESHOLD)

    if status is CacheStatus.HIT:
        return _openai_response(matched["response"], request.model, status)

    # MISS_NO_MATCH or MISS_CONFIG_MISMATCH: nothing usable in the cache,
    # so actually call the LLM. Note the cache is keyed off only the last
    # user message + system prompt (a single-turn approximation), but the
    # provider gets the full conversation -- caching trades away multi-turn
    # precision for a simple, single embedding per request; the real call
    # still gets full context.
    raw_messages = [message.model_dump() for message in request.messages]
    content = provider_complete(raw_messages, request.model, request.temperature)

    cache.store(
        prompt,
        content,
        vector,
        system_prompt=system_prompt,
        model=request.model,
        temperature=request.temperature,
    )

    return _openai_response(content, request.model, status)
