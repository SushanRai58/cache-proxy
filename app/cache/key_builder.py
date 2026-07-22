"""Config-partition key derivation for the semantic cache."""

from __future__ import annotations

import hashlib
import json

# 64 bits, not the full 256. This hash partitions cache entries by config,
# it isn't defending against an adversary crafting a collision -- so the
# extra collision resistance of a full-length SHA-256 buys nothing here.
HASH_LENGTH = 16


def build_cache_key(system_prompt: str, model: str, temperature: float) -> str:
    # json.dumps on a list, not an f-string join: a delimited string like
    # f"{system_prompt}|{model}" is ambiguous if system_prompt itself
    # contains "|". JSON escapes each field, so the fields can never bleed
    # into each other regardless of their contents.
    payload = json.dumps([system_prompt, model, temperature])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LENGTH]
