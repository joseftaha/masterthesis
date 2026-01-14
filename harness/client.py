"""Async OpenAI-compatible streaming client used to submit a single
request to a running vLLM server and capture per-request metrics.

Uses ``/v1/chat/completions`` with a single ``user`` message so that the
served model's native chat template (Meta-Llama-3 instruct) is applied
server-side. Streaming is enabled and the first non-empty content delta
is taken as ``first_token`` for TTFT.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp


async def submit(session: aiohttp.ClientSession, base_url: str, *,
                 model: str, prompt: str, max_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0,
                 seed: int = 0, request_id: str | None = None) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-ID"] = request_id

    t_dispatch = time.monotonic()
    t_first: float | None = None
    output_tokens = 0
    completion_text_len = 0
    err: str | None = None

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                err = f"http {resp.status}: {(await resp.text())[:200]}"
            else:
                async for raw in resp.content:
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])
                    for ch in obj.get("choices", []):
                        delta = ch.get("delta") or {}
                        text = delta.get("content") or ""
                        if text:
                            if t_first is None:
                                t_first = time.monotonic()
                            completion_text_len += len(text)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        err = f"client: {type(e).__name__}: {e}"

    t_done = time.monotonic()
    ttft = (t_first - t_dispatch) if t_first is not None else None
    e2e = t_done - t_dispatch
    decode_s = (t_done - t_first) if t_first is not None else None
    return {
        "dispatch_ts": t_dispatch,
        "first_token_ts": t_first,
        "finish_ts": t_done,
        "ttft": ttft,
        "e2e": e2e,
        "decode_s": decode_s,
        "output_tokens": output_tokens,
        "completion_chars": completion_text_len,
        "error": err,
    }
