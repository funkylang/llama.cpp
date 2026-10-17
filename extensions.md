# llama.cpp — funkylang fork extensions

Extensions to [upstream llama.cpp](https://github.com/ggml-org/llama.cpp) for concurrency-safe cache management and observability, built for the **fllama / Hermes** stack. All changes are additive and backward compatible — no existing `llama-server` endpoint, schema, or client behavior is broken.

> **Written for commit [`28b96c138`](https://github.com/funkylang/llama.cpp/commit/28b96c138fbb2966a57ced2837249533d4aa3ec8)** — funkylang/main, 2026-10-17.
> This is a point-in-time reference: log formats, flags, and algorithms can change with upstream merges and later patches. If something here doesn't match your build, check the commit hash of your checkout against the one above and diff from there.

---

## Concurrency-safe prompt-cache management

The biggest behavioral difference. Upstream evicts prompt-cache entries FIFO when it needs space, so a heavy client can displace another's warm state. This fork replaces that with:

1. **Unified longest-LCP selection.** When a request needs a slot, the server compares every *live slot* and every *cache entry* by their common prefix length (LCP) with the new prompt. The candidate with the longest matching prefix wins; ties favor a live slot.
2. **MOVE vs COPY restore.**
   - **MOVE** — full match: the cached state is consumed into the new slot, so it doesn't double-count.
   - **COPY** — partial match: only the shared prefix is copied in; the original entry stays in the cache and is re-appended to the back of the list (a natural LRU).
3. **Unconditional displaced-state save.** Whenever a non-empty slot is about to be overwritten by a different prompt, its KV state is saved to the prompt cache first — so switching conversations no longer silently discards reusable work.

Net effect: concurrent clients coexist without evicting each other, and repeated conversations reuse their KV instead of re-prefilling. Upstream uses `-ctxcp` (max context checkpoints per slot) and `-cms` (min checkpoint spacing) flags here — those are upstream's; this fork's selection/eviction logic is what differs.

---

## Context-checkpoint eviction

Within a single slot, when the checkpoint count reaches the `-ctxcp` cap:

- The **internal** checkpoint minimizing `gap_left + gap_right` (distance to each neighbor) is evicted first.
- The **first and last** checkpoints are never deleted.
- Ties break to the earliest candidate.

This keeps uniformly useful checkpoints rather than dropping whichever upstream FIFO hit.

---

## Environment variables

### `LLAMA_PROMPT_CACHE_INSTRUMENT`

Set to any non-empty value (e.g., `=1`) to enable prompt-cache and checkpoint lifecycle logging. Disabled by default, so it's free when you don't need it. On startup it prints whether instrumentation is on:

```
prompt cache instrumentation: ENABLED
```

Every cache operation emits a single multi-line log message — header line plus an aligned table of current cache entries:

| Event | Meaning | Format |
|-------|---------|--------|
| `PCACHE_CREATE` | Entry added to the prompt cache | `addr=0x… tok=N size=X MiB entries=N total=X MiB` |
| `PCACHE_EXTRACT` | Entry removed from the list for restore | `addr=0x… tok=N entries_before=N` |
| `PCACHE_APPEND` | Entry re-inserted (COPY restore) | `addr=0x… tok=N entries_after=N` |
| `PCACHE_HIT` | Cached state restored into a slot | `addr=0x… tok=N matched=N type=move\|copy slot=N entries=N total=X MiB` |
| `PCACHE_DELETE` | Entry removed from the cache | `addr=0x… tok=N reason=<r> remaining=N total=X MiB` |

DELETE reasons: `obsolete` (a longer cached prompt fully covers it), `size-limit`, `token-limit`, or `consumed` (MOVE restore took ownership of the entry).

For CREATE / size-limit / token-limit DELETE, the header is followed by a state-dump table — one row per surviving entry showing address, token count, MiB, and snapshot count. For MOVE-consumed DELETEs there are no surviving rows to dump, so it's header-only.

### `LLAMA_PROMPT_DUMP`

Set to a file path to write the exact rendered prompt bytes (after templating/chat-formatting) for every `/v1/chat/completions` request to that file. Overwrites the previous contents on each call. Use with `-` or an empty value to disable.

Example usage:
```sh
LLAMA_PROMPT_DUMP=/tmp/prompt.txt llama-server --port 8080
```

---

## Human-readable log timestamps

Server logs now use human-readable local timestamps instead of raw Unix epochs:

```
2026-10-17 09:15:42.387 I srv ...
```

Format: `YYYY-MM-DD HH:MM:SS.mmm` (local timezone, milliseconds). The date/time is colored for readability in supporting terminals.

---

## Logprobs for speculative-decoding accepted tokens

Upstream left `logprobs` empty for tokens accepted via speculative decoding. This fork populates them correctly so clients relying on logprob scores still work with speculation enabled.
