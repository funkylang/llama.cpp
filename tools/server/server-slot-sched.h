#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    // OpenAI prompt_cache_key of the conversation that owns this state.
    // Empty for tasks without a key. Follows the state to the prompt cache and back, like lora.
    std::string cache_key;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora,
            cache_key,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // write the state of an extracted entry into a slot (MOVE: consumes it, COPY: caller keeps it)
    // matched_len: LCP of incoming prompt vs this entry, for PCACHE_HIT logging only
    bool restore_entry(server_prompt & prompt, server_prompt_cache_state & extracted, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, bool is_move, int matched_len);

    // remove the entry at position idx from the cache; out of range yields an empty state
    server_prompt_cache_state extract_entry(size_t idx);

    // put an entry back at the end of the cache (LRU position); skips empty prompts
    void append_entry(server_prompt_cache_state && state);

    void update();
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then restore the selected entry (if any)
    bool update_cache = false;

    // index into prompt_cache->states of the selected cache entry, -1 when none
    int entry_index = -1;

    // MOVE: full match consumes the entry; COPY: partial match keeps it in the cache
    bool is_move = false;

    // LCP of the incoming prompt vs the selected cache entry, for PCACHE_HIT logging only; -1 when none
    int matched_len = -1;

    // its best content match is a row whose task is still decoding: defer this
    // task on that row for a short grace window (close/STOP usually arrives
    // within milliseconds of a new same-conversation request). -1 when none.
    // The caller owns the 100ms deadline; after it expires, re-select with
    // busy rows excluded and take the next best candidate.
    int wait_row = -1;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // unified longest-LCP selection over idle slots and prompt-cache entries (improve/concept-cache-mgmt.txt)
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task,
            const std::vector<common_adapter_lora_info> & lora, const server_prompt_cache * cache) const;
};
