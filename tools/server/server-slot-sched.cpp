#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <iterator>

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

//
// server_prompt_cache
//

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            if (pc_instr_enabled()) {
                SRV_INF("PCACHE_DELETE addr=%p tok=%zu reason=obsolete\n",
                        (const void*)&(*it), (size_t)it->prompt.n_tokens());
            }
            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            if (pc_instr_enabled()) {
                SRV_INF("PCACHE_DELETE addr=%p tok=%zu reason=size-limit\n",
                        (const void*)&states.front(), (size_t)states.front().prompt.n_tokens());
            }
            states.pop_front();
        }
    }

    common_state_data state_data_tgt;
    common_state_data state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
            /*.lora        =*/ prompt.lora,
            /*.cache_key   =*/ prompt.cache_key,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    // Instrumentation: log creation
    if (pc_instr_enabled()) {
        SRV_INF("PCACHE_CREATE addr=%p tok=%zu\n",
                (const void*)&states.back(), (size_t)states.back().prompt.n_tokens());
    }

    return &states.back();
}

server_prompt_cache_state server_prompt_cache::extract_entry(size_t idx) {
    if (idx >= states.size()) {
        return server_prompt_cache_state(); // empty state
    }

    auto it = states.begin();
    std::advance(it, (std::ptrdiff_t) idx);

    // take the entry out of the list so it cannot be evicted by a subsequent alloc()
    server_prompt_cache_state extracted = std::move(*it);
    states.erase(it);

    return extracted;
}

void server_prompt_cache::append_entry(server_prompt_cache_state && state) {
    if (state.prompt.tokens.empty()) {
        return; // nothing to append
    }

    states.push_back(std::move(state));
}

// restore the state of an extracted entry into a slot.
// MOVE: the whole prompt is consumed, the entry data moves into the slot.
// COPY: only a prefix matches, the state is written in but the caller keeps
//       the entry to re-append it (the owner conversation stays restorable).
bool server_prompt_cache::restore_entry(server_prompt & prompt, server_prompt_cache_state & extracted,
        llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, bool is_move, int matched_len) {
    if (extracted.prompt.tokens.empty()) {
        return true; // nothing to restore
    }

    // Instrumentation: log hit
    if (pc_instr_enabled()) {
        SRV_INF("PCACHE_HIT addr=%p tok=%zu matched=%d type=%s slot=%d\n",
                (const void*)&extracted, (size_t)extracted.prompt.n_tokens(), matched_len,
                is_move ? "move" : "copy", id_slot);
    }

    auto restore_data = [&](common_state_data & data, llama_context * ctx, bool fatal) -> bool {
        if (data.empty()) {
            return true;
        }

        const size_t size = data.size();
        const size_t n   = llama_state_seq_set_data_ext(ctx, data.data(), size, id_slot, 0);
        if (n != size) {
            if (fatal) {
                SRV_ERR("failed to restore state with size %zu\n", size);
            } else {
                SRV_WRN("failed to restore draft state with size %zu\n", size);
            }

            return false;
        }

        return true;
    };

    if (!restore_data(extracted.data.main, ctx_tgt, true)) {
        return false;
    }

    if (ctx_dft && !restore_data(extracted.data.drft, ctx_dft, false)) {
        return false;
    }

    if (is_move) {
        prompt = std::move(extracted.prompt); // consumed: entry is gone from the cache
    } else {
        prompt = extracted.prompt.clone(); // COPY: original stays with the caller, to be re-appended
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            if (pc_instr_enabled()) {
                SRV_INF("PCACHE_DELETE addr=%p tok=%zu reason=size-limit\n",
                        (const void*)&states.front(), (size_t)states.front().prompt.n_tokens());
            }
            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            if (pc_instr_enabled()) {
                SRV_INF("PCACHE_DELETE addr=%p tok=%zu reason=token-limit\n",
                        (const void*)&states.front(), (size_t)states.front().prompt.n_tokens());
            }
            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task,
        const std::vector<common_adapter_lora_info> & lora, const server_prompt_cache * cache) const {

    const server_slot_info * ret = nullptr;

    // Unified LCP-based candidate selection (improve/concept-cache-mgmt.txt).
    // Candidates: idle live slots AND prompt-cache entries.
    //   qualified : the match covers at least one of the candidate's checkpoints
    //   all       : the match covers the candidate's last checkpoint
    // An "all" match on a live slot is an early winner (no cache scan needed).
    // Otherwise the longest LCP wins; ties prefer the live slot.
    //
    // Slot pinning: task.id_slot != -1 selects that exact slot.
    // A busy pinned slot is returned untouched so the caller defers the task.
    //
    // prompt_cache_key routing (improve/concept-prompt-cache-key-impl.txt):
    // a task carrying a key always routes to its conversation's state - a live
    // owner slot first (busy -> defer), then a matching prompt-cache entry, so
    // identity beats the content-based LCP selection below.

    struct candidate {
        const server_slot_info * slot = nullptr;      // live-slot candidate
        size_t                   entry_idx = 0;       // prompt-cache entry candidate
        bool                     is_entry = false;
        size_t                   lcp = 0;
        bool                     qualified = false;
        bool                     is_all = false;
    };

    bool early_winner = false;

    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task; this includes
            // STOP rows (close detected, not yet released): their release sweep runs first
            // and leaves warm KV for a same-conversation attach on the next pass.
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // cache prompts only for the tasks whose state a later task can reuse;
    // cache_prompt = false marks one-shot queries, neither read nor written
    const bool use_cache = !task.is_stateless() && (cache != nullptr) && task.params.cache_prompt;

    // route by prompt_cache_key before the LCP scan: a task carrying a key must reach its own conversation
    bool   key_matched_slot = false;             // an owner slot was found live, ret points at it
    const server_prompt_cache_state * key_state = nullptr; // owner state waiting in the prompt cache
    size_t key_entry        = 0;                 // its index for extract_entry()

    if (!task.prompt_cache_key.empty() && task.id_slot == -1) {
        // a child slot is tagged with the key too, but keeps no tokens of its own:
        // only a keyed slot that holds state can be the owner, an empty one merely
        // reserves the conversation a home when nothing else exists anymore
        const server_slot_info * key_slot_empty = nullptr;

        for (const auto & slot : slots) {
            if (slot.prompt->cache_key != task.prompt_cache_key) {
                continue;
            }

            // a busy owner (including a STOP row awaiting its release sweep): the caller
            // parks this task on that row; the owner's close/release takes it over with
            // warm KV. Direct launch is left to the sweep, which does the old task's teardown.
            if (slot.is_processing) {
                server_slot_plan p;
                p.id_slot  = slot.id;
                p.wait_row = slot.id;
                return p;
            }

            if (slot.prompt->n_tokens() > 0) {
                SLT_INF(slot, "selected slot by prompt_cache_key, n_tokens = %d\n", (int) slot.prompt->n_tokens());

                if (pc_instr_enabled()) {
                    SRV_INF("PCACHE_KEY slot=%d key=%s\n", slot.id, task.prompt_cache_key.c_str());
                }

                ret              = &slot;
                key_matched_slot = true;
                break;
            }

            if (!key_slot_empty) {
                key_slot_empty = &slot;
            }
        }

        // no live owner: its state may wait in the prompt cache from an earlier eviction
        if (!key_matched_slot && use_cache) {
            size_t idx = 0;
            for (const auto & st : cache->states) {
                if (!st.data.main.empty() && st.prompt.cache_key == task.prompt_cache_key) {
                    key_state = &st;
                    key_entry = idx;
                    break;
                }
                ++idx;
            }
        }

        // nothing to restore either: an empty tagged slot at least reserves the conversation a home
        if (!key_matched_slot && key_state == nullptr && key_slot_empty != nullptr) {
            SLT_INF(*key_slot_empty, "%s", "selected empty slot by prompt_cache_key\n");

            if (pc_instr_enabled()) {
                SRV_INF("PCACHE_KEY slot=%d key=%s\n", key_slot_empty->id, task.prompt_cache_key.c_str());
            }

            ret              = key_slot_empty;
            key_matched_slot = true;
        }

        if (key_state != nullptr) {
            SRV_INF(" - prompt cache entry %p owns key '%s', n_tokens = %d\n",
                    (const void *) key_state, task.prompt_cache_key.c_str(), (int) key_state->prompt.n_tokens());

            if (pc_instr_enabled()) {
                SRV_INF("PCACHE_KEYRESTORE addr=%p key=%s\n", (const void *) key_state, task.prompt_cache_key.c_str());
            }
        }
    }

    auto classify_prompt = [](const server_prompt & p, size_t lcp, bool & qualified, bool & is_all) {
        if (!p.checkpoints.empty()) {
            qualified = lcp >= static_cast<size_t>(p.checkpoints.front()->n_tokens);
            is_all    = lcp >= static_cast<size_t>(p.checkpoints.back()->n_tokens);
        } else {
            // no checkpoints: usable only when the whole prompt is shared
            qualified = (lcp > 0) && (lcp >= p.tokens.size());
            is_all    = qualified;
        }
    };

    // a pinned idle slot is used directly, like an ALL match: no LCP scan,
    // and the save paths below preserve its displaced state.
    if (ret != nullptr) {
        early_winner = true;
    }

    // return true if candidate c beats the current best (longer LCP wins; tie -> live slot preferred)
    auto consider_better = [](const candidate & c, const candidate & best) {
        if (!c.qualified) {
            return false;
        }
        if (!best.qualified) {
            return true;
        }
        if (c.lcp > best.lcp) {
            return true;
        }
        return (c.lcp == best.lcp) && !c.is_entry && best.is_entry;
    };

    // scan live slots: an "all" match is an early winner; otherwise collect the best partial.
    // stopped rows are attachable like idle ones (their close already arrived).
    // keyed slots are off-limits to tasks that do not own the key (id-less requests and
    // keyed ones whose conversation is not live): they may only take them through the
    // forced-LRU fallback below, which saves the displaced state first.
    candidate best;
    // best match among rows still decoding: a grace-wait target when it out-ranks
    // every attachable candidate (the steering close usually arrives right after)
    candidate busy_best;
    if (!early_winner) {
        for (const auto & slot : slots) {
            // the key rule applies to every row, stopped ones included: one
            // conversation's state is never adopted by another task
            if (!slot.prompt->cache_key.empty() && slot.prompt->cache_key != task.prompt_cache_key) {
                SLT_TRC(slot, "%s", " - skipping, owned by another prompt_cache_key\n");
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // the task would clear this slot, its state cannot be reused with another lora config
            if (lora_should_clear_cache(slot.prompt->lora, lora)) {
                SLT_TRC(slot, "%s", " - skipping, other lora config\n");
                continue;
            }

            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            bool qualified, is_all;
            classify_prompt(*slot.prompt, lcp_len, qualified, is_all);

            // a still-busy row (DECODE or STOP-awaiting-release) is not launchable yet,
            // but remember it when it is this task's best content match: the caller may
            // wait briefly - a steering close usually lands within milliseconds, and the
            // row releases with warm KV for exactly this task.
            if (slot.is_processing) {
                candidate cb;
                cb.slot      = &slot;
                cb.lcp       = lcp_len;
                cb.qualified = qualified;
                if (consider_better(cb, busy_best)) {
                    busy_best = std::move(cb);
                }
                SLT_TRC(slot, " - decoding, match=%zu [%s] (grace candidate)\n",
                        lcp_len, is_all ? "ALL" : (qualified ? "PART" : "-"));
                continue;
            }

            SLT_TRC(slot, " - checking LCP: prompt=%zu, cached=%zu, match=%zu [%s]\n",
                    task.tokens.size(), tokens.size(), lcp_len, is_all ? "ALL" : (qualified ? "PART" : "-"));

            if (is_all) {
                ret = &slot; // early winner: the conversation is already in live memory
                SLT_INF(slot, "selected slot %d (ALL): prompt=%zu, slot=%zu, match=%zu\n",
                        slot.id, task.tokens.size(), tokens.size(), lcp_len);
                early_winner = true;
                break;
            }

            candidate c;
            c.slot      = &slot;
            c.lcp       = lcp_len;
            c.qualified = qualified;
            if (consider_better(c, best)) {
                best = std::move(c);
                ret  = &slot;
            }
        }
    }

    // scan the prompt cache for a better candidate than any live slot.
    // cache_best always tracks the LONGEST qualified entry seen so far;
    // entry_beats_slots flips ret once some entry out-ranks all slots.
    candidate cache_best;
    bool entry_beats_slots = false;
    if (use_cache && !early_winner && task.id_slot == -1 && key_state == nullptr) {
        size_t idx = 0;
        for (const auto & st : cache->states) {
            if (st.data.main.empty()) {
                ++idx;
                continue; // entry already consumed (moved to a slot)
            }

            // a prompt made with another lora config is cleared before the new one starts, so it counts as empty
            if (lora_should_clear_cache(st.prompt.lora, lora)) {
                ++idx;
                continue;
            }

            // owned state is never adopted by content matching: only its owner's key path does
            if (!st.prompt.cache_key.empty() && st.prompt.cache_key != task.prompt_cache_key) {
                ++idx;
                continue;
            }

            const size_t lcp = st.prompt.tokens.get_common_prefix(task.tokens);
            bool qualified, is_all;
            classify_prompt(st.prompt, lcp, qualified, is_all);

            if (!qualified) {
                ++idx;
                continue;
            }

            candidate c;
            c.entry_idx  = idx;
            c.is_entry   = true;
            c.lcp        = lcp;
            c.qualified  = qualified;
            c.is_all     = is_all;

            SRV_TRC(" - checking cache entry: idx=%zu, cached=%zu, match=%zu [%s]\n",
                    idx, st.prompt.tokens.size(), lcp, is_all ? "ALL" : "PART");

            if (!cache_best.qualified || consider_better(c, cache_best)) {
                cache_best = c; // longest qualified entry so far
                if (consider_better(c, best)) {
                    entry_beats_slots = true;
                }
            } else if (consider_better(cache_best, best)) {
                entry_beats_slots = true;
            }

            ++idx;
        }

        if (entry_beats_slots) {
            ret = nullptr; // winner is a cache entry, not a live slot
        }
    } else if (use_cache && key_state != nullptr) {
        // the owner's key entry wins outright: identity beats content; it is restored
        // even when no longer matching (the agent was reset, its state is reclaimed)
        candidate c;
        c.entry_idx = key_entry;
        c.is_entry  = true;
        c.qualified = true;
        c.is_all    = true; // the owner's state moves to the slot, returning keyed after displacement
        c.lcp       = key_state->prompt.tokens.get_common_prefix(task.tokens);
        cache_best  = c;

        ret = nullptr; // winner is the entry, not a live slot
    }

    server_slot_plan plan;

    // handle the winner and prepare the plan:
    if (ret == nullptr && use_cache && cache_best.qualified) {
        // Winner = a prompt-cache entry: the caller finds a free slot, saves its
        // displaced state first, then extracts and restores this entry into it.
        ret = get_free_slot_infos(slots, 1, -1).empty() ? nullptr : get_free_slot_infos(slots, 1, -1)[0];

        if (ret != nullptr) {
            SLT_INF(*ret, "selected slot %d for cache restore, n_tokens = %d, t_last = %" PRId64 "\n",
                    ret->id, (int) ret->prompt->n_tokens(), ret->t_last_used);

            plan.id_slot     = ret->id;
            plan.update_cache = true;
            plan.entry_index = (int) cache_best.entry_idx;
            plan.is_move     = cache_best.is_all; // ALL: consume the entry, PART: keep it for the owner
            plan.matched_len = (int) cache_best.lcp; // for PCACHE_HIT logging
        }
    } else if (ret != nullptr && !early_winner) {
        // Winner = a live slot with a PARTIAL match (not an ALL continuation):
        // its existing state is displaced, so save it to keep the conversation restorable.
        if (best.slot == ret && best.qualified) {
            SLT_INF(*ret, "selected slot %d (PART): prompt=%zu, slot=%zu, match=%zu\n",
                    ret->id, task.tokens.size(), ret->prompt->tokens.size(), best.lcp);
        }

        plan.id_slot      = ret->id;
        plan.update_cache = use_cache;
    } else if (ret == nullptr) {
        // Nothing matched: take the least recently used idle slot and reuse it fresh,
        // saving its displaced state first so nothing recoverable is lost.
        // empty slots first, then least recently used
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot %d by LRU (no match), n_tokens = %d, t_last = %" PRId64 "\n",
                    ret->id, (int) ret->prompt->n_tokens(), ret->t_last_used);

            plan.id_slot      = ret->id;
            plan.update_cache = use_cache && !ret->prompt->tokens.empty();
        }
    } else {
        // early winner: an ALL match or a pinned slot, the state continues in place
        plan.id_slot = ret->id;
    }

    // Grace-wait decision (the decode-loop caller owns the 100ms deadline): park on
    // a still-decoding row when it is strictly better than what we were about to take.
    // The steering close for such a row usually arrives within milliseconds of this
    // request; waiting 100ms then re-selecting costs little and saves the re-prefill.
    // Early winners never wait: their slot is attachable right now.
    if (!early_winner && plan.id_slot != -1 && busy_best.qualified) {
        size_t winner_lcp = 0; // forced-LRU/empty fallback: nothing reusable yet
        if (ret != nullptr && best.slot == ret) {
            winner_lcp = best.lcp; // PART live match
        } else if (ret == nullptr && cache_best.qualified) {
            winner_lcp = cache_best.lcp; // prompt-cache entry won the scan
        }

        if (busy_best.lcp > winner_lcp) {
            SLT_INF(*busy_best.slot, "row %d as grace wait: match=%zu beats winner lcp=%zu - defer task until its STOP\n",
                    busy_best.slot->id, busy_best.lcp, winner_lcp);
            plan.wait_row = busy_best.slot->id;
        }
    }

    return plan;
}
