// Standalone test of the strict select() rules (2026-11-28, concepts/
// prefix-sharing-server.txt): a live row is only attachable via its
// prompt_cache_key or an ALL match; a PARTIAL match never takes over.
// Run via ctest or directly.

#include "server-slot-sched.h"
#include "server-task.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

struct check {
    int failed = 0;
    void operator()(bool cond, const char * what) {
        if (!cond) {
            std::cerr << "FAILED: " << what << std::endl;
            ++failed;
        }
    }
} chk;

// the token list [first .. first + n)
static std::vector<llama_token> range(int64_t first, size_t n) {
    std::vector<llama_token> t(n);
    for (size_t i = 0; i < n; ++i) {
        t[i] = static_cast<llama_token>(first + i);
    }
    return t;
}

// the incident scenario shape: shared head of 40, foreign tail of 60
static std::vector<llama_token> foreign_prompt() {
    auto p = range(1, 40);
    auto q = range(5000, 60);
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

static server_task make_task(std::vector<llama_token> prompt_tokens, const std::string & key = "") {
    server_task t(SERVER_TASK_TYPE_COMPLETION);
    t.tokens = server_tokens(llama_tokens(prompt_tokens), false);
    t.prompt_cache_key = key;
    t.params.cache_prompt = true;
    return t;
}

static std::shared_ptr<common_prompt_checkpoint> make_ckpt(int64_t n_tokens) {
    auto c  = std::make_shared<common_prompt_checkpoint>();
    c->n_tokens = n_tokens;
    c->pos_min  = 0;
    c->pos_max  = n_tokens - 1;
    c->data_tgt.push_back(0); // non-empty: saved state present (qualifies for sharing)
    return c;
}

// the incident shape: 160-token warm row (ckpts at 40..160), a foreign task
// sharing only its first 40 tokens, plus one empty slot to fall back on.
struct fixture {
    server_prompt              warm_prompt;
    server_slot_info           warm;     // idle, keyless
    server_slot_info           empty;    // idle, keyless
    std::vector<server_slot_info> slots;
    server_prompt_cache        cache{0, 0};

    fixture() {
        warm_prompt.tokens = server_tokens(llama_tokens(range(1, 160)), false);
        for (int64_t n : {40, 80, 120, 160}) {
            warm_prompt.checkpoints.push_back(make_ckpt(n));
        }

        warm.id      = 0;
        warm.is_processing = false;
        warm.t_last_used   = 1000;
        warm.prompt    = &warm_prompt;

        static server_prompt empty_prompt; // an empty slot still has a prompt object
        empty.id      = 1;
        empty.is_processing = false;
        empty.t_last_used   = 900;
        empty.prompt    = &empty_prompt;

        slots = { warm, empty };
    }
};

// S1: the incident case - PARTIAL never steals the warm row
static void test_partial_no_steal() {
    fixture fx;

    auto plan = server_slot_sched().select(fx.slots, make_task(foreign_prompt()), {}, &fx.cache);

    chk(plan.id_slot == 1, "S1 partial takes the empty slot, not the warm row");
    chk(plan.update_cache == false, "S1 no displacement save of an empty slot");
    chk(plan.entry_index == -1, "S1 no cache entry involved");
}

// S2: an ALL match still attaches (continuation)
static void test_all_attaches() {
    fixture fx;

    auto plan = server_slot_sched().select(fx.slots, make_task(range(1, 160)), {}, &fx.cache);

    chk(plan.id_slot == 0, "S2 all match attaches to the warm row");
}

// S3: the key always wins, regardless of content match
static void test_key_wins() {
    fixture fx;
    fx.warm_prompt.cache_key = "K-1"; // warm.prompt points into warm_prompt

    auto plan = server_slot_sched().select(fx.slots, make_task(range(90000, 8), "K-1"), {}, &fx.cache);

    chk(plan.id_slot == 0, "S3 same key attaches to its row regardless of content");
}

// S4: a busy row with only a PARTIAL match is not a grace-parking target
static void test_partial_no_parking() {
    fixture fx;
    fx.slots[0].is_processing = true; // the vector copies hold what select() sees

    auto plan = server_slot_sched().select(fx.slots, make_task(foreign_prompt()), {}, &fx.cache);

    chk(plan.id_slot == 1, "S4 busy partial row skipped, empty slot taken");
    chk(plan.wait_row == -1, "S4 no grace parking on a partial busy match");
}

// S5: a busy row with an ALL match still parks (steering handover)
static void test_all_parks() {
    fixture fx;
    fx.slots[0].is_processing = true; // the vector copies hold what select() sees

    auto plan = server_slot_sched().select(fx.slots, make_task(range(1, 160)), {}, &fx.cache);

    chk(plan.id_slot == 1, "S5 fallback slot planned while waiting");
    chk(plan.wait_row == 0, "S5 all match on busy row parks (steering)");
}

// S6: regression - a PARTIAL prompt-cache ENTRY still restores (COPY keeps it)
static void test_entry_partial_still_restores() {
    fixture fx;

    server_prompt_cache_state st;
    st.prompt.tokens = server_tokens(llama_tokens(range(1, 160)), false);
    for (int64_t n : {40, 80, 120, 160}) {
        st.prompt.checkpoints.push_back(make_ckpt(n));
    }
    st.data.main.push_back(0); // non-empty: a live entry
    fx.cache.states.push_back(std::move(st));

    auto plan = server_slot_sched().select(fx.slots, make_task(foreign_prompt()), {}, &fx.cache);

    chk(plan.id_slot == 1, "S6 entry restore lands on the empty slot");
    chk(plan.entry_index == 0, "S6 the partial cache entry still wins");
    chk(plan.is_move == false, "S6 partial entry match is a COPY (owner stays restorable)");
}

// S7: a fresh conversation on the LRU fallback shares the warm row's best prefix checkpoint
static void test_lru_fallback_shares_prefix() {
    fixture fx;

    auto plan = server_slot_sched().select(fx.slots, make_task(foreign_prompt()), {}, &fx.cache);

    chk(plan.id_slot == 1, "S7 partial takes the empty slot, not the warm row");
    chk(plan.share_from == 0, "S7 shares from the warm row");
    chk(plan.share_ckpt != nullptr && plan.share_ckpt->n_tokens == 40, "S7 shared range = best checkpoint within the prefix (40)");
}

// S8: a prompt-cache entry win suppresses sharing - the entry restore carries its own state
static void test_entry_win_no_share() {
    fixture fx;

    server_prompt_cache_state st;
    st.prompt.tokens = server_tokens(llama_tokens(range(1, 160)), false);
    for (int64_t n : {40, 80, 120, 160}) {
        st.prompt.checkpoints.push_back(make_ckpt(n));
    }
    st.data.main.push_back(0); // non-empty: a live entry
    fx.cache.states.push_back(std::move(st));

    auto plan = server_slot_sched().select(fx.slots, make_task(foreign_prompt()), {}, &fx.cache);

    chk(plan.entry_index == 0, "S8 the cache entry still wins");
    chk(plan.share_ckpt == nullptr, "S8 no sharing when an entry restore wins");
    chk(plan.share_from == -1, "S8 share source cleared");
}

// S9: an ALL match attaches - no sharing (the state is already there)
static void test_all_match_no_share() {
    fixture fx;

    auto plan = server_slot_sched().select(fx.slots, make_task(range(1, 160)), {}, &fx.cache);

    chk(plan.id_slot == 0, "S9 all match attaches to the warm row");
    chk(plan.share_ckpt == nullptr, "S9 no sharing for an attach");
}

// S10: the LRU victim is itself the share source - no self-restore (its KV is already in place)
static void test_no_self_share() {
    fixture fx;
    fx.slots.resize(1); // only the warm row exists: it is both the prefix holder and the LRU victim

    auto plan = server_slot_sched().select(fx.slots, make_task(foreign_prompt()), {}, &fx.cache);

    chk(plan.id_slot == 0, "S10 the partial row is taken by forced LRU");
    chk(plan.share_ckpt == nullptr, "S10 no self-share when victim and source are the same row");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << "test-select-rules: starting\n";

    test_partial_no_steal();
    test_all_attaches();
    test_key_wins();
    test_partial_no_parking();
    test_all_parks();
    test_entry_partial_still_restores();
    test_lru_fallback_shares_prefix();
    test_entry_win_no_share();
    test_all_match_no_share();
    test_no_self_share();

    if (chk.failed == 0) {
        std::cout << "test-select-rules: all OK\n";
        return 0;
    }

    std::cerr << "test-select-rules: " << chk.failed << " check(s) failed\n";
    return EXIT_FAILURE;
}
