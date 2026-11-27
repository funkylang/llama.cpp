#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

// The slot table: single source of truth for SLOT COORDINATION shared by the
// HTTP connection threads (request admission, close detection) and the decode
// scheduler loop. Replaces the scattered deques + select()/plan tickets as the
// tasks' coordination path (concepts/slot-table-impl-plan.txt).
//
// Rows are lightweight; each points at the heavyweight per-slot runtime object
// (server_slot: KV sequence, prompt, generation state) owned by server_context.
// The decode loop keeps talking to server_slot directly.
//
// THREADING CONTRACT:
//   - All row mutations happen inside table methods, under one mutex.
//     Callers never touch rows directly.
//   - The mutex is held for bookkeeping only. llama_process(), prompt-cache
//     saves and the rank callback must NEVER run while holding it.
//   - No new threads: existing HTTP workers + the decode loop are the only
//     parties; condition variables wake waiters on row release.

struct server_slot; // defined in server-context.cpp; only referenced by pointer here

// Ranks candidate idle rows for an incoming request; called OUTSIDE the table
// lock, from inside claim_or_wait(). Returns a score, higher wins; the table
// breaks ties by lowest row id. Prompt access comes via captured state.
// Must not re-enter the table.
using server_slot_rank_t = std::function<int(int id_row)>;

struct server_slot_table {
private:
    // Row status. IDLE keeps its prompt_cache_key as a RESERVATION so a keyed
    // conversation can reclaim its warm KV state instead of re-prefilling cold.
    enum status_t : uint8_t {
        ROW_IDLE   = 0, // no active task; prompt_cache_key may hold a reservation
        ROW_DECODE = 1, // an active task runs on this row
        ROW_STOP   = 2, // close detected for the active task; the decode loop
                        // honors it by finishing its current cycle, then takes
                        // the row back to IDLE preserving KV under the key
    };

    struct row {
        status_t      status = ROW_IDLE;
        std::string   prompt_cache_key; // empty when unkeyed; kept when IDLE (reservation)
        int           task_id  = -1;    // owner of the row while DECODE/STOP; join key for close -> STOP
        server_slot * slot     = nullptr;

        bool is_busy() const { return status != ROW_IDLE; }
        bool has_key() const { return !prompt_cache_key.empty(); }
    };

    std::vector<row> rows_;
    mutable std::mutex mutex_;
    std::condition_variable cv_release_; // signals "a row released"; waiters
                                         // re-check their predicate themselves

    server_slot_rank_t rank_idle_;

    // KV pool accounting (unified mode): written only by the decode loop
    // after each cycle; readers (load query) take them without locking.
    std::atomic<size_t> cells_used_{0};
    std::atomic<size_t> cells_total_{0};

private:
    // Helpers called with mutex_ already held, except where noted.
    // Returns -1 if none found.
    int find_idle_for_key(const std::string & key);   // rule 1: reserved idle with same key
    int find_busy_with_key(const std::string & key);  // rule 2: busy row with same key

    // rule 3: pick best idle row by rank; skips reserved rows for unkeyed requests
    // unless there is no other choice (then takes any)
    int find_best_idle(int requested_row, const std::string & key);

    row & row_ref(int id_row);

public:
    explicit server_slot_table(size_t n_slots);
    ~server_slot_table() = default;
    server_slot_table(const server_slot_table &)            = delete;
    server_slot_table & operator=(const server_slot_table &) = delete;

    size_t n_rows() const;

    // Install the idle-row ranking callback before traffic starts. Used for
    // LCP-based selection among candidate idle rows (see claim_or_wait).
    void set_rank_idle(server_slot_rank_t rank);

    // Bind slot ptr to its row (server_context_init); called once per row.
    void bind_row(int id_row, server_slot * slot);

    // ------------------------------------------------------------------
    // request admission (HTTP worker threads)
    // ------------------------------------------------------------------

    // Attempt to claim a row for an incoming task; runs on the connection thread.
    // Selection rules (in order):
    //   1. key given: an IDLE row reserved with that key is taken immediately
    //      (the old holder is stale by definition).
    //   2. key given and a busy row owns it: queue on that row (WAIT_KEY); the
    //      same-key takeover arrives as CLOSE/STOP from the old connection.
    //   3. otherwise an idle row is selected by rank_idle_ — unkeyed requests
    //      never steal reserved rows while a non-reserved one exists.
    //   4. else: WAIT_OTHER (all busy, no key match).
    //
    // Outcomes:
    //   CLAIM_OK     : *out_id is BOUND (DECODE, task_id set); caller launches the task on it.
    //   WAIT_KEY     : *out_id names the busy same-key row we queue on - a wait
    //                  TARGET, not an owned row. The caller MUST only act on
    //                  rows from CLAIM_OK results.
    //   WAIT_OTHER   : *out_id = -1.
    enum claim_t : uint8_t {
        CLAIM_OK     = 0,
        WAIT_KEY     = 1,
        WAIT_OTHER   = 2,
    };

    claim_t claim_or_wait(const std::string & key, int task_id, int * out_id);

    // Wait for a row release after a WAIT_* outcome; HTTP workers only.
    // Returns true when woken (re-run claim_or_wait), false on timeout.
    // key: the request's key ("" for id-less); any=true for WAIT_OTHER waiters.
    bool wait_release(const std::string & key, bool any, int64_t timeout_ms);

    // ------------------------------------------------------------------
    // close / stop (HTTP worker that owns the connection)
    // ------------------------------------------------------------------

    // The connection carrying task_id closed: mark its row STOP so the decode
    // loop releases it after the current cycle. No-op when the task already
    // finished. Child tasks (n_cmpl batches, realtime) must all be covered by
    // their parent's close: pass the whole family.
    void request_stop(int task_id);

    // ------------------------------------------------------------------
    // decode scheduler (main loop only)
    // ------------------------------------------------------------------

    // Called once per active slot in each decode cycle, with the slot's row id:
    // true when that row was marked STOP by a connection close. The decode loop
    // finishes its current llama_process() cycle for it, then calls
    // finish_or_stop(). Never cuts an in-flight llama_process() short.
    bool is_stopped(int id_row);

    // A decode cycle on this row is done; its task finished or was stopped.
    // Transitions the row back to IDLE preserving KV under its key
    // (reservation) and wakes matching waiters. The server_slot release path
    // (prompt save etc.) still runs in the decode loop as today.
    void finish_or_stop(int id_row);

    // ------------------------------------------------------------------
    // load query (read-only; for The Director)
    // ------------------------------------------------------------------

    size_t slots_used() const;   // rows not IDLE
    size_t slots_total() const;

    // KV cells in use / pool size, unified mode. Updated by the decode loop:
    // after each cycle it reports the sum of sequence lengths (main + draft).
    size_t cells_used() const;
    size_t cells_total() const;
    void   set_pool_size(size_t n_cells);
    void   report_cells_used(size_t n_cells);

    // ------------------------------------------------------------------
    // eviction (defensive; Phase 3)
    // ------------------------------------------------------------------

    // A RUNNING task cannot allocate more KV: stop the running row with the
    // fewest tokens, save it to the prompt cache, free its cells. Newcomers
    // never trigger this; only a starved incumbent does. Evicted rows hold
    // absolute priority on reschedule. Returns -1 when nothing is evictable.
    int evict_smallest_running(size_t n_cells_needed);

    // ------------------------------------------------------------------
    // shadow observation (Phase 0)
    // ------------------------------------------------------------------

    // Mirror the transitions the real system performs, without taking effect
    // on scheduling: same row state changes under the mutex, but divergences
    // (e.g. launching on a busy row, releasing an idle row) are LOGGED, not
    // fatal. Verifies the table model matches reality before Phase 1 routes
    // live traffic through the real API.
    void shadow_launch(int id_row, int task_id, const std::string & key);
    void shadow_release(int id_row);
    void shadow_stop(int task_id);
};
