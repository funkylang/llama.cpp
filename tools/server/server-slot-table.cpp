#include "server-slot-table.h"

#include "server-common.h" // SRV_INF/SRV_DBG/SRV_WRN
#include "common.h"        // GGML_ASSERT/GGML_ABORT, ggml_time_us

server_slot_table::server_slot_table(size_t n_slots) : rows_(n_slots) {
    GGML_ASSERT(rows_.size() > 0);
}

size_t server_slot_table::n_rows() const {
    return rows_.size();
}

void server_slot_table::set_rank_idle(server_slot_rank_t rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    rank_idle_ = std::move(rank);
}

void server_slot_table::bind_row(int id_row, server_slot * slot) {
    // Called once per row during server context init, before traffic starts.
    GGML_ASSERT((size_t)id_row < rows_.size());
    std::lock_guard<std::mutex> lock(mutex_);
    if (rows_[id_row].slot != nullptr && rows_[id_row].slot != slot) {
        GGML_ABORT("server_slot_table::bind_row(%d): row already bound to a different slot", id_row);
    }
    rows_[id_row].slot = slot;
}

// All find_* helpers run with mutex_ held. They return -1 when nothing matches.
int server_slot_table::find_idle_for_key(const std::string & key) {
    for (size_t i = 0; i < rows_.size(); i++) {
        if (!rows_[i].is_busy() && rows_[i].prompt_cache_key == key) {
            return (int) i;
        }
    }
    return -1;
}

int server_slot_table::find_busy_with_key(const std::string & key) {
    for (size_t i = 0; i < rows_.size(); i++) {
        if (rows_[i].is_busy() && rows_[i].prompt_cache_key == key) {
            return (int) i;
        }
    }
    return -1;
}

int server_slot_table::find_best_idle(int requested_row, const std::string & key) {
    // The rank callback scores idle rows for the incoming request (higher wins;
    // it must not re-enter the table). Called while the mutex is HELD in Phase 0
    // and 1 where selection is simple; later phases move the ranking outside
    // the lock and re-validate at commit time.
    auto rank = [this](int id_row) -> int {
        return rank_idle_ ? rank_idle_(id_row) : 0;
    };

    int best_unkeyed = -1, best_unkeyed_score = -1;
    int best_any     = -1, best_any_score     = -1;

    for (size_t i = 0; i < rows_.size(); i++) {
        if (rows_[i].is_busy()) {
            continue;
        }

        const bool keyed_row  = rows_[i].has_key();
        const int  score      = rank((int) i);

        if (!keyed_row && (best_unkeyed < 0 || score > best_unkeyed_score)) {
            best_unkeyed     = (int) i;
            best_unkeyed_score = score;
        }

        // Keyed requests may fall back to reserved rows of OTHER keys when no
        // unkeyed idle row exists; unkeyed requests never steal reservations.
        if (!key.empty() && (best_any < 0 || score > best_any_score)) {
            best_any     = (int) i;
            best_any_score = score;
        }
    }

    // Explicit pinning: a caller may request a specific idle row.
    if (requested_row >= 0 && requested_row < (int) rows_.size() && !rows_[requested_row].is_busy()) {
        return requested_row;
    }

    // Unkeyed rows first (reservation protection), then reserved for keyed reqs.
    if (best_unkeyed >= 0) {
        return best_unkeyed;
    }
    if (!key.empty() && best_any >= 0) {
        return best_any;
    }
    return -1;
}

server_slot_table::claim_t server_slot_table::claim_or_wait(const std::string & key, int task_id, int * out_id) {
    GGML_ASSERT(task_id >= 0);
    GGML_ASSERT(out_id != nullptr);

    *out_id = -1;

    std::lock_guard<std::mutex> lock(mutex_);

    if (!key.empty()) {
        // Rule 1: same-key takeover of a reserved idle row.
        int idle_same_key = find_idle_for_key(key);
        if (idle_same_key >= 0) {
            rows_[idle_same_key].status  = ROW_DECODE;
            rows_[idle_same_key].task_id = task_id;
            *out_id                      = idle_same_key;
            SRV_INF("slot_table: claim row %d for task %d (same-key takeover)\n", idle_same_key, task_id);
            return CLAIM_OK;
        }

        // Rule 2: same-key busy row - queue on it; its close takes over.
        int busy_same_key = find_busy_with_key(key);
        if (busy_same_key >= 0) {
            *out_id = busy_same_key;
            SRV_INF("slot_table: wait on busy row %d for task %d (same-key)\n", busy_same_key, task_id);
            return WAIT_KEY;
        }
    }

    // Rule 3: best idle row by rank (reservation protection built in).
    int best = find_best_idle(-1, key);
    if (best < 0) {
        SRV_DBG("slot_table: no claimable row for task %d - WAIT_OTHER\n", task_id);
        return WAIT_OTHER;
    }

    rows_[best].status           = ROW_DECODE;
    rows_[best].task_id          = task_id;
    rows_[best].prompt_cache_key = key; // reservation follows the claim ("" clears a stolen one)
    *out_id                      = best;
    SRV_INF("slot_table: claim row %d for task %d%s\n", best, task_id, key.empty() ? "" : " (keyed)");
    return CLAIM_OK;
}

bool server_slot_table::wait_release(const std::string & key, bool any, int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wake conditions:
    //  - any-waiters (WAIT_OTHER): any idle row at all
    //  - key-waiters (WAIT_KEY): an idle row reserved with this exact key,
    //    i.e. the same-key holder released and its reservation is takeable
    const auto matched = [this, &key, any] {
        for (const auto & r : rows_) {
            if (!r.is_busy() && (any || r.prompt_cache_key == key)) {
                return true;
            }
        }
        return false;
    };

    bool woken;
    if (timeout_ms > 0) {
        woken = cv_release_.wait_until(lock,
                                       std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms),
                                       matched);
        if (!woken) {
            // Timeout: one last race check in case a release landed just before.
            woken = matched();
        }
    } else {
        cv_release_.wait(lock, matched);
        woken = true;
    }

    if (woken) {
        SRV_DBG("slot_table: wait_release woken (%s) - retry claim\n", key.empty() ? "any" : key.substr(0, 8).c_str());
    } else {
        SRV_WRN("slot_table: wait_release timed out after %lld ms (key %.8s)\n", (long long) timeout_ms, key.c_str());
    }
    return woken;
}

bool server_slot_table::request_stop(int task_id) {
    GGML_ASSERT(task_id >= 0);
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < rows_.size(); i++) {
        if (rows_[i].task_id == task_id && rows_[i].status != ROW_IDLE) {
            rows_[i].status = ROW_STOP;
            SRV_INF("slot_table: row %zu -> STOP (task %d close signal)\n", i, task_id);
            return true;
        }
    }

    // Task already finished or never bound - nothing to stop.
    SRV_DBG("slot_table: request_stop(%d): no matching row\n", task_id);
    return false;
}

bool server_slot_table::reserve_waiter(int id_row, int task_id) {
    GGML_ASSERT((size_t)id_row < rows_.size());
    std::lock_guard<std::mutex> lock(mutex_);
    auto & r = rows_[id_row];
    if (r.waiter_id >= 0) {
        SRV_WRN("slot_table: row %d already has waiter %d - rejecting duplicate wait (task %d)\n",
                id_row, r.waiter_id, task_id);
        return false;
    }
    r.waiter_id = task_id;
    SRV_INF("slot_table: row %d waiter set to task %d (grace wait)\n", id_row, task_id);
    return true;
}

void server_slot_table::release_waiter(int id_row, int task_id) {
    GGML_ASSERT((size_t)id_row < rows_.size());
    std::lock_guard<std::mutex> lock(mutex_);
    if (rows_[id_row].waiter_id == task_id) {
        SRV_DBG("slot_table: row %d waiter released (task %d)\n", id_row, task_id);
        rows_[id_row].waiter_id = -1;
    }
}

void server_slot_table::begin_decode(int id_row, int task_id, const std::string & key) {
    GGML_ASSERT((size_t)id_row < rows_.size());
    GGML_ASSERT(task_id >= 0);
    std::lock_guard<std::mutex> lock(mutex_);
    auto & r = rows_[id_row];
    if (r.status != ROW_IDLE) {
        // a launch on a non-IDLE row skips the release sweep's teardown of the
        // previous task - that is a scheduling bug we want to see, not crash on
        SRV_WRN("slot_table: DIVERGENCE begin_decode(row %d, task %d): expected IDLE, got status=%d\n",
                id_row, task_id, (int) r.status);
    }
    r.status           = ROW_DECODE;
    r.task_id          = task_id;
    r.prompt_cache_key = key; // reservation follows the launch ("" for unkeyed)
}

bool server_slot_table::is_stopped(int id_row) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT((size_t)id_row < rows_.size());
    return rows_[id_row].status == ROW_STOP;
}

void server_slot_table::finish_or_stop(int id_row) {
    GGML_ASSERT((size_t)id_row < rows_.size());
    std::lock_guard<std::mutex> lock(mutex_);

    auto & r = rows_[id_row];
    if (r.status == ROW_IDLE) {
        // Only a live transition may return a row to IDLE; double release or a
        // release without a launch is a caller bug we want to see, not crash on.
        SRV_WRN("slot_table: DIVERGENCE finish_or_stop(row %d): row already idle\n", id_row);
        return;
    }

    r.task_id  = -1;
    r.status   = ROW_IDLE; // prompt_cache_key kept: reservation
    // the parked grace waiter (if any) is woken by pop_deferred_task on
    // release and re-selects; its seat here is stale either way
    r.waiter_id = -1;
    SRV_DBG("slot_table: row %d -> IDLE%s\n", id_row, r.has_key() ? " (reservation kept)" : "");

    cv_release_.notify_all();
}

size_t server_slot_table::slots_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::count_if(rows_.begin(), rows_.end(), [](const row & r) { return r.is_busy(); });
}

size_t server_slot_table::slots_total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rows_.size();
}

size_t server_slot_table::cells_used() const {
    return cells_used_.load(std::memory_order_acquire);
}

size_t server_slot_table::cells_total() const {
    return cells_total_.load(std::memory_order_acquire);
}

void server_slot_table::set_pool_size(size_t n_cells) {
    cells_total_.store(n_cells, std::memory_order_release);
}

void server_slot_table::report_cells_used(size_t n_cells) {
    // The decode loop is the sole writer; a stale report would overstate
    // occupancy only transiently. Plain store is correct here.
    cells_used_.store(n_cells, std::memory_order_release);
}

int server_slot_table::evict_smallest_running(size_t n_cells_needed) {
    // Phase 3: defensive eviction when a RUNNING task cannot allocate more KV.
    // Not yet wired; returns -1 (nothing evicted).
    GGML_ASSERT(n_cells_needed > 0);
    (void)n_cells_needed;
    SRV_WRN("slot_table: evict_smallest_running(%zu) not implemented (Phase 3)\n", n_cells_needed);
    return -1;
}
