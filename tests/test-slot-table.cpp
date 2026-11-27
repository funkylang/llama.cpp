#include "server-slot-table.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

// Standalone coordination test for server_slot_table (no llama dependency:
// rows run with nullptr slot pointers). Run via ctest or directly.

static const int N_ROWS = 4;

struct check {
    int failed = 0;
    void operator()(bool cond, const char * what) {
        if (!cond) {
            std::cerr << "FAILED: " << what << std::endl;
            ++failed;
        }
    }
} chk;

// The worker protocol: loop claim -> wait until the row is ours (or timeout).
// This is the exact shape the Phase 2 HTTP handler wiring will take.
static bool acquire_row(server_slot_table & tbl, const std::string & key, int task_id, int & row, int64_t timeout_ms) {
    while (true) {
        auto res = tbl.claim_or_wait(key, task_id, &row);
        if (res == server_slot_table::CLAIM_OK) {
            return true;
        }
        if (!tbl.wait_release(key, res == server_slot_table::WAIT_OTHER, timeout_ms)) {
            return false; // the 503 path
        }
    }
}

// Scenario 1: basic lifecycle - claim, stop, is_stopped, finish, reservation kept.
static void test_lifecycle() {
    server_slot_table tbl(N_ROWS);

    int row_id = -1;
    auto res = tbl.claim_or_wait("key-1", 100, &row_id);
    chk(res == server_slot_table::CLAIM_OK, "S1 claim ok");
    chk(row_id >= 0 && row_id < N_ROWS, "S1 row id valid");
    chk(tbl.is_stopped(row_id) == false, "S1 not stopped before close");

    tbl.request_stop(100);
    chk(tbl.is_stopped(row_id) == true, "S1 stopped after close signal");

    tbl.finish_or_stop(row_id);
    chk(tbl.is_stopped(row_id) == false, "S1 no longer stopped");

    // Same key can now take over the reserved row immediately.
    int row2 = -1;
    res = tbl.claim_or_wait("key-1", 101, &row2);
    chk(res == server_slot_table::CLAIM_OK, "S1 takeover claim ok");
    chk(row2 == row_id, "S1 takeover on the same row");

    tbl.request_stop(101);
    tbl.finish_or_stop(row2);
}

// Scenario 2: the 2026-11-26 race - same key, busy holder, close then reclaim.
static void test_same_key_race() {
    server_slot_table tbl(N_ROWS);

    // Holder with key X is running on some row.
    int holder_row = -1;
    chk(tbl.claim_or_wait("key-X", 200, &holder_row) == server_slot_table::CLAIM_OK, "S2 holder claim");

    // New request for the same key must queue on that exact row, not steal others.
    int w_row = -1;
    auto res = tbl.claim_or_wait("key-X", 201, &w_row);
    chk(res == server_slot_table::WAIT_KEY, "S2 same-key waiter waits");
    chk(w_row == holder_row, "S2 waiter queued on the busy row");

    std::thread waiter([&] {
        chk(acquire_row(tbl, "key-X", 201, w_row, 5000), "S2 waiter re-claim ok");
        chk(w_row == holder_row, "S2 waiter got the warm row");

        tbl.request_stop(201);
        tbl.finish_or_stop(w_row);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // let the waiter park

    // Old connection closes; decode loop finishes the cycle and releases.
    tbl.request_stop(200);
    chk(tbl.is_stopped(holder_row) == true, "S2 holder stopped");
    tbl.finish_or_stop(holder_row);

    waiter.join();
}

// Scenario 3: reservation protection - unkeyed requests never steal reserved rows.
static void test_reservation_protection() {
    server_slot_table tbl(N_ROWS);

    // Fill every row with a reservation (claimed then finished).
    for (int i = 0; i < N_ROWS; i++) {
        int r = -1;
        chk(tbl.claim_or_wait("key-" + std::to_string(i), 300 + i, &r) == server_slot_table::CLAIM_OK, "S3 fill claim");
        tbl.request_stop(300 + i);
        tbl.finish_or_stop(r);
    }

    // Unkeyed request: nothing claimable -> WAIT_OTHER (no stealing).
    int r = -1;
    chk(tbl.claim_or_wait("", 400, &r) == server_slot_table::WAIT_OTHER, "S3 unkeyed waits");

    // Keyed request for a key not in the table: may take a reserved row.
    chk(tbl.claim_or_wait("key-other", 401, &r) == server_slot_table::CLAIM_OK, "S3 keyed fallback claim");
    chk(r >= 0 && r < N_ROWS, "S3 fallback row valid");

    tbl.finish_or_stop(r);
    // Still all remaining rows reserved -> unkeyed waits again.
    chk(tbl.claim_or_wait("", 402, &r) == server_slot_table::WAIT_OTHER, "S3 unkeyed still waits");
}

// Scenario 4: wait timeout and wake behavior.
static void test_wait_timeout() {
    server_slot_table tbl(N_ROWS);

    // Fill all rows so WAIT_OTHER is the outcome.
    std::vector<int> rows_claimed;
    for (int i = 0; i < N_ROWS; i++) {
        int r = -1;
        chk(tbl.claim_or_wait("", 500 + i, &r) == server_slot_table::CLAIM_OK, "S4 fill claim");
        rows_claimed.push_back(r);
    }

    // WAIT_OTHER with a short timeout must return false while all rows are busy.
    chk(tbl.wait_release("", true, 200) == false, "S4 wait times out");

    // Releasing one row wakes an any-waiter quickly.
    auto t0 = std::chrono::steady_clock::now();
    std::thread waiter([&] {
        chk(tbl.wait_release("", true, 5000) == true, "S4 any-waiter woke");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    tbl.finish_or_stop(rows_claimed[0]);
    waiter.join();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    chk(ms < 1000, "S4 wake is fast");

    // Release the rest (rows_claimed[0] was already finished above).
    for (size_t i = 1; i < rows_claimed.size(); i++) { tbl.finish_or_stop(rows_claimed[i]); }
}

// Scenario 5: concurrent stress - connection threads claim/wait/stop/finish in
// random order; one decoder thread services STOP rows. Invariants checked at
// the end and throughout via the table's own accounting.
static void test_stress() {
    server_slot_table tbl(N_ROWS);
    const int N_CONNS = 8;
    const int N_REQUESTS_PER_CONN = 20;

    std::atomic<bool> running{true};
    std::atomic<int> n_claims{0}, n_timeouts{0}, n_stops_issued{0};

    auto connection = [&](int conn_id) {
        // Each connection owns a stable key, like an agent conversation.
        const std::string key = "conn-" + std::to_string(conn_id);

        for (int req = 0; req < N_REQUESTS_PER_CONN && running.load(); req++) {
            int task_id = conn_id * 1000 + req;

            // Claim loop using the worker protocol (the res enum is authoritative;
            // *out_id on WAIT_KEY names the wait target, NOT an owned row).
            bool acquired = false;
            int row = -1;
            while (running.load() && !acquired) {
                auto res = tbl.claim_or_wait(key, task_id, &row);
                if (res == server_slot_table::CLAIM_OK) {
                    acquired = true;
                } else if (!tbl.wait_release(key, res == server_slot_table::WAIT_OTHER, 300)) {
                    n_timeouts++;
                    row = -1; // the 503 path: we own nothing
                } else {
                    row = -1; // woken but not yet claimed
                }
            }

            if (!acquired || !running.load()) {
                continue;
            }

            n_claims++;

            // Simulate work; randomly close (STOP) or finish normally.
            std::this_thread::sleep_for(std::chrono::milliseconds(1 + (req % 3)));

            if ((task_id % 4) == 0) {
                tbl.request_stop(task_id);     // connection closed mid-task
                n_stops_issued++;
                // The decoder thread will see is_stopped and finish_or_stop.
            } else {
                tbl.finish_or_stop(row);       // normal completion
            }
        }
    };

    std::vector<std::thread> conns;
    for (int i = 0; i < N_CONNS; i++) {
        conns.emplace_back(connection, i);
    }

    // Decoder thread: service STOP rows exactly as the real loop would.
    std::thread decoder([&] {
        while (true) {
            for (int r = 0; r < N_ROWS; r++) {
                if (tbl.is_stopped(r)) {
                    tbl.finish_or_stop(r);
                }
            }

            if (!running.load()) {
                // Drain remaining stopped rows before exiting.
                for (int r = 0; r < N_ROWS; r++) {
                    if (tbl.is_stopped(r)) { tbl.finish_or_stop(r); }
                }
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    for (auto & t : conns) { t.join(); }
    running.store(false);
    decoder.join();

    // Invariants:
    // - every claim either finished or was stopped+finished; table must be quiescent:
    chk(tbl.slots_used() == 0, "S5 all rows released at end");
    chk(tbl.slots_total() == (size_t) N_ROWS, "S5 row count stable");
    chk(n_claims.load() > 0, "S5 some work got done");
    std::cout << "  S5: claims=" << n_claims.load()
              << " timeouts=" << n_timeouts.load()
              << " stops_issued=" << n_stops_issued.load() << std::endl;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << "test-slot-table: starting\n";

    test_lifecycle();
    test_same_key_race();
    test_reservation_protection();
    test_wait_timeout();
    test_stress();

    if (chk.failed == 0) {
        std::cout << "test-slot-table: all OK" << std::endl;
        return 0;
    }

    std::cerr << "test-slot-table: " << chk.failed << " check(s) failed" << std::endl;
    return EXIT_FAILURE;
}
