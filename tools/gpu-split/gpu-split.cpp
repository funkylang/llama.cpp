// [FORK] gpu-split — GPU device scan + device placement & KV pool planner
//
// The tool ALWAYS measures (Michael, 2026-11-13): it is an offline planning
// tool run when deployment is being set up, so paying for a real model load is
// correct. It never does a "fake" no_alloc probe load:
//   1. scan devices → raw free VRAM (nothing loaded yet)
//   2. REAL model load with the DEFAULT split — llama_model's native
//      auto-split-by-free-VRAM places layers, exactly as production does
//   3. small context + warmup decode over the real weights → every kernel the
//      deployment will launch is uploaded (kernels only materialize on first
//      execution — concept §17)
//   4. scan devices again → per-GPU delta = the kernel/shader overhead of a
//      live deployment
// The GGUF/hparams table code below is carried over from llama-gguf-info so
// both tools share the same accounting logic; the model loaded for measurement
// supplies its hparams too (single load, no fake probe).
//
// The full model table (per-layer tensor sizes, KV cell bytes, mtp/swa/ssm
// columns, TOTAL) is computed exactly as in llama-gguf-info:
//   - what must be DISTRIBUTED TO THE GPUS — layers, norm/output globals,
//     and any other deployed non-blk tensors; the input embedding always
//     stays on CPU (shown separately as embed(CPU)) EXCEPT tied models,
//     where its lm_head copy joins the table as the last row
//   - KV cell bytes per layer from architecture metadata, selectable via
//     -ctk/-ctv like llama-server
//   - fixed per-layer state in the swa/ssm column (window / snapshot rows)
//   - MTP/nextn usage in dedicated mtp columns

#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <chrono>
#include <map>
#include <vector>

static const int NUM_COL_W = 13; // numeric column cell width (incl. side padding)

// same allowed KV cache types as llama-server (common/arg.cpp)
static const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const char * s) {
    for (const auto & type : kv_cache_types) {
        if (!strcmp(ggml_type_name(type), s)) {
            return type;
        }
    }
    fprintf(stderr, "Unsupported cache type: %s\n", s);
    exit(1);
}

static std::string all_kv_cache_types() {
    std::string msg;
    for (const auto & type : kv_cache_types) {
        if (!msg.empty()) msg += ", ";
        msg += ggml_type_name(type);
    }
    return msg;
}

static void log_callback_quiet(enum ggml_log_level, const char *, void *) {}

// ---- device scan -----------------------------------------------------------
// Two-phase VRAM measurement of every device the backend registry sees:
//   free          — raw, right after llama_backend_init(), nothing loaded yet
//   after-measure — after the real model load + warmup decode (see main).
//                   The delta = what a live deployment on these devices pays
//                   (weights + KV pool + compute buffers + kernels/shaders);
//                   the per-parameter decomposition (n_ctx, n_batch,
//                   --ubatch-size, weights) is computed in the solver phase.
// Row filtering: CPU rows are hidden while GPUs/IGPUs are present (noise for
// planning); they reappear on GPU-less systems so the table stays non-empty.

static const char * dev_type_str(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_META: return "META";
        default:                            return "other";
    }
}

// GiB with two decimals; VRAM figures are read in GiB for planning
static std::string gib_str(size_t bytes) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

// Exact byte count; plain format so values are copy-pasteable into spreadsheets as-is
static std::string bytes_str(size_t bytes) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%zu", bytes);
    return buf;
}

struct dev_snapshot { size_t free_bytes = 0, total_bytes = 0; bool ok = false; };

static void read_all_devices(std::vector<dev_snapshot> & snaps) {
    const size_t ndevs = ggml_backend_dev_count();
    snaps.resize(ndevs);
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        snaps[i].free_bytes = 0;
        snaps[i].total_bytes = 0;
        ggml_backend_dev_memory(dev, &snaps[i].free_bytes, &snaps[i].total_bytes);
        snaps[i].ok = true;
    }
}

// ---- provisional split -----------------------------------------------------
// Read back where the measurement load ACTUALLY placed the weights: every
// tensor of the loaded model carries its buffer, and that buffer's type maps
// back to its owning device via ggml_backend_buft_get_device() — ground truth
// from the same code path that did the splitting (llama_model auto-split by
// free VRAM), no re-derivation.
//
// CRITICAL: do NOT match on ggml_backend_dev_host_buffer_type(dev) — that is
// the backend's HOST buffer type and collides across GPUs sharing a backend
// (both ROCm0 and ROCm1 report the same HIP host buft). Use buft_get_device
// instead, which is per-device.

static bool gpus_present() {
    const size_t ndevs = ggml_backend_dev_count();
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

// Per-GPU accounting: which layers landed where + total weight bytes, read
// back from the loaded model's tensor buffers. Returns device_index -> info.
struct dev_usage { size_t bytes = 0; int first_layer = -1, last_layer = -1; std::vector<int> layers; };

static std::vector<dev_usage> compute_dev_usage(const llama_model * model) {
    const size_t ndevs = ggml_backend_dev_count();
    std::vector<dev_usage> usage(ndevs);

    for (const auto & [tname, tensor] : llama_internal_get_tensor_map(model)) {
        if (!tensor || !tensor->buffer) {
            continue;   // not loaded to a device buffer
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
        // map this tensor's buffer type back to its owning device — round-trips
        // per-device even when several GPUs share one backend (both are HIP)
        ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
        if (!buft_dev) {
            continue;
        }
        const size_t bytes = ggml_nbytes(tensor);

        int il = -1;
        const char * t = tname.c_str();
        if (!strncmp(t, "blk.nextn.", 10)) {
            // nextn block <i> sits at hp index n_layer + i (trunk is [0, n_layer))
            il = (int)model->hparams.n_layer() + atoi(t + 10);
        } else if (!strncmp(t, "blk.", 4)) {
            il = atoi(t + 4);
        }

        for (size_t i = 0; i < ndevs; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || dev != buft_dev) {
                continue;
            }
            usage[i].bytes += bytes;
            if (il >= 0) {
                usage[i].layers.push_back(il);
                if (usage[i].first_layer < 0 || il < usage[i].first_layer) usage[i].first_layer = il;
                if (il > usage[i].last_layer) usage[i].last_layer = il;
            }
            break;
        }
    }

    // de-dupe .layers: it collected one entry per tensor occurrence above, but
    // downstream rate math must see each layer index exactly once
    for (auto & u : usage) {
        std::sort(u.layers.begin(), u.layers.end());
        u.layers.erase(std::unique(u.layers.begin(), u.layers.end()), u.layers.end());
    }

    return usage;
}

static void print_provisional_split(const llama_model * model) {
    const size_t ndevs = ggml_backend_dev_count();
    if (ndevs == 0 || !gpus_present()) {
        return;   // no GPUs: nothing to split
    }
    std::vector<dev_usage> usage = compute_dev_usage(model);

    fprintf(stderr, "\n");
    fprintf(stderr, "  provisional split (as placed by the measurement load):\n\n");
    fprintf(stderr, "    %-4s %-6s %-10s %-14s  %s\n", "idx", "type", "layers", "weights", "name");
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU && gpus_present()) {
            continue;   // CPU rows hidden while GPUs are present
        }

        char range[24];
        snprintf(range, sizeof(range),
                 usage[i].first_layer >= 0 ? "%d-%d" : "-",
                 usage[i].first_layer, usage[i].last_layer);
        fprintf(stderr, "    %-4zu %-6s %-10s %-14s  %s\n",
                i, dev_type_str(type), range, gib_str(usage[i].bytes).c_str(),
                ggml_backend_dev_name(dev));
    }
}

// Per-device probe-memory rates (nextn layers counted separately — they serve
// the MTP draft context, not the main one). Indexed by backend device order.
struct kv_rate_info {
    uint64_t kv_bytes_per_cell = 0; int n_attn_layers = 0; // attention KV pool
    uint64_t ssm_bytes_per_row = 0; int n_ssm_layers = 0;  // recurrent state, per checkpoint row
    uint64_t mtp_kv_bytes_per_cell = 0; int n_mtp_layers = 0; // nextn KV pool (MTP draft)
};

static std::vector<kv_rate_info> compute_kv_rates(const llama_model * model,
                                                  const std::vector<dev_usage> & usage,
                                                  ggml_type type_k, ggml_type type_v,
                                                  bool flash_attn) {
    const auto & hp = model->hparams;
    const uint32_t n_layer_all   = hp.n_layer_all;
    const uint32_t n_layer_nextn = hp.n_layer_nextn;
    const bool v_t = !flash_attn;
    std::vector<kv_rate_info> kv_rate(usage.size());
    for (size_t i = 0; i < usage.size(); ++i) {
        for (int il : usage[i].layers) {
            if (il >= (int)n_layer_all) continue;
            // nextn layers: their KV pool lives in the MTP draft context
            if (il >= (int)(n_layer_all - n_layer_nextn)) {
                const uint32_t ek = hp.n_embd_k_gqa(il);
                const uint32_t ev = v_t ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);
                uint64_t cb = (uint64_t)ggml_row_size(type_k, ek);
                if (!hp.is_mla()) cb += (uint64_t)ggml_row_size(type_v, ev);
                kv_rate[i].mtp_kv_bytes_per_cell += cb;
                ++kv_rate[i].n_mtp_layers;
                continue;
            }
            if (hp.is_recr(il)) {
                // recurrent state rows are always F32 (create_memory hardcodes them)
                uint64_t cb = (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_r());
                cb += (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
                if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
                    cb += (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.ple_conv_state());
                }
                kv_rate[i].ssm_bytes_per_row += cb;
                ++kv_rate[i].n_ssm_layers;
            } else if (hp.has_kv(il)) {
                const uint32_t ek = hp.n_embd_k_gqa(il);
                const uint32_t ev = v_t ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);
                uint64_t cb = (uint64_t)ggml_row_size(type_k, ek);
                if (!hp.is_mla()) cb += (uint64_t)ggml_row_size(type_v, ev);
                kv_rate[i].kv_bytes_per_cell += cb;
                ++kv_rate[i].n_attn_layers;
            }
        }
    }
    return kv_rate;
}

// ---- KV pool solver ---------------------------------------------------------
// Largest KV pool size (in 256-cell blocks) that a greedy layer placement
// affords on every GPU's available budget, found by bisecting (log2 steps).
// Integer arithmetic throughout; verbose trace of every step so the search
// and each decision can be verified (removed once proven correct).

struct layer_cost {
    uint64_t weights = 0;       // tensor bytes for this layer (GGUF metadata)
    uint64_t kv_bpc  = 0;       // KV-cache bytes per cell, main pool (attention layers)
    uint64_t mtp_kv_bpc = 0;    // KV-cache bytes per cell, MTP draft pool (nextn layers)
    uint64_t ssm_state = 0;     // recurrent checkpoint state (rows x row-set), else 0
};

static std::vector<layer_cost> compute_layer_costs(const llama_model * model,
                                                   const std::map<int, int64_t> & layer_weights,
                                                   ggml_type type_k, ggml_type type_v,
                                                   bool flash_attn, uint32_t state_rows) {
    const auto & hp = model->hparams;
    const bool v_t = !flash_attn;
    const uint32_t n_nextn_first = hp.n_layer_all - hp.n_layer_nextn;   // first nextn index
    std::vector<layer_cost> out(hp.n_layer_all);
    for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
        auto it = layer_weights.find((int)il);
        if (it != layer_weights.end()) {
            out[il].weights = (uint64_t)it->second;
        }
        if (hp.is_recr(il)) {
            // recurrent state rows are always F32 (create_memory hardcodes them)
            uint64_t cb = (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_r());
            cb += (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
            if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
                cb += (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.ple_conv_state());
            }
            out[il].ssm_state = cb * state_rows;
        } else if (hp.has_kv(il)) {
            const uint32_t ek = hp.n_embd_k_gqa(il);
            const uint32_t ev = v_t ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);
            uint64_t cb = (uint64_t)ggml_row_size(type_k, ek);
            if (!hp.is_mla()) {
                cb += (uint64_t)ggml_row_size(type_v, ev);
            }
            // nextn layers back the MTP draft context, not the main pool
            if (il >= n_nextn_first) {
                out[il].mtp_kv_bpc = cb;
            } else {
                out[il].kv_bpc = cb;
            }
        }
    }
    return out;
}

// Try placing n 256-cell blocks: walk layers in order — each intermediate GPU
// takes layers greedily until the next one would exceed its budget; the LAST
// GPU has no choice, it must take all remaining layers, so it only fits if
// their total cost stays inside its budget (otherwise "over limit" is shown).
// Every decision is traced via fprintf (verification aid).
static bool try_place_kv_pool(uint32_t n_blocks,
                              const std::vector<layer_cost> & layers,
                              const std::vector<uint64_t> & budgets,
                              const std::vector<std::string> & names) {
    const uint64_t cells = (uint64_t)n_blocks * 256;
    fprintf(stderr, "\n    testing with n = %-5u blocks (%llu cells)\n",
            n_blocks, (unsigned long long)cells);

    // prefix sums over layers: O(1) range queries for cost and kv bytes-per-cell.
    // The per-cell charge covers BOTH pools a layer's cells live in: the main
    // KV pool (attention layers) and the MTP draft pool (nextn layers).
    const size_t nl = layers.size();
    std::vector<uint64_t> p_wss(nl + 1, 0), p_bpc(nl + 1, 0);
    for (size_t i = 0; i < nl; ++i) {
        p_wss[i + 1] = p_wss[i] + layers[i].weights + layers[i].ssm_state;
        p_bpc[i + 1] = p_bpc[i] + layers[i].kv_bpc + layers[i].mtp_kv_bpc;
    }

    size_t il = 0;
    for (size_t g = 0; g < budgets.size() && il < nl; ++g) {
        const bool is_last = (g + 1 == budgets.size());
        const size_t first_layer = il;
        uint64_t used = 0;

        if (!is_last) {
            // intermediate GPU: take layers while they fit, hand over the rest
            while (il < nl) {
                const uint64_t cost = layers[il].weights + cells * (layers[il].kv_bpc
                                                                   + layers[il].mtp_kv_bpc)
                                    + layers[il].ssm_state;
                if (used + cost > budgets[g]) break;   // next GPU continues at il
                used += cost;
                ++il;
            }
        } else {
            // last GPU: no choice — all remaining layers are its, so the range
            // total against its budget decides the candidate outright
            used = (p_wss[nl] - p_wss[first_layer])
                 + cells * (p_bpc[nl] - p_bpc[first_layer]);
        }

        const bool has_layers = is_last || (il > first_layer);   // last GPU always "owns" the rest
        const size_t shown_last = has_layers ? (is_last ? nl - 1 : il - 1) : 0;
        char range[64];
        snprintf(range, sizeof range, "%s", has_layers ? "" : "(none)");
        if (has_layers) {
            snprintf(range, sizeof range, "%zu..%zu", first_layer, shown_last);
        }

        const uint64_t bpc = has_layers ? (p_bpc[shown_last + 1] - p_bpc[first_layer]) : 0;
        const bool fits = (used <= budgets[g]);   // always true for intermediate GPUs (greedy)

        fprintf(stderr, "      %s: layers = %s, cache size = %llu × %llu = %llu",
                names[g].c_str(), range,
                (unsigned long long)cells, (unsigned long long)bpc,
                (unsigned long long)cells * bpc);
        if (!is_last || fits) {
            fprintf(stderr, ", not used = %llu\n",
                    (unsigned long long)(budgets[g] - used));
        } else {
            fprintf(stderr, ", over limit = %llu\n",
                    (unsigned long long)(used - budgets[g]));
        }

        if (is_last) return fits;
    }
    return true;   // all layers placed — candidate fits
}

// Bisect for the largest fitting n of 256-cell blocks: at most log2(n_max)
// steps. Returns the winning block count (0 = nothing fits).
static uint32_t solve_kv_pool(uint32_t n_max,
                              const std::vector<layer_cost> & layers,
                              const std::vector<uint64_t> & budgets,
                              const std::vector<std::string> & names) {
    fprintf(stderr, "\n  kv pool solver: bisect n of 256-cell blocks (integer arithmetic)\n");
    if (n_max == 0 || layers.empty() || budgets.empty()) {
        fprintf(stderr, "    n_max = 0 — nothing to solve\n");
        return 0;
    }
    // the budgets each candidate is checked against
    fprintf(stderr, "    budgets:");
    for (size_t g = 0; g < budgets.size(); ++g) {
        fprintf(stderr, " %s = %llu", names[g].c_str(),
                (unsigned long long)budgets[g]);
    }
    fprintf(stderr, "\n");
    uint32_t low = 0, high = n_max;   // bisection over the whole range
    while (low < high) {
        const uint32_t mid = low + (high - low + 1) / 2;   // rounds UP -> monotonic progress
        if (try_place_kv_pool(mid, layers, budgets, names)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    fprintf(stderr, "    OPTIMAL: n=%-7u blocks = %llu kv cells\n",
            low, (unsigned long long)low * 256ULL);
    return low;
}

// On Linux the ggml CPU backend reports free == total unconditionally
// (ggml-cpu.cpp:377) — a deliberate assumption, not a measurement. Read the
// real figure from /proc/meminfo so the table is truthful there; leave the
// raw value where /proc/meminfo doesn't provide it (Windows, old kernels).
static bool meminfo_available(size_t * available) {
#if defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) {
        return false;
    }
    char line[256];
    size_t kb = 0;
    bool ok = false;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
            ok = true;
            break;
        }
    }
    fclose(f);
    *available = ok ? kb * 1024 : 0;
    return ok;
#else
    (void)available;
    return false;
#endif
}

// Replace the CPU rows' free with the real MemAvailable value (see
// meminfo_available). No shaders exist for CPUs, so apply to every phase —
// both measurements are the same quantity.
static void fix_cpu_free(std::vector<dev_snapshot> & snaps) {
    const size_t ndevs = ggml_backend_dev_count();
    for (size_t i = 0; i < ndevs && i < snaps.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU &&
            meminfo_available(&snaps[i].free_bytes)) {
            // total stays as reported; only free is corrected
        }
    }
}

// ---- warmup decode ---------------------------------------------------------
// Real-load measurement (Michael, 2026-11-13): the tool loads the model with
// default params and decodes a real prompt once so every kernel the deployment
// will launch is uploaded to the devices — kernels only materialize on their
// first execution (verified across all backends, concept §17). The prompt is
// ~20K chars of Hamlet (warmup-text.h): natural-language tokens with high
// diversity, so a MoE router exercises many experts and their kernels load.
// Returns the number of tokens decoded in prefill.

#include "warmup-text.h"

static int warmup_decode(llama_context * ctx, const std::vector<int32_t> & tokens) {
    const int n_tok = (int)tokens.size();
    if (n_tok <= 0) {
        return 0;
    }

    // batch construction follows fllama.c exactly (and it does not crash):
    // raw mallocs, a flat seq_id pointer array without per-token inner
    // allocations, and manual frees. llama_batch_init/llama_batch_free use
    // per-token inner allocations this deployment pattern never touches.
    int seq_id = 0;   // non-const: batch.seq_id pointers must stay valid during llama_decode

    auto build_batch = [&](std::vector<int32_t> * tok_vec, int32_t pos_start, bool logits_last_only) {
        const size_t n = tok_vec->size();
        llama_batch batch;
        batch.n_tokens = (int32_t)n;
        batch.token    = (llama_token *)  malloc(n * sizeof(llama_token));
        batch.embd     = nullptr;
        batch.pos      = (llama_pos *)     malloc(n * sizeof(llama_pos));
        batch.n_seq_id = (int32_t *)       malloc(n * sizeof(int32_t));
        batch.seq_id   = (llama_seq_id **) malloc(n * sizeof(llama_seq_id *));
        batch.logits   = (int8_t *)        malloc(n);
        for (size_t i = 0; i < n; ++i) {
            batch.token[i]    = (*tok_vec)[i];
            batch.pos[i]      = pos_start + (llama_pos)i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = &seq_id;
            // output only at the single marked position, else no outputs
            batch.logits[i]   = logits_last_only && i == n - 1 ? 1 : 0;
        }
        return batch;
    };

    auto free_batch = [](llama_batch & batch) {
        free(batch.token);
        free(batch.pos);
        free(batch.n_seq_id);
        free(batch.seq_id);   // the pointer array only — seq_id[i] point into seq_id above
        free(batch.logits);
    };

    // prefill: submit the prompt in n_batch-sized chunks (the backend
    // subdivides each into ubatches internally — same graph sizes as
    // production). Positions stay contiguous across chunks.
    const int32_t n_batch = llama_n_batch(ctx);
    for (int32_t start = 0; start < n_tok; start += n_batch) {
        const int32_t end = std::min(n_tok, start + n_batch);
        std::vector<int32_t> chunk(tokens.begin() + start, tokens.begin() + end);
        // only the final chunk's last token produces logits
        llama_batch batch = build_batch(&chunk, start, end == n_tok);
        const int rc = llama_decode(ctx, batch);
        free_batch(batch);
        if (rc != 0) {
            fprintf(stderr, "  warmup: prefill decode failed at token %d\n", start);
            return 0;
        }
    }

    // a few generate steps so the single-token decode path is exercised too
    std::vector<int32_t> next(1);
    for (int g = 0; g < 4; ++g) {
        next[0] = tokens[n_tok - 1];
        llama_batch gb = build_batch(&next, n_tok + g, true);
        const int rc = llama_decode(ctx, gb);
        free_batch(gb);
        if (rc != 0) {
            fprintf(stderr, "  warmup: generate decode %d failed\n", g);
            break;
        }
    }

    return n_tok;
}
static void print_device_scan(const std::vector<dev_snapshot> & raw,
                              const std::vector<dev_snapshot> & measure,
                              const llama_model * model, uint32_t n_ctx_measure,
                              ggml_type type_k, ggml_type type_v, bool flash_attn,
                              uint32_t streams, uint32_t state_rows, bool mtp_measured,
                              double keep_free,
                              const std::map<int, int64_t> & layer_weights) {
    // decide row visibility first: when any GPU/IGPU is present, CPU rows are
    // noise for planning and get hidden (Michael, 2026-11-12); without GPUs the
    // CPU row is shown again so the scan never renders an empty table.
    bool gpu_present = false;
    const size_t ndevs = ggml_backend_dev_count();
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_present = true;
            break;
        }
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "  device scan:\n\n");

    if (ndevs == 0) {
        fprintf(stderr, "    no devices registered\n");
        return;
    }

    char hdr[256];
    snprintf(hdr, sizeof(hdr), "    %-4s %-6s %-14s %-14s %-14s  %s",
             "idx", "type", "total", "free", "after-measure", "name");
    fprintf(stderr, "%s\n", hdr);

    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            fprintf(stderr, "    %-4zu (null)\n", i);
            continue;
        }

        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (gpu_present && type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;  // hidden while GPUs are present
        }
        const size_t free_measure = (i < measure.size() && measure[i].ok) ? measure[i].free_bytes : 0;

        char line[512];
        snprintf(line, sizeof(line),
                 "    %-4zu %-6s %-14s %-14s %-14s  %s",
                 i,
                 dev_type_str(type),
                 gib_str(raw[i].total_bytes).c_str(),
                 gib_str(raw[i].free_bytes).c_str(),
                 gib_str(free_measure).c_str(),
                 ggml_backend_dev_name(dev));
        fprintf(stderr, "%s\n", line);
    }

    // per-GPU decomposition of the free-VRAM delta: what the live deployment
    // load consumed, split into its three components —
    //   weights : tensor bytes placed on this device (read back from the model)
    //   probe-KV: the measurement context's probe memory = attention KV pool
    //             (n_ctx cells x this device's rate, x the stream count)
    //             + MTP draft pool (if measured) + recurrent SSM state
    //             (state rows per layer at this device's state rate)
    //   kernels : shaders + compute buffers + alignment slack = delta − weights
    //             − probe-KV. This residual is the fixed per-GPU overhead a
    //             deployment always pays (the number we actually budget for).
    fprintf(stderr, "\n");
    fprintf(stderr, "  measured overhead per device (free delta decomposition):\n\n");
    fprintf(stderr, "    %-4s %-6s %12s %12s %12s %12s  %s\n",
            "idx", "type", "delta", "weights", "probe-KV", "kernels", "name");

    std::vector<dev_usage> usage;
    std::vector<kv_rate_info> kv_rate;
    const bool have_decomp = (model && n_ctx_measure > 0);
    if (have_decomp) {
        usage   = compute_dev_usage(model);
        kv_rate = compute_kv_rates(model, usage, type_k, type_v, flash_attn);
    }

    // per-device decomposition values: computed once, rendered in GiB and bytes.
    // The byte view also shows the arithmetic behind probe-KV for inspection.
    struct dev_row {
        size_t idx; const char * type_str; const char * name;
        uint64_t delta, weights, probe_kv, kernels;
        uint64_t kv_cells, kv_bpc, streams, state_rows, ssm_bpr, mtp_bpc;  // probe-KV formula inputs
        int attn_layers, ssm_layers, mtp_layers;
    };
    std::vector<dev_row> rows;
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (gpu_present && type == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        if (!(i < measure.size() && measure[i].ok)) continue;
        const uint64_t delta = (raw[i].free_bytes > measure[i].free_bytes)
                             ? raw[i].free_bytes - measure[i].free_bytes : 0;

        uint64_t weights_i = 0, probe_kv_i = 0, kv_cells = 0, kv_bpc = 0, ssm_bpr = 0, mtp_bpc = 0;
        int attn_layers = 0, ssm_layers = 0, mtp_layers = 0;
        if (have_decomp && i < usage.size() && i < kv_rate.size()) {
            weights_i   = usage[i].bytes;
            kv_cells    = ((uint64_t)n_ctx_measure + 255) / 256 * 256; // GGML_PAD, as the cache does
            kv_bpc      = kv_rate[i].kv_bytes_per_cell;
            ssm_bpr     = kv_rate[i].ssm_bytes_per_row;
            attn_layers = kv_rate[i].n_attn_layers;
            ssm_layers  = kv_rate[i].n_ssm_layers;
            if (mtp_measured) {
                mtp_bpc    = kv_rate[i].mtp_kv_bytes_per_cell;
                mtp_layers = kv_rate[i].n_mtp_layers;
            }
            // probe memory = attention KV pool + MTP draft pool (if measured)
            //               + recurrent SSM state rows (= n_seq_max x (1 + n_rs_seq) per layer)
            // The KV pools are shared: one pool of n_ctx cells serves all parallel
            // streams (unified), so streams must NOT multiply its size — only the
            // per-sequence SSM state rows do.
            probe_kv_i  = kv_cells * (kv_bpc + mtp_bpc)
                        + (uint64_t)state_rows * ssm_bpr;
        }
        // kernels = residual after removing the two accounted-for terms
        uint64_t kernels_i = (delta > weights_i + probe_kv_i) ? delta - weights_i - probe_kv_i : 0;

        rows.push_back({i, dev_type_str(type), ggml_backend_dev_name(dev),
                        delta, weights_i, probe_kv_i, kernels_i,
                        kv_cells, kv_bpc, streams, state_rows, ssm_bpr, mtp_bpc,
                        attn_layers, ssm_layers, mtp_layers});
    }

    for (const auto & r : rows) {
        char line[512];
        snprintf(line, sizeof(line),
                 "    %-4zu %-6s %12s %12s %12s %12s  %s",
                 r.idx, r.type_str,
                 gib_str(r.delta).c_str(),
                 have_decomp ? gib_str(r.weights).c_str() : "-",
                 have_decomp ? gib_str(r.probe_kv).c_str() : "-",
                 have_decomp ? gib_str(r.kernels).c_str() : "-",
                 r.name);
        fprintf(stderr, "%s\n", line);
    }

    // byte view: exact values (no rounding), plus the probe-KV arithmetic
    fprintf(stderr, "\n");
    fprintf(stderr, "  measured overhead per device (bytes, exact):\n\n");
    fprintf(stderr, "    %-4s %-6s %12s %12s %12s %12s  %s\n",
            "idx", "type", "delta", "weights", "probe-KV", "kernels", "name");
    for (const auto & r : rows) {
        fprintf(stderr,
                "    %-4zu %-6s %12s %12s %12s %12s  %s\n",
                r.idx, r.type_str,
                bytes_str(r.delta).c_str(),
                have_decomp ? bytes_str(r.weights).c_str() : "-",
                have_decomp ? bytes_str(r.probe_kv).c_str() : "-",
                have_decomp ? bytes_str(r.kernels).c_str() : "-",
                r.name);
    }

    if (have_decomp) {
        fprintf(stderr, "\n  probe-KV derivation (bytes): context x cell + mtp draft x cell"
                        " + state rows x ssm_state:"
                        " (unified pools: streams do not multiply the KV size)\n");
        for (const auto & r : rows) {
            fprintf(stderr, "    %s (%d attention / %d SSM layers, %s K / %s V):\n",
                    r.name, r.attn_layers, r.ssm_layers,
                    ggml_type_name(type_k), ggml_type_name(type_v));
            // Factors print right-aligned in fixed-width fields so the 'x' and
            // '=' characters stack vertically across rows (term values are the
            // aligned summands that make up probe-KV below).
            if (r.kv_bpc > 0) {
                fprintf(stderr, "      %-12s %9lu × %9lu = %14lu\n",
                        "context:", r.kv_cells, r.kv_bpc,
                        r.kv_cells * r.kv_bpc);
            }
            if (r.mtp_bpc > 0) {
                fprintf(stderr, "      %-12s %9lu × %9lu = %14lu   (%d nextn layers)\n",
                        "mtp draft:", r.kv_cells, r.mtp_bpc,
                        r.kv_cells * r.mtp_bpc, r.mtp_layers);
            }
            if (r.ssm_bpr > 0) {
                fprintf(stderr, "      %-12s %9lu × %9lu = %14lu   (%d SSM layers)\n",
                        "checkpoints:", r.state_rows, r.ssm_bpr,
                        r.state_rows * r.ssm_bpr, r.ssm_layers);
            }
            fprintf(stderr, "      %s\n", "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500");
            fprintf(stderr, "      %-12s                       = %14lu\n",
                    "probe-KV:", r.probe_kv);
        }
    }

    // ---- split budget --------------------------------------------------------
    // How much VRAM each GPU leaves for planning:
    //   reserve   = total × keep_free
    //   budget    = total - reserve              (= total × (1 - keep_free))
    //   loaded    = weights + probe-KV  (tensors + cache/checkpoint state)
    //   available = budget - loaded
    // Integer arithmetic throughout, so every printed value chains from the
    // previous one. The accumulated available across all GPUs is the budget
    // the proportional split will distribute.
    if (have_decomp && gpu_present) {
        fprintf(stderr, "\n");
        fprintf(stderr, "  split budget (bytes): reserve = total × keep_free(%.2f); "
                        "budget = total - reserve;\n", keep_free);
        fprintf(stderr, "                         loaded = weights + probe-KV; "
                        "available = budget - loaded\n\n");
        fprintf(stderr, "    %-4s %-6s %15s %15s %15s %15s %15s  %s\n",
                "idx", "type", "total", "reserve", "budget", "loaded", "available", "name");
        int64_t accumulated_available = 0;
        std::vector<uint64_t> gpu_budgets;   // solver input: per-GPU available, in row order
        std::vector<std::string> gpu_names;  // solver input: matching device names
        for (const auto & r : rows) {
            if (!raw[r.idx].ok || raw[r.idx].total_bytes == 0) continue;
            const int64_t total   = (int64_t)raw[r.idx].total_bytes;
            const int64_t reserve = (int64_t)((double)total * keep_free);
            const int64_t budget  = total - reserve;
            const int64_t loaded  = (int64_t)r.weights + (int64_t)r.probe_kv;
            const int64_t available = budget - loaded;
            accumulated_available += available;
            // negative available = overcommitted GPU: it cannot hold anything
            gpu_budgets.push_back(available > 0 ? (uint64_t)available : 0);
            gpu_names.push_back(r.name);
            fprintf(stderr, "    %-4zu %-6s %15lld %15lld %15lld %15lld %15lld  %s\n",
                    r.idx, r.type_str,
                    (long long)total, (long long)reserve, (long long)budget,
                    (long long)loaded, (long long)available, r.name);
        }
        fprintf(stderr, "    %-76s%15lld\n",
                "total available:", (long long)accumulated_available);

        // theoretical kv-cache ceiling: how many cells fit into the accumulated
        // available budget at full per-cell cost. The pool capacity is a
        // property of the pool itself — parallel sequences share it, they do
        // not multiply its size, so streams plays no role here.
        //   total cell size = sum over GPUs of attention bytes-per-cell
        //   upper limit     = total available ÷ total cell size    (integer division)
        uint64_t cell_size_per_token = 0;
        for (const auto & r : rows) {
            if (!raw[r.idx].ok || raw[r.idx].total_bytes == 0) continue;
            cell_size_per_token += r.kv_bpc;
        }
        fprintf(stderr, "    %-76s%15llu\n",
                "total cell size:", (unsigned long long)cell_size_per_token);
        if (cell_size_per_token > 0 && accumulated_available > 0) {
            const uint64_t kv_upper =
                (uint64_t)accumulated_available / cell_size_per_token;
            fprintf(stderr, "    %-76s%15llu   tokens\n",
                    "kv cache upper limit:", (unsigned long long)kv_upper);

            // ---- kv pool solver ----------------------------------------------
            // step 3: the pool size must be a multiple of 256 cells (GGML_PAD),
            // so bisect the count of 256-cell blocks up to that ceiling.
            const uint32_t n_max = (uint32_t)(kv_upper / 256);
            fprintf(stderr, "\n    n_max = kv upper limit / 256 = %llu\n",
                    (unsigned long long)n_max);
            auto layer_costs = compute_layer_costs(model, layer_weights,
                                                   type_k, type_v, flash_attn, state_rows);
            solve_kv_pool(n_max, layer_costs, gpu_budgets, gpu_names);
        } else {
            fprintf(stderr, "    %-76s%s\n", "kv cache upper limit:", "n/a");
        }
    }

    fprintf(stderr, "\n");
}

// ---- table printing -----------------------------------------------------------
// widths[] holds per-column cell width INCLUDING the 1-space padding on each
// side of the content. Column 0 is a left-aligned label; all others are
// right-aligned numbers.

static std::string box(const char * tl, const char * mid, const char * tr,
                       const std::vector<int> & widths) {
    std::string s(tl);
    for (size_t i = 0; i < widths.size(); ++i) {
        if (i > 0) s += mid;
        for (int j = 0; j < widths[i]; ++j) s += "─";
    }
    return s + tr;
}

static std::string data_row(const char * label, const std::vector<int64_t> & vals,
                            const std::vector<int> & widths) {
    std::string s("│");
    const int l = (int)strlen(label);
    // col 0: right-aligned (layer numbers and labels share the column)
    s += std::string(widths[0] - 1 - l, ' ') + label + " │";
    for (size_t i = 1; i < vals.size(); ++i) {
        char buf[24];
        snprintf(buf, sizeof(buf), "%lld", (long long)vals[i]);
        const int n = (int)strlen(buf);
        s += std::string(widths[i] - n - 1, ' ') + buf + " │";
    }
    return s;
}

static std::string header_row(const std::vector<const char *> & hdr,
                              const std::vector<int> & widths) {
    std::string s("│");
    for (size_t i = 0; i < hdr.size(); ++i) {
        const int l = (int)strlen(hdr[i]);
        // visually centered over the full width; " │" supplies one right
        // padding space, so subtract it from pad_r. On uneven padding the
        // text leans left:
        const int pad_l = (widths[i] - l) / 2;
        const int pad_r = widths[i] - l - pad_l - 1;
        s += std::string(pad_l, ' ') + hdr[i] + std::string(pad_r, ' ') + " │";
    }
    return s;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s [options] <model.gguf>\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help             show this help and exit\n");
    fprintf(stderr, "  --keep-free FLOAT      reserve this fraction of each GPU's VRAM (default 0.03)\n");
    fprintf(stderr, "  --ubatch-size INT      ubatch size for the measurement graphs (default 512)\n");
    fprintf(stderr, "  --parallel INT         plan P parallel sequences: 9P SSM state rows, KV pool x P\n");
    fprintf(stderr, "  --checkpoints INT      SSM state rows per recurrent layer (default 1; exclusive with --parallel)\n");
    fprintf(stderr, "  --cache-type-k TYPE    KV cache K quantization (default f16)\n");
    fprintf(stderr, "  --cache-type-v TYPE    KV cache V quantization (default f16)\n");
    fprintf(stderr, "  --flash-attn           use the flash-attention V layout (default)\n");
    fprintf(stderr, "  --no-flash-attn        use the plain V layout\n");
    fprintf(stderr, "  --no-mtp               skip the MTP draft context measurement\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Allowed cache types: %s\n", all_kv_cache_types().c_str());
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * path = nullptr;   // first non-option argument, wherever it appears
    ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    bool flash_attn = true;
    uint32_t ubatch_size = 512;   // default: matches llama.cpp's n_ubatch
    int checkpoints_arg = -1;     // -1 = not given
    int parallel_arg = -1;        // -1 = not given
    double keep_free = 0.03;      // design default: reserve 3% of each GPU's VRAM
    bool use_mtp = true;          // auto: MTP draft context is measured when the model has a nextn block

    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--cache-type-k" || arg == "--cache-type-v" || arg == "--ubatch-size" ||
            arg == "--checkpoints" || arg == "--parallel" || arg == "--keep-free") {
            if (++a >= argc) {
                fprintf(stderr, "error: option '%s' requires a value\n", argv[a - 1]);
                return 1;
            }
        }
        if      (arg == "--cache-type-k")  type_k = kv_cache_type_from_str(argv[a]);
        else if (arg == "--cache-type-v")  type_v = kv_cache_type_from_str(argv[a]);
        else if (arg == "--flash-attn")    flash_attn = true;
        else if (arg == "--no-flash-attn") flash_attn = false;
        else if (arg == "--ubatch-size")   ubatch_size = atoi(argv[a]);
        else if (arg == "--checkpoints")   checkpoints_arg = atoi(argv[a]);
        else if (arg == "--parallel")      parallel_arg = atoi(argv[a]);
        else if (arg == "--keep-free")     keep_free = atof(argv[a]);
        else if (arg == "--no-mtp")        use_mtp = false;
        else if (!path)                    path = argv[a];
        else {
            fprintf(stderr, "error: unexpected argument '%s' (model file already given as '%s')\n",
                    arg.c_str(), path);
            return 1;
        }
    }

    if (!path) {
        print_usage(argv[0]);
        return 1;
    }

    if (checkpoints_arg >= 0 && checkpoints_arg < 1) {
        fprintf(stderr, "error: --checkpoints must be >= 1 (the live state row always exists) — got %d\n",
                checkpoints_arg);
        return 1;
    }

    if (parallel_arg >= 0 && parallel_arg < 1) {
        fprintf(stderr, "error: --parallel must be >= 1 — got %d\n", parallel_arg);
        return 1;
    }

    if (checkpoints_arg >= 0 && parallel_arg >= 0) {
        fprintf(stderr, "error: --parallel and --checkpoints cannot be combined "
                        "(--parallel P plans 9 state rows per stream for P streams)\n");
        return 1;
    }

    // SSM state plan for the probe run (rows per recurrent layer):
    //   --parallel P        : P streams x (8 backtrack + 1 live) = 9P rows,
    //                         via cp.n_seq_max = P and cp.n_rs_seq = 8
    //   --checkpoints N / default: 1 stream, N rows via cp.n_rs_seq = N - 1
    const uint32_t probe_parallel = (parallel_arg >= 0) ? (uint32_t)parallel_arg : 1;
    const uint32_t probe_rs_seq   = (parallel_arg >= 0) ? 8
                                        : ((checkpoints_arg >= 0) ? (uint32_t)checkpoints_arg - 1 : 0);
    const uint32_t state_rows     = probe_parallel * (1u + probe_rs_seq);

    if (!(keep_free >= 0.0 && keep_free < 1.0)) {
        fprintf(stderr, "error: --keep-free must be in [0, 1) — got %.3f\n", keep_free);
        return 1;
    }

    const bool v_trans = !flash_attn;

    // ---- read GGUF metadata directly for tensor sizes ----------------------
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        fprintf(stderr, "failed to open %s\n", path);
        return 1;
    }

    const char * arch = nullptr;
    int64_t id_arch = gguf_find_key(ctx, "general.architecture");
    if (id_arch >= 0) arch = gguf_get_val_str(ctx, id_arch);

    const char * name = nullptr;
    int64_t id_name = gguf_find_key(ctx, "general.name");
    if (id_name >= 0) name = gguf_get_val_str(ctx, id_name);

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::map<int, int64_t> layer_weights;
    // nextn block indices (runtime layout): blk.nextn.<i> maps to hp layer
    // n_layer + i — the trunk occupies [0, n_layer), nextn sits right after.
    // Read from GGUF metadata: this no_alloc parse predates the model load.
    int64_t n_layer_meta = 0;
    if (arch) {
        const std::string nk = std::string(arch) + ".n_layer";
        int64_t id_nlayer = gguf_find_key(ctx, nk.c_str());
        if (id_nlayer >= 0) {
            n_layer_meta = (int64_t)gguf_get_val_u32(ctx, id_nlayer);
        }
    }
    // per-layer routed-expert flags derived from the GGUF tensor names —
    // needed because some architectures (e.g. granite-hybrid) never fill
    // n_ff_exp_arr and key their experts off a global n_expert only.
    // Expert tensors: ffn_{gate,up,down}_exps (packed 3D) or the per-expert
    // naming ffn_{gate,up,down}.<exp_id>
    std::vector<bool> layer_has_exps;
    // global tensors (outside blk.*), each one a visible table row:
    //   token_embd.weight  -> "embed"  row (before layer 0)
    //   output_norm.weight -> "norm"   row (after the last layer)
    //   output.weight      -> "output" row (after norm — lm_head, data flow
    //                          order: embed -> layers -> norm -> output)
    // everything else non-blk gets its own named row as well — TOTAL must
    // equal the sum of the visible rows, never include hidden tensors.
    int64_t embed_bytes = 0, norm_bytes = 0, output_bytes = 0;
    std::vector<std::pair<std::string, int64_t>> other_tensors;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(ctx, tid);
        size_t tsize = gguf_get_tensor_size(ctx, tid);

        if (!strcmp(tname, "token_embd.weight")) { embed_bytes += tsize; continue; }
        if (!strcmp(tname, "output_norm.weight")) { norm_bytes += tsize; continue; }
        if (!strcmp(tname, "output.weight"))      { output_bytes += tsize; continue; }

        int il = -1;
        const char * p_nextn = strstr(tname, "blk.nextn.");
        if (p_nextn) {
            // nextn tensor: blk.nextn.<i>.<rest> -> hp layer n_layer + i
            int nin = 0;
            if (n_layer_meta > 0 && sscanf(p_nextn + strlen("blk.nextn."), "%d", &nin) == 1) {
                il = (int)n_layer_meta + nin;
            }
        } else if (strstr(tname, "blk.")) {
            const char * p = strstr(tname, "blk.");
            sscanf(p + 4, "%d", &il);
        }
        if (il >= 0) {
            layer_weights[il] += tsize;
            // routed-expert tensors mark this layer as MoE (see note above)
            const bool exp_tensor = strstr(tname, "ffn_gate_exp") || strstr(tname, "ffn_up_exp") ||
                                    strstr(tname, "ffn_down_exp") || strstr(tname, "ffn_gate.") ||
                                    strstr(tname, "ffn_up.") || strstr(tname, "ffn_down.");
            if (exp_tensor) {
                if (il >= (int)layer_has_exps.size()) layer_has_exps.resize(il + 1, false);
                layer_has_exps[il] = true;
            }
            continue;
        }

        other_tensors.emplace_back(tname, (int64_t)tsize);
    }

    // ---- load model via llama API (real weights) ---------------------------------
    llama_log_set(log_callback_quiet, nullptr);  // suppress verbose loading logs

    llama_backend_init();

    // two-phase device scan — raw VRAM (nothing loaded yet), then after the
    // real model load + warmup decode (scan_measure below).
    std::vector<dev_snapshot> scan_raw, scan_measure;
    read_all_devices(scan_raw);
    fix_cpu_free(scan_raw);

    // count real GPUs and sum their free VRAM (planning units)
    size_t n_gpus = 0;
    uint64_t gpu_free_total = 0;
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            n_gpus++;
            gpu_free_total += scan_raw[i].free_bytes;
        }
    }

    // Guard 1: this tool measures GPU deployments — no GPUs, nothing to do.
    if (n_gpus == 0) {
        fprintf(stderr, "error: no GPU devices found — gpu-split requires at least one GPU\n");
        gguf_free(ctx);
        llama_backend_free();
        return 1;
    }

    // total model weight bytes from the GGUF (all tensors), for the fit check
    int64_t model_bytes = embed_bytes + norm_bytes + output_bytes;
    for (const auto & [il, w] : layer_weights)      model_bytes += w;
    for (const auto & [tn, sz] : other_tensors)     model_bytes += sz;

    // Guard 2: the whole model must fit on the GPUs. Weights alone exceeding
    // total free VRAM is a definite no — fail before paying for the load.
    if ((uint64_t)model_bytes > gpu_free_total) {
        fprintf(stderr, "error: model too large — weights (%s) exceed total free GPU VRAM (%s)\n",
                gib_str((size_t)model_bytes).c_str(), gib_str(gpu_free_total).c_str());
        gguf_free(ctx);
        llama_backend_free();
        return 1;
    }

    // MTP capability, decided BEFORE the real load via a metadata-only peek
    // (no_alloc pattern from gguf-info): a nextn block that is not repurposed
    // as a router layer. On router models the trunk attends to those layers on
    // EVERY pass, so no separate draft context exists there — and
    // llama_init_from_model rejects LLAMA_CONTEXT_TYPE_MTP for them
    // (llama-context.cpp:4388).
    bool has_nextn = false;
    {
        llama_model_params mp_peek = llama_model_default_params();
        mp_peek.no_alloc  = true;
        mp_peek.load_mode = LLAMA_LOAD_MODE_NONE;
        llama_model * model_peek = llama_model_load_from_file(path, mp_peek);
        if (model_peek) {
            has_nextn = model_peek->hparams.n_layer_nextn > 0 && model_peek->hparams.router_layer < 0;
            llama_model_free(model_peek);
        } else {
            fprintf(stderr, "  pre-look: metadata load failed — assuming no MTP block\n");
        }
    }
    const bool measure_mtp = use_mtp && has_nextn;

    // REAL model load with the DEFAULT params: n_gpu_layers = -1 (all layers),
    // tensor_split = nullptr → llama_model's native auto-split places the layers
    // proportionally to each device's free VRAM — exactly as production does.
    // This single load serves both the measurement and the hparams for the table
    // below; no fake no_alloc probe is used.
    // load_mtp follows the MTP plan: without it the loader skips the nextn
    // tensors (llama-model.cpp:3928) and an MTP context would crash building
    // its graph on the missing eh_proj (qwen35.cpp:508).
    llama_model_params mp = llama_model_default_params();
    mp.load_mtp = measure_mtp;

    fprintf(stderr, "  loading: %s (all layers GPU, auto-split by free VRAM, load_mtp=%d)\n",
            path, (int)mp.load_mtp);
    const auto t_load0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        gguf_free(ctx);
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();

    // measurement: context at the common 16384-cell deployment size + decode
    // through the real weights. Every kernel the deployment will launch is
    // uploaded during this run — kernels only materialize on their first execution
    // (concept §17). The probe KV pool is accounted for in the overhead
    // decomposition below, so its size no longer skews the shader overhead.
    uint32_t n_ctx_measure = 0;
    bool mtp_measured = false;   // true when the nextn KV pool is in the measured delta
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_batch  = 2048;   // default; exercises the large-batch decode path
        cp.n_ubatch = ubatch_size;
        cp.type_k = type_k;   // measure what the deployment uses — not the f16 default
        cp.type_v = type_v;
        cp.n_seq_max = probe_parallel;  // parallel streams: scales the SSM state rows, not the KV pool
        cp.n_rs_seq  = probe_rs_seq;    // SSM rows per stream-rollback set = state_rows / n_seq_max
        cp.kv_unified = true;           // unified cache (the deployment model): one shared pool

        // tokenize the excerpt UP FRONT so the token count is visible while the
        // (slow) warmup decode runs — the user sees what the system is processing
        std::vector<int32_t> prompt_tokens(strlen(WARMUP_TEXT) + 8);
        const int n_tok = llama_tokenize(
            llama_model_get_vocab(model), WARMUP_TEXT, strlen(WARMUP_TEXT),
            prompt_tokens.data(), (int)prompt_tokens.size(), true, false);
        if (n_tok <= 0) {
            fprintf(stderr, "  measurement: tokenization failed\n");
            return 1;
        }
        prompt_tokens.resize(n_tok);

        // common deployment size — the warmup prompt fits with room to spare
        cp.n_ctx = 16384;
        n_ctx_measure = cp.n_ctx;   // capture for the overhead decomposition below

        // Guard 3: after the load, confirm the warmup KV pool fits in what's left.
        // The cache allocates n_ctx cells per stream on each GPU according to its
        // attention layers (x the MTP draft pool when measured); if any GPU can't
        // hold its share, abort with exact numbers instead of an opaque OOM at ctx init.
        {
            std::vector<dev_usage> usage = compute_dev_usage(model);
            std::vector<kv_rate_info> kv_rate = compute_kv_rates(model, usage, type_k, type_v, flash_attn);

            // Re-scan free VRAM AFTER the load to compare against what remains.
            bool fits = true;
            std::vector<dev_snapshot> scan_now;
            read_all_devices(scan_now);
            for (size_t i = 0; i < usage.size() && i < scan_now.size(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (!dev) continue;
                const auto t = ggml_backend_dev_type(dev);
                if (!(t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU)) continue;
                const uint64_t mtp_bpc = measure_mtp ? kv_rate[i].mtp_kv_bytes_per_cell : 0;   // MTP draft pool, same n_ctx
                const uint64_t need_i = (uint64_t)cp.n_ctx * (kv_rate[i].kv_bytes_per_cell + mtp_bpc);   // unified: one shared pool
                const uint64_t have_i = scan_now[i].free_bytes;
                if (need_i > have_i) {
                    fprintf(stderr, "error: warmup KV pool too large — GPU %zu needs %s for %u unified cells but only %s free\n",
                            i, gib_str(need_i).c_str(), cp.n_ctx, gib_str(have_i).c_str());
                    fits = false;
                }
            }
            if (!fits) {
                gguf_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        fprintf(stderr, "  measuring: n_ctx=%u n_batch=%u n_ubatch=%u parallel=%u checkpoints=%u mtp=%s (warmup: Hamlet excerpt, %d tokens)\n",
                (unsigned)cp.n_ctx, (unsigned)cp.n_batch, (unsigned)cp.n_ubatch,
                (unsigned)probe_parallel, (unsigned)state_rows, measure_mtp ? "yes" : "no", n_tok);
        const auto t_warmup0 = std::chrono::steady_clock::now();
        llama_context * ctx0 = llama_init_from_model(model, cp);
        if (!ctx0) {
            fprintf(stderr, "  measurement: context init failed\n");
        } else {
            warmup_decode(ctx0, prompt_tokens);

            // MTP draft context — auto-enabled when the model has a nextn block,
            // --no-mtp to skip (speculative.cpp pattern): same n_ctx as the target,
            // no recurrent state of its own, reads h_nextn from the main context.
            // It must still exist when scan_measure runs below, so its nextn KV
            // pool is part of the measured delta — freed only afterwards.
            llama_context * ctx_mtp = nullptr;
            if (measure_mtp) {
                llama_context_params cp_mtp = cp;
                cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                cp_mtp.n_rs_seq = 0;
                cp_mtp.ctx_other = ctx0;
                ctx_mtp = llama_init_from_model(model, cp_mtp);
                if (ctx_mtp) {
                    mtp_measured = true;   // the nextn KV pool now exists in the measured delta
                    // one single-token MTP graph run at the first position after
                    // the prompt (the main context decoded it during warmup):
                    // uploads every kernel an MTP deployment will launch
                    int seq_id = 0;   // non-const: batch.seq_id must point to a stable llama_seq_id
                    llama_batch mb;
                    mb.n_tokens = 1;
                    mb.token    = (llama_token *)  malloc(sizeof(llama_token));
                    mb.pos      = (llama_pos *)     malloc(sizeof(llama_pos));
                    mb.n_seq_id = (int32_t *)       malloc(sizeof(int32_t));
                    mb.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *));
                    mb.logits   = (int8_t *)        malloc(1);
                    mb.token[0]    = prompt_tokens.back();
                    mb.pos[0]      = (llama_pos)prompt_tokens.size();
                    mb.n_seq_id[0] = 1;
                    mb.seq_id[0]   = &seq_id;
                    mb.logits[0]   = 1;
                    const int rc = llama_decode(ctx_mtp, mb);
                    free(mb.token); free(mb.pos); free(mb.n_seq_id); free(mb.seq_id); free(mb.logits);
                    if (rc != 0) {
                        fprintf(stderr, "  mtp draft: decode failed (%d)\n", rc);
                    } else {
                        fprintf(stderr, "  mtp draft: context ready — nextn KV pool in the measured delta\n");
                    }
                } else {
                    fprintf(stderr, "  mtp draft: context init failed — nextn KV pool NOT in the measured delta\n");
                }
            }

            const double warmup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warmup0).count();
            fprintf(stderr, "  measurement: done; load %.1f s + warmup %.1f s (%.0f tok/s)\n",
                    load_s, warmup_s, warmup_s > 0 ? (double)n_tok / warmup_s : 0.0);
            read_all_devices(scan_measure);
            fix_cpu_free(scan_measure);
            if (ctx_mtp) {
                llama_free(ctx_mtp);
            }
            llama_free(ctx0);
        }
    }

    const auto & hp = model->hparams;
    const uint32_t n_layer_all   = hp.n_layer_all;
    const uint32_t n_layer_nextn = hp.n_layer_nextn;

    // per-layer MoE predicate: hparams experts OR expert tensors in the file
    const auto is_moe_layer = [&](uint32_t il) -> bool {
        return hp.n_ff_exp(il) > 0 ||
               (il < layer_has_exps.size() && layer_has_exps[il]);
    };

    // ---- capabilities -------------------------------------------------------
    const bool has_moe = hp.n_expert > 0;
    const bool has_mtp = n_layer_nextn > 0;
    bool has_swa = false, has_ssm = false;
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        has_swa |= hp.is_swa(il);
        has_ssm |= hp.is_recr(il);
    }

    // ---- parameters ---------------------------------------------------------
    fprintf(stderr, "\n");
    fprintf(stderr, "  file:       %s\n", path);
    fprintf(stderr, "  name:       %s\n", name ? name : "(unknown)");
    fprintf(stderr, "  arch:       %s\n", arch ? arch : llm_arch_name(model->arch));
    fprintf(stderr, "  layers:     %u (all)\n", (unsigned)n_layer_all);

    // ---- MoE scheme -----------------------------------------------------------
    // per-layer MoE membership = has routed experts (n_ff_exp(il) > 0);
    // dense runs and MoE runs are reported as compact layer ranges
    {
        bool any_dense = false;
        std::string scheme;
        for (uint32_t il = 0; il < n_layer_all; ) {
            const bool moe_layer = is_moe_layer(il);
            uint32_t j = il;
            while (j < n_layer_all && is_moe_layer(j) == moe_layer) ++j;
            char range[48];
            snprintf(range, sizeof(range), "%s%u-%u", scheme.empty() ? "" : " ", il, j - 1);
            scheme += moe_layer ? range : std::string("d") + range; // 'd' = dense run
            if (!moe_layer) any_dense = true;
            il = j;
        }
        if (has_moe) {
            fprintf(stderr, "  MoE:        yes (%u experts, top-%u)%s\n",
                    (unsigned)hp.n_expert, (unsigned)hp.n_expert_used_max(),
                    any_dense ? (" scheme: " + scheme).c_str() : "");
        } else {
            fprintf(stderr, "  MoE:        no\n");
        }
    }
    fprintf(stderr, "  SWA:        %s\n", has_swa ? "yes" : "no");
    fprintf(stderr, "  SSM:        %s\n", has_ssm ? "yes" : "no");
    fprintf(stderr, "  MTP:        %s\n", has_mtp ? "yes" : "no");
    fprintf(stderr, "  kv_type_k:  %s\n", ggml_type_name(type_k));
    fprintf(stderr, "  kv_type_v:  %s\n", ggml_type_name(type_v));
    fprintf(stderr, "  keep_free:  %.2f\n", keep_free);

    // the token embedding ALWAYS stays on the CPU (llama-model.cpp: "there is
    // very little benefit to offloading the input layer") — it is not part of
    // what gets distributed to GPUs, so it is shown here, outside the table.
    // EXCEPTION: tied models (no separate output.weight) get a second copy as
    // the lm_head on the output device — that copy appears as the LAST ROW of
    // the table (the bytes to distribute).
    if (embed_bytes > 0) {
        fprintf(stderr, "  embed(CPU): %lld\n", (long long)embed_bytes);
    }

    // device scan prints AFTER the model header block (Michael, 2026-11-12);
    // the measurements themselves were taken above, before any output.
    print_device_scan(scan_raw, scan_measure, model, n_ctx_measure, type_k, type_v, flash_attn,
                      probe_parallel, state_rows, mtp_measured, keep_free, layer_weights);

    // where the weights actually landed (GPU machines only)
    print_provisional_split(model);

    // ---- table layout: mtp + swa/ssm columns are capability-driven ----------
    // col 0 is sized to fit every label that will be printed (layer numbers,
    // embed/norm/output/TOTAL, and any exotic global tensor names)

    int col0_w = 7; // at least "layer"
    for (const auto & [tname, tsize] : other_tensors) {
        const int l = (int)tname.size() + 2;
        if (l > col0_w) col0_w = l;
    }

    std::vector<const char *> hdr;
    std::vector<int> widths;
    hdr.push_back("layer");       widths.push_back(col0_w); // label/layer column
    hdr.push_back("tensor size"); widths.push_back(NUM_COL_W);
    if (has_moe) { hdr.push_back("moe"); widths.push_back(5); }
    if (has_mtp) { hdr.push_back("mtp tensor"); widths.push_back(NUM_COL_W); }
    hdr.push_back("cell size");   widths.push_back(NUM_COL_W);
    if (has_mtp) { hdr.push_back("mtp cell");   widths.push_back(NUM_COL_W); }
    if (has_swa || has_ssm) { hdr.push_back("swa/ssm"); widths.push_back(NUM_COL_W); }

    // accounting invariant (code-enforced): emit_row accumulates every weight
    // byte it prints; the TOTAL row is built from exactly these accumulators,
    // so TOTAL always equals the sum of the visible rows — no hidden tensors.
    int64_t printed_base = 0, printed_mtp = 0;

    auto emit_row = [&](const char * label, int64_t tsize, int moe_flag,
                        int64_t mtens, int64_t csize, int64_t mcell,
                        int64_t state) {
        std::vector<int64_t> vals;
        vals.push_back(0);                              // slot 0 is the label (unused)
        vals.push_back(tsize);                          // tensor size        (always)
        if (has_moe) vals.push_back(moe_flag);          // moe flag           (optional)
        if (has_mtp) vals.push_back(mtens);             // mtp tensor         (optional)
        vals.push_back(csize);                          // cell size          (always)
        if (has_mtp) vals.push_back(mcell);             // mtp cell           (optional)
        if (has_swa || has_ssm) vals.push_back(state);  // swa/ssm            (optional, last)
        fprintf(stderr, "%s\n", data_row(label, vals, widths).c_str());
        printed_base += tsize;                          // accumulate what was printed
        printed_mtp  += mtens;
    };

    int64_t total_rate = 0, mtp_rate = 0;
    int64_t total_state = 0, mtp_state = 0;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s\n", box("┌", "┬", "┐", widths).c_str());
    fprintf(stderr, "%s\n", header_row(hdr, widths).c_str());
    fprintf(stderr, "%s\n", box("├", "┼", "┤", widths).c_str());

    for (uint32_t il = 0; il < n_layer_all; il++) {
        int64_t weights = layer_weights[il];
        const bool is_mtp_layer = il >= (n_layer_all - n_layer_nextn);

        // base deployment cost: without MTP, nextn layers are neither loaded
        // (load_mtp=false -> TENSOR_SKIP) nor executed — their weights and KV
        // appear in the mtp columns as the ADDITIONAL cost when MTP is enabled.

        int64_t tsize = 0, mtens = 0, csize = 0, mcell = 0, state = 0;

        // Skip layers without KV cache OR recurrent layers (hybrid models) —
        // they don't allocate unified pool KV cells
        if (hp.has_kv(il) && !hp.is_recr(il)) {
            // block-rounded rows: identical to what the allocator computes,
            // correct for quantized KV types (q8_0 packs 32 elements + scale)
            const uint32_t n_embd_k = hp.n_embd_k_gqa(il);
            const uint32_t n_embd_v = v_trans ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);

            int64_t k_bytes = ggml_row_size(type_k, n_embd_k);
            int64_t v_bytes = hp.is_mla() ? 0 : (int64_t)ggml_row_size(type_v, n_embd_v);
            int64_t cell_bytes = k_bytes + v_bytes;

            // one sliding window per SWA layer: fixed cost that always exists
            // ("we will always need one to hold the current state")
            state = hp.is_swa(il) ? (int64_t)hp.n_swa * cell_bytes : 0;

            if (is_mtp_layer) {
                mtens = weights; mcell = cell_bytes;
                mtp_rate += cell_bytes; mtp_state += state;
            } else {
                tsize = weights; csize = cell_bytes;
                total_rate += cell_bytes; total_state += state;
            }
        } else {
            // recurrent / no-KV layer: its weights deploy with the base model
            tsize = is_mtp_layer ? 0 : weights;
            mtens = is_mtp_layer ? weights : 0;

            if (hp.is_recr(il)) {
                // one snapshot row-set of this recurrent layer's R/S tensors:
                // the live row and every rollback snapshot cost the same row-set
                // each (llama-memory-recurrent.cpp allocates n_rows =
                // mem_size * (1 + n_rs_seq)); r/s are F32 per create_memory,
                // PLE layers add their conv-history row
                state  = (int64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_r());
                state += (int64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
                if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
                    state += (int64_t)ggml_row_size(GGML_TYPE_F32, hp.ple_conv_state());
                }
                if (is_mtp_layer) { mtp_state += state; } else { total_state += state; }
            }
        }

        char label[16];
        snprintf(label, sizeof(label), "%u", il);
        emit_row(label, tsize, is_moe_layer(il) ? 1 : 0, mtens, csize, mcell, state);
    }

    // globals after the last layer, in data flow order:
    // norm -> output (lm_head); any remaining non-blk tensors each get a row
    if (norm_bytes > 0) {
        emit_row("norm", norm_bytes, 0, 0, 0, 0, 0);
    }
    if (output_bytes > 0) {
        emit_row("output", output_bytes, 0, 0, 0, 0, 0);
    }
    for (const auto & [tname, tsize] : other_tensors) {
        emit_row(tname.c_str(), tsize, 0, 0, 0, 0, 0);
    }
    // tied lm_head: without a separate output tensor the loader duplicates the
    // token embedding as an OUTPUT tensor (llama-model-loader.cpp:1174) — that
    // GPU copy is part of what must be distributed, hence this last row.
    if (output_bytes == 0 && embed_bytes > 0) {
        emit_row("embed", embed_bytes, 0, 0, 0, 0, 0);
    }

    // ---- totals: exactly the sum of the rows printed above ------------------
    // the moe column total = count of base-model MoE layers (mtp excluded,
    // matching how the mtp columns are an ADDITIONAL cost)
    int64_t total_moe = 0;
    for (uint32_t il = 0; il < n_layer_all - n_layer_nextn; ++il) {
        if (is_moe_layer(il)) ++total_moe;
    }
    fprintf(stderr, "%s\n", box("├", "┼", "┤", widths).c_str());
    {
        std::vector<int64_t> vals;
        vals.push_back(0);                          // label slot (unused)
        vals.push_back(printed_base);               // sum of visible rows
        if (has_moe) vals.push_back(total_moe);     // moe layer count
        if (has_mtp) vals.push_back(printed_mtp);
        vals.push_back(total_rate);
        if (has_mtp) vals.push_back(mtp_rate);
        if (has_swa || has_ssm) vals.push_back(total_state + mtp_state);
        fprintf(stderr, "%s\n", data_row("TOTAL", vals, widths).c_str());
    }
    fprintf(stderr, "%s\n", box("└", "┴", "┘", widths).c_str());

    llama_model_free(model);
    gguf_free(ctx);
    llama_backend_free();
    return 0;
}
