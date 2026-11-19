// [FORK] Unified KV pool planning — the gpu-split algorithm, library-side.
//
// The application supplies its deployment parameters (model params, context
// params, an optional extra/MTP context); this library SOLVES the optimal
// tensor split and the maximum unified KV cache pool size against live
// device VRAM. All C++ lives here; callers speak the public C API in
// llama-ext.h only.
//
// Measurement-based (algorithm verified on rocky2 via tools/gpu-split): one
// REAL model load with llama_model's native free-VRAM auto-split as the
// provisional placement, a measurement context at a common deployment size,
// and a Hamlet warmup decode so every kernel the deployment will launch is
// uploaded to the devices — kernels only materialize on their first execution
// (no preload/prewarm API exists on any backend). Readback of the actual
// placement plus architecture-derived KV/SSM rates and a greedy bisection
// solver derive the plan from that single measurement: what is measured is
// exactly what the deployment will pay.
//
// Logging: detailed traces use LLAMA_LOG_PLANNING (visible without all of
// llama.cpp's debug logs); failures and conclusions use the standard levels.
// The consumer decides visibility.

#include "llama-impl.h"   // LLAMA_LOG_* macros
#include "llama-ext.h"    // public planning API declarations
#include "llama-model.h"  // llama_internal_get_tensor_map, llama_hparams

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "warmup-text.h"   // Hamlet excerpt (shared with tools/gpu-split)

// ============================================================================
// Options struct (public API, ABI versioned — see llama-ext.h for the rules)
// ============================================================================

LLAMA_API struct llama_kv_fit_params * llama_kv_fit_params_new(uint32_t min_version) {
    if (min_version > LLAMA_KV_FIT_PARAMS_VERSION) {
        fprintf(stderr, "error: libllama KV-fit planning is too old for the caller's "
                        "params version (%u > %u) — update libllama\n",
                (unsigned)min_version, (unsigned)LLAMA_KV_FIT_PARAMS_VERSION);
        exit(EXIT_FAILURE);   // a mismatched libllama is a deployment error, never a fallback
    }

    auto * p = new llama_kv_fit_params();
    p->vram_fraction  = 0.97;   // keep 3% of each device's VRAM free (gpu-split default)
    p->margins        = nullptr;
    p->extra          = nullptr;
    p->min_pool_cells = 32768;  // report deployments that cannot plan at least this many cells
    p->n_checkpoints  = 0;      // legacy: derive from context params when not set
    return p;
}

LLAMA_API void llama_kv_fit_params_free(struct llama_kv_fit_params * p) {
    delete p;
}

// ============================================================================
// Device scan
// ============================================================================

static const char * kvfit_dev_type_str(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_META: return "META";
        default:                            return "other";
    }
}

struct kvfit_dev_snapshot { size_t free_bytes = 0, total_bytes = 0; bool ok = false; };

static void kvfit_read_all_devices(std::vector<kvfit_dev_snapshot> & snaps) {
    const size_t ndevs = ggml_backend_dev_count();
    snaps.resize(ndevs);
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        ggml_backend_dev_memory(dev, &snaps[i].free_bytes, &snaps[i].total_bytes);
        snaps[i].ok = true;
    }
}

static bool kvfit_gpus_present() {
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

// ============================================================================
// Provisional placement readback
// ============================================================================
// Read back where a loaded model ACTUALLY placed its weights: every tensor
// carries its buffer, and the buffer's type maps back to its owning device
// via ggml_backend_buft_get_device(). CRITICAL: do NOT match on
// ggml_backend_dev_host_buffer_type(dev) — that host buft collides across
// GPUs sharing a backend (both ROCm0 and ROCm1 report the same HIP host buft).

struct kvfit_dev_usage { size_t bytes = 0; int first_layer = -1, last_layer = -1; std::vector<int> layers; };

static std::vector<kvfit_dev_usage> kvfit_compute_dev_usage(const llama_model * model) {
    const size_t ndevs = ggml_backend_dev_count();
    std::vector<kvfit_dev_usage> usage(ndevs);

    for (const auto & [tname, tensor] : llama_internal_get_tensor_map(model)) {
        if (!tensor || !tensor->buffer) continue;   // not on a device buffer
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
        ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
        if (!buft_dev) continue;
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
            if (!dev || dev != buft_dev) continue;
            usage[i].bytes += bytes;
            if (il >= 0) {
                usage[i].layers.push_back(il);
                if (usage[i].first_layer < 0 || il < usage[i].first_layer) usage[i].first_layer = il;
                if (il > usage[i].last_layer) usage[i].last_layer = il;
            }
            break;
        }
    }

    // de-dupe .layers: one entry per tensor occurrence was collected above, but
    // downstream rate math must see each layer index exactly once
    for (auto & u : usage) {
        std::sort(u.layers.begin(), u.layers.end());
        u.layers.erase(std::unique(u.layers.begin(), u.layers.end()), u.layers.end());
    }
    return usage;
}

// Provisional split readback table (gpu-split parity): where a measurement
// load ACTUALLY placed the weights. GPU rows only while any GPU is present —
// this is a GPU planning tool, CPU placement is noise for planning.
static void kvfit_print_provisional_split(const llama_model * model) {
    if (!kvfit_gpus_present()) return;

    const auto usage = kvfit_compute_dev_usage(model);
    LLAMA_LOG_PLANNING("\n      provisional split (as placed by the measurement load):\n\n");
    LLAMA_LOG_PLANNING("        %-4s %-6s %-10s %-14s  %s\n", "idx", "type", "layers", "weights", "name");
    for (size_t i = 0; i < usage.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU || t == GGML_BACKEND_DEVICE_TYPE_META) continue;

        char range[24];
        snprintf(range, sizeof(range),
                 usage[i].first_layer >= 0 ? "%d-%d" : "-",
                 usage[i].first_layer, usage[i].last_layer);
        char gib[32];
        snprintf(gib, sizeof(gib), "%.2f GiB",
                 (double)usage[i].bytes / (1024.0 * 1024.0 * 1024.0));
        LLAMA_LOG_PLANNING("        %-4zu %-6s %-10s %-14s  %s\n",
                           i, kvfit_dev_type_str(t), range, gib,
                           ggml_backend_dev_name(dev));
    }
}

// ============================================================================
// Architecture-derived per-device memory rates
// ============================================================================
// Per-device probe-memory rates (nextn layers counted separately — they serve
// the MTP draft context, not the main one). Indexed by backend device order.

struct kvfit_rate_info {
    uint64_t kv_bytes_per_cell = 0; int n_attn_layers = 0; // attention KV pool
    uint64_t ssm_bytes_per_row = 0; int n_ssm_layers = 0;  // recurrent state, per checkpoint row
    uint64_t mtp_kv_bytes_per_cell = 0; int n_mtp_layers = 0; // nextn KV pool (MTP draft)
    bool main_ctx = false;   // main context runs attention on this device (trunk layers)
    bool mtp_ctx  = false;   // MTP draft context runs on this device (nextn layers, when measured)
};

// Worst-case graph width sched_reserve builds the pp reservation with
// (llama-context.cpp:422-424 + :832): n_batch capped by n_ctx, n_ubatch
// min(n_batch, requested), reserve tokens = min(n_ctx, n_ubatch). The kq_mask
// graph input is sized from this, so pool-dependent compute buffers track it.
static uint32_t kvfit_n_tokens_reserve(const llama_context_params * cparams, uint32_t n_ctx_eff) {
    const uint32_t n_batch  = std::min(n_ctx_eff, (cparams->n_batch > 0) ? (uint32_t)cparams->n_batch : n_ctx_eff);
    const uint32_t n_ubatch = (cparams->n_ubatch > 0) ? (uint32_t)cparams->n_ubatch : n_batch;
    return std::min(n_ctx_eff, std::min(n_batch, n_ubatch));
}

// Per-context mask factor: one shared kq_mask graph input per context on the
// device, bytes per pool cell = n_tokens_reserve x element size of the mask.
static uint64_t kvfit_mask_bpc_per_ctx(const llama_context_params * cparams, uint32_t n_ctx_eff) {
    const bool fa = (cparams->flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED);
    return (uint64_t)kvfit_n_tokens_reserve(cparams, n_ctx_eff) * (fa ? 2u : 4u);
}

static std::vector<kvfit_rate_info> kvfit_compute_rates(const llama_model * model,
                                                        const std::vector<kvfit_dev_usage> & usage,
                                                        ggml_type type_k, ggml_type type_v,
                                                        bool flash_attn) {
    const auto & hp = model->hparams;
    const uint32_t n_layer_all   = hp.n_layer_all;
    const uint32_t n_layer_nextn = hp.n_layer_nextn;
    const bool v_t = !flash_attn;
    std::vector<kvfit_rate_info> out(usage.size());
    for (size_t i = 0; i < usage.size(); ++i) {
        for (int il : usage[i].layers) {
            if (il >= (int)n_layer_all) continue;
            // nextn layers: their KV pool lives in the MTP draft context
            if (il >= (int)(n_layer_all - n_layer_nextn)) {
                const uint32_t ek = hp.n_embd_k_gqa(il);
                const uint32_t ev = v_t ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);
                uint64_t cb = (uint64_t)ggml_row_size(type_k, ek);
                if (!hp.is_mla()) cb += (uint64_t)ggml_row_size(type_v, ev);
                out[i].mtp_kv_bytes_per_cell += cb;
                ++out[i].n_mtp_layers;
                out[i].mtp_ctx = true;   // MTP draft context computes on this device
                continue;
            }
            if (hp.is_recr(il)) {
                // recurrent state rows are always F32 (create_memory hardcodes them)
                uint64_t cb = (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_r());
                cb += (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
                if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
                    cb += (uint64_t)ggml_row_size(GGML_TYPE_F32, hp.ple_conv_state());
                }
                out[i].ssm_bytes_per_row += cb;
                ++out[i].n_ssm_layers;
            } else if (hp.has_kv(il)) {
                const uint32_t ek = hp.n_embd_k_gqa(il);
                const uint32_t ev = v_t ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);
                uint64_t cb = (uint64_t)ggml_row_size(type_k, ek);
                if (!hp.is_mla()) cb += (uint64_t)ggml_row_size(type_v, ev);
                out[i].kv_bytes_per_cell += cb;
                ++out[i].n_attn_layers;
                out[i].main_ctx = true;   // main context computes on this device
            }
        }
    }
    return out;
}

// ============================================================================
// KV pool solver
// ============================================================================
// Largest KV pool size (in 256-cell blocks) that a greedy layer placement
// affords on every GPU's budget, found by bisecting (log2 steps). Integer
// arithmetic throughout; every step is traced so the search and each
// decision can be verified.

struct kvfit_layer_cost {
    uint64_t weights = 0;       // tensor bytes for this layer (GGUF metadata)
    uint64_t kv_bpc  = 0;       // KV-cache bytes per cell, main pool (attention layers)
    uint64_t mtp_kv_bpc = 0;    // KV-cache bytes per cell, MTP draft pool (nextn layers)
    uint64_t ssm_state = 0;     // recurrent checkpoint state (rows x row-set), else 0
};

static std::vector<kvfit_layer_cost> kvfit_compute_layer_costs(const llama_model * model,
                                                               const std::map<int, int64_t> & layer_weights,
                                                               ggml_type type_k, ggml_type type_v,
                                                               bool flash_attn, uint32_t state_rows) {
    const auto & hp = model->hparams;
    const bool v_t = !flash_attn;
    const uint32_t n_nextn_first = hp.n_layer_all - hp.n_layer_nextn;   // first nextn index
    std::vector<kvfit_layer_cost> out(hp.n_layer_all);
    for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
        auto it = layer_weights.find((int)il);
        if (it != layer_weights.end()) {
            out[il].weights = (uint64_t)it->second;
        }
        if (hp.is_recr(il)) {
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

// Greedy walk shared by the solver and the final layout: each GPU g takes
// layers [first_layer[g], first_layer[g+1)) while they fit; the last GPU gets
// everything remaining. Takes are gated on the FULL charge (layer weights +
// cell costs + this device's compute-buffer rate, constant per GPU) so an
// intermediate GPU never ends up over budget with buffers included. Returns
// per-GPU start indices (size = budgets.size() + 1 sentinel so range queries
// are one subtraction).
static std::vector<size_t> kvfit_greedy_walk(uint32_t n_blocks,
                                             const std::vector<kvfit_layer_cost> & layers,
                                             const std::vector<uint64_t> & budgets,
                                             const std::vector<uint64_t> & buf_bpc) {
    const uint64_t cells = (uint64_t)n_blocks * 256;
    const size_t nl = layers.size();
    std::vector<size_t> first(budgets.size() + 1, nl);

    size_t il = 0;
    for (size_t g = 0; g < budgets.size(); ++g) {
        first[g] = il;
        if (g + 1 == budgets.size()) break;   // last GPU takes everything remaining
        const uint64_t buffers_g = (g < buf_bpc.size() ? buf_bpc[g] : 0) * cells;
        if (buffers_g > budgets[g]) break;   // cannot afford its buffers: takes no layers
        const uint64_t capacity = budgets[g] - buffers_g;   // layer budget after buffers
        uint64_t used = 0;
        while (il < nl) {
            const uint64_t cost = layers[il].weights + cells * (layers[il].kv_bpc
                                                               + layers[il].mtp_kv_bpc)
                                + layers[il].ssm_state;
            if (used + cost > capacity) break;   // hand over at il
            used += cost;
            ++il;
        }
    }
    return first;
}

// Print the per-GPU cost breakdown through print_table: fixed base costs and
// per-cell rates. Used for both the provisional split (before the bisection)
// and the final plan — the caller supplies per-GPU arrays in its own placement.
// One stream write per line, straight to stderr (same convention as the
// measurement breakdowns above — a log call would prefix every row).
static void kvfit_print_cost_breakdown(const char * title,
                                       const std::vector<std::string> & names,
                                       const std::vector<uint64_t> & wts,
                                       const std::vector<uint64_t> & base,
                                       const std::vector<uint64_t> & ssm_state,
                                       const std::vector<uint64_t> & kv_bpc,
                                       const std::vector<uint64_t> & mtp_bpc,
                                       const std::vector<uint64_t> & buf_bpc,
                                       const std::vector<uint64_t> & mtp_buf_bpc) {
    fprintf(stderr, "\n      %s\n", title);

    // columns: fixed base costs, then the per-cell rates (buffers kept separate),
    // then their sum
    static const char * hdrs[8] = { "weights", "base-buffer", "ssm-state",
                                    "kv/cell", "mtp/cell", "buffers/cell",
                                    "mtp-buffers/cell", "cell-total" };
    const int n_rows = (int)names.size() + 1;   // one per GPU + TOTAL

    std::vector<std::string> keys;
    keys.reserve(n_rows);
    for (const auto & n : names) keys.push_back(n);
    keys.push_back("TOTAL");

    std::vector<long> flat((size_t)n_rows * 8, 0);
    const auto get = [](const std::vector<uint64_t> & v, size_t g) -> long {
        return (g < v.size()) ? (long)v[g] : 0;
    };
    for (size_t g = 0; g < names.size(); ++g) {
        const size_t o = g * 8;
        flat[o + 0] = get(wts, g);
        flat[o + 1] = get(base, g);
        flat[o + 2] = get(ssm_state, g);
        flat[o + 3] = get(kv_bpc, g);
        flat[o + 4] = get(mtp_bpc, g);
        flat[o + 5] = get(buf_bpc, g);
        flat[o + 6] = get(mtp_buf_bpc, g);
        flat[o + 7] = flat[o + 3] + flat[o + 4] + flat[o + 5] + flat[o + 6];
    }
    // TOTAL row: column sums (cell-total = sum of the four per-cell columns)
    for (int c = 0; c < 8; ++c) {
        long t = 0;
        for (size_t g = 0; g < names.size(); ++g) t += flat[g * 8 + c];
        flat[(size_t)(names.size()) * 8 + c] = t;
    }

    std::vector<const long *> row_ptrs(n_rows);
    for (int r = 0; r < n_rows; ++r) row_ptrs[r] = flat.data() + (size_t)r * 8;
    std::vector<const char *> keys_arr(n_rows);
    for (int r = 0; r < n_rows; ++r) keys_arr[r] = keys[r].c_str();

    print_table(stderr, n_rows, 8, hdrs, keys_arr.data(), "", row_ptrs.data(), nullptr, true);
}

// Try placing n 256-cell blocks with the candidate placement from the shared
// greedy walk (buffers-gated), then check every GPU's FULL charge against its
// budget: layer weights + pool cells + compute buffers (+ global tensors on
// the last GPU). The last GPU has no choice — all remaining layers are its —
// so its range total decides the candidate outright.
//
// buf_bpc[g]: measured compute-buffer bytes per pool cell on GPU g (hi-lo
// difference of the memory breakdown). Always charged, on every GPU,
// independent of layer distribution — the graph reserves them per device.
// The caller passes main + MTP draft rates combined when a device hosts the
// MTP draft context's own compute buffers.
static bool kvfit_try_place(uint32_t n_blocks,
                            const std::vector<kvfit_layer_cost> & layers,
                            const std::vector<uint64_t> & budgets,
                            const std::vector<std::string> & names,
                            uint64_t last_gpu_extra,
                            const std::vector<uint64_t> & buf_bpc) {
    const uint64_t cells = (uint64_t)n_blocks * 256;
    LLAMA_LOG_PLANNING("\n        testing with n = %-5u blocks (%llu cells)\n",
                       n_blocks, (unsigned long long)cells);

    // prefix sums over layers: O(1) range queries for each cost component.
    const size_t nl = layers.size();
    std::vector<uint64_t> p_wss(nl + 1, 0), p_kv(nl + 1, 0), p_mtp(nl + 1, 0),
                         p_wts(nl + 1, 0), p_ssm(nl + 1, 0);
    for (size_t i = 0; i < nl; ++i) {
        p_wss[i + 1] = p_wss[i] + layers[i].weights + layers[i].ssm_state;
        p_kv[i + 1]  = p_kv[i]  + layers[i].kv_bpc;         // main KV pool cells
        p_mtp[i + 1] = p_mtp[i] + layers[i].mtp_kv_bpc;     // MTP draft pool cells (nextn)
        p_wts[i + 1] = p_wts[i] + layers[i].weights;
        p_ssm[i + 1] = p_ssm[i] + layers[i].ssm_state;
    }

    // candidate placement for this n: the shared greedy walk, gated on every
    // device's compute-buffer charge (constant per GPU — no ctx-flag gating)
    const std::vector<size_t> walk_first = kvfit_greedy_walk(n_blocks, layers, budgets, buf_bpc);

    for (size_t g = 0; g < budgets.size(); ++g) {
        const bool is_last = (g + 1 == budgets.size());
        const size_t first_layer = walk_first[g];
        const size_t last_layer  = is_last ? nl : walk_first[g + 1];   // exclusive; last owns the rest

        const uint64_t r_bufs = cells * ((g < buf_bpc.size()) ? buf_bpc[g] : 0);   // always, every GPU
        uint64_t used = 0;
        if (is_last) {
            // last GPU: no choice — all remaining layers are its, plus the global
            // tensors (norm, output); the range total against its budget decides
            // the candidate outright
            used = (p_wss[nl] - p_wss[first_layer])
                 + cells * (p_kv[nl] - p_kv[first_layer])
                 + cells * (p_mtp[nl] - p_mtp[first_layer])
                 + r_bufs
                 + last_gpu_extra;
        } else if (last_layer > first_layer) {
            used = (p_wss[last_layer] - p_wss[first_layer])
                 + cells * (p_kv[last_layer] - p_kv[first_layer])
                 + cells * (p_mtp[last_layer] - p_mtp[first_layer])
                 + r_bufs;
        } else {
            used = r_bufs;   // empty range still pays its compute buffers
        }

        const bool has_layers = is_last || (last_layer > first_layer);
        const size_t shown_last = has_layers ? (is_last ? nl - 1 : last_layer - 1) : 0;
        std::string range = has_layers ? "" : "(none)";
        if (has_layers) {
            char rb[64];
            snprintf(rb, sizeof rb, "%zu..%zu", first_layer, shown_last);
            range = rb;
        }

        const uint64_t wts = (has_layers ? (p_wts[shown_last + 1] - p_wts[first_layer]) : 0)
                           + (is_last ? last_gpu_extra : 0);   // globals ride the last GPU's tensors
        const uint64_t ssm = has_layers ? (p_ssm[shown_last + 1] - p_ssm[first_layer]) : 0;
        const uint64_t kv_bpc  = has_layers ? (p_kv[shown_last + 1] - p_kv[first_layer]) : 0;
        const uint64_t mtp_bpc = has_layers ? (p_mtp[shown_last + 1] - p_mtp[first_layer]) : 0;
        const uint64_t cache   = cells * kv_bpc;
        const uint64_t mtp     = cells * mtp_bpc;
        const uint64_t buffers = r_bufs;

        const bool fits = (used <= budgets[g]);   // overflow-safe: all terms fit in uint64_t

        LLAMA_LOG_PLANNING("          %s: layers = %s\n"
                           "            tensors   = %llu\n"
                           "            swa/ssm   = %llu\n"
                           "            cache     = cells(%llu) × bpc(%llu) = %llu\n"
                           "            mtp       = cells(%llu) × bpc(%llu) = %llu\n"
                           "            buffers   = cells(%llu) × bpc(%llu) = %llu\n",
                           names[g].c_str(), range.c_str(),
                           (unsigned long long)wts, (unsigned long long)ssm,
                           (unsigned long long)cells, (unsigned long long)kv_bpc,
                           (unsigned long long)cache,
                           (unsigned long long)cells, (unsigned long long)mtp_bpc,
                           (unsigned long long)mtp,
                           (unsigned long long)cells,
                           (unsigned long long)((g < buf_bpc.size()) ? buf_bpc[g] : 0),
                           (unsigned long long)buffers);
        if (fits) {
            LLAMA_LOG_PLANNING("            used      = tensors(%llu) + swa/ssm(%llu) + cache(%llu) + mtp(%llu) + buffers(%llu) = %llu\n"
                               "            not used  = budget(%llu) - used(%llu) = %llu\n",
                               (unsigned long long)wts, (unsigned long long)ssm,
                               (unsigned long long)cache, (unsigned long long)mtp,
                               (unsigned long long)buffers, (unsigned long long)used,
                               (unsigned long long)budgets[g], (unsigned long long)used,
                               (unsigned long long)(budgets[g] - used));
        } else {
            LLAMA_LOG_PLANNING("            used      = tensors(%llu) + swa/ssm(%llu) + cache(%llu) + mtp(%llu) + buffers(%llu) = %llu\n"
                               "            over limit = used(%llu) - budget(%llu) = %llu\n",
                               (unsigned long long)wts, (unsigned long long)ssm,
                               (unsigned long long)cache, (unsigned long long)mtp,
                               (unsigned long long)buffers, (unsigned long long)used,
                               (unsigned long long)used, (unsigned long long)budgets[g],
                               (unsigned long long)(used - budgets[g]));
        }

        if (is_last) return fits;
    }
    return true;   // all layers placed — candidate fits
}

// Bisect for the largest fitting n of 256-cell blocks: at most log2(n_max)
// steps. Returns the winning block count (0 = nothing fits).
static uint32_t kvfit_solve(uint32_t n_max,
                            const std::vector<kvfit_layer_cost> & layers,
                            const std::vector<uint64_t> & budgets,
                            const std::vector<uint64_t> & reserves,
                            const std::vector<std::string> & names,
                            uint64_t last_gpu_extra,
                            const std::vector<uint64_t> & buf_bpc) {
    LLAMA_LOG_PLANNING("\n        kv pool solver: bisect n of 256-cell blocks (integer arithmetic)\n");
    if (n_max == 0 || layers.empty() || budgets.empty()) {
        LLAMA_LOG_PLANNING("          n_max = 0 — nothing to solve\n");
        return 0;
    }
    {
        // same layout as gpu-split's solver header: "%s = <value>" per device,
        // comma+space separated, 14-wide right-aligned values (up to 99 TB).
        // Two rows: the budgets each candidate is checked against, and the
        // reserves kept back from each device (total x keep_free [+ margin]).
        auto row = [&](const char * label, const std::vector<uint64_t> & vals) {
            char prefix[24];
            snprintf(prefix, sizeof prefix, "          %-8s", label);
            std::string line = prefix;
            for (size_t g = 0; g < vals.size(); ++g) {
                char b[64];
                snprintf(b, sizeof b, "%s%s = %14llu",
                         (g ? ", " : " "), names[g].c_str(), (unsigned long long)vals[g]);
                line += b;
            }
            LLAMA_LOG_PLANNING("%s\n", line.c_str());
        };
        row("budget:", budgets);
        row("reserve:", reserves);
    }
    uint32_t low = 0, high = n_max;   // bisection over the whole range
    while (low < high) {
        const uint32_t mid = low + (high - low + 1) / 2;   // rounds UP -> monotonic progress
        if (kvfit_try_place(mid, layers, budgets, names, last_gpu_extra, buf_bpc)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    LLAMA_LOG_PLANNING("          OPTIMAL: n=%-7u blocks = %llu kv cells\n",
                       low, (unsigned long long)low * 256ULL);
    return low;
}

// ============================================================================
// GGUF metadata (device-invariant weight accounting)
// ============================================================================
// Per-layer weight bytes, global tensors and the total model size — pure GGUF
// contents, read without loading the model. The solver charges these per layer;
// Guard 2 uses the total.

struct kvfit_gguf_info {
    std::string name;                          // general.name (model header)
    std::string arch;                          // general.architecture (model header)
    std::map<int, int64_t> layer_weights;
    int64_t norm_bytes   = 0;
    int64_t output_bytes = 0;
    int64_t embed_bytes  = 0;
    int64_t model_bytes  = 0;   // all tensors (Guard 2)
};

static bool kvfit_read_gguf(const char * path, kvfit_gguf_info & info) {
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        LLAMA_LOG_ERROR("kvfit: failed to open GGUF metadata for %s\n", path);
        return false;
    }

    const char * arch = nullptr;
    int64_t id_arch = gguf_find_key(ctx, "general.architecture");
    if (id_arch >= 0) arch = gguf_get_val_str(ctx, id_arch);
    info.arch = arch ? arch : "";

    const char * name = nullptr;
    int64_t id_name = gguf_find_key(ctx, "general.name");
    if (id_name >= 0) name = gguf_get_val_str(ctx, id_name);
    info.name = name ? name : "";

    // nextn block indices: blk.nextn.<i> maps to hp layer n_layer + i — read the
    // trunk's n_layer from GGUF metadata since this parse predates the model load
    int64_t n_layer_meta = 0;
    if (arch) {
        const std::string nk = std::string(arch) + ".n_layer";
        int64_t id_nlayer = gguf_find_key(ctx, nk.c_str());
        if (id_nlayer >= 0) {
            n_layer_meta = (int64_t)gguf_get_val_u32(ctx, id_nlayer);
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(ctx, tid);
        const int64_t tsize = (int64_t)gguf_get_tensor_size(ctx, tid);
        info.model_bytes += tsize;

        if (!strcmp(tname, "token_embd.weight")) { info.embed_bytes += tsize; continue; }
        if (!strcmp(tname, "output_norm.weight")) { info.norm_bytes += tsize; continue; }
        if (!strcmp(tname, "output.weight"))      { info.output_bytes += tsize; continue; }

        int il = -1;
        const char * p_nextn = strstr(tname, "blk.nextn.");
        if (p_nextn) {
            int nin = 0;
            if (n_layer_meta > 0 && sscanf(p_nextn + strlen("blk.nextn."), "%d", &nin) == 1) {
                il = (int)n_layer_meta + nin;
            }
        } else if (strstr(tname, "blk.")) {
            const char * p = strstr(tname, "blk.");
            sscanf(p + 4, "%d", &il);
        }
        if (il >= 0) {
            info.layer_weights[il] += tsize;
        }
    }

    gguf_free(ctx);
    return true;
}

// ============================================================================
// Warmup decode
// ============================================================================
// The measurement load must execute the real deployment's compute paths so
// every kernel the run will launch is uploaded to the devices — only a live
// decode measures what a live run pays. The prompt is ~20K chars of Hamlet
// (warmup-text.h): natural-language tokens with high diversity, so a MoE
// router exercises many experts and their kernels load too. Returns the
// number of tokens decoded in prefill (0 on failure).

static int kvfit_warmup_decode(llama_context * ctx, const std::vector<int32_t> & tokens) {
    const int n_tok = (int)tokens.size();
    if (n_tok <= 0) {
        return 0;
    }

    // batch construction follows fllama.c exactly (and it does not crash):
    // raw mallocs, a flat seq_id pointer array without per-token inner
    // allocations, and manual frees.
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
            LLAMA_LOG_ERROR("kvfit: warmup prefill decode failed at token %d\n", start);
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
            LLAMA_LOG_ERROR("kvfit: warmup generate decode %d failed\n", g);
            break;
        }
    }

    return n_tok;
}

// ============================================================================
// Verification run: prove the plan works at runtime
// ============================================================================
// Load the model with the COMPUTED split and measure free VRAM at two pool
// sizes: 16384 (probe) and the planned pool. This proves the placement fits
// and shows the actual memory state. Both tests share ONE model load — only
// n_ctx differs between them, so no second load is needed.

static void kvfit_verify_plan(const char * path,
                              const llama_model_params * mparams_base,
                              const llama_context_params * cparams_base,
                              bool measure_mtp,
                              const std::vector<int> & first_layer,
                              uint32_t n_cells_planned) {
    // Build tensor_layer_bounds array from the plan's placement
    const size_t nd = first_layer.size();
    int32_t * bounds = new int32_t[nd];
    for (size_t g = 0; g < nd; ++g) {
        bounds[g] = first_layer[g];
    }

    // Model load with the COMPUTED split
    llama_model_params mp = *mparams_base;
    mp.n_gpu_layers  = -1;              // all layers on GPUs
    mp.tensor_split  = nullptr;         // superseded by layer bounds
    mp.tensor_layer_bounds = bounds;    // the plan's placement
    mp.load_mtp      = measure_mtp;

    LLAMA_LOG_PLANNING("\n      verification: loading with computed split (first_layer per GPU):\n");
    for (size_t g = 0; g < nd; ++g) {
        const int a = first_layer[g];
        LLAMA_LOG_PLANNING("        GPU %zu: starts at layer %d\n", g, a);
    }

    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        LLAMA_LOG_ERROR("kvfit_verify_plan: failed to load model with computed split\n");
        delete[] bounds;
        return;
    }

    // Hamlet warmup text (reuse the same prompt as the measurement)
    std::vector<int32_t> prompt_tokens(strlen(WARMUP_TEXT) + 8);
    const int n_tok = llama_tokenize(
            llama_model_get_vocab(model), WARMUP_TEXT, strlen(WARMUP_TEXT),
            prompt_tokens.data(), (int)prompt_tokens.size(), true, false);
    if (n_tok <= 0) {
        LLAMA_LOG_ERROR("kvfit_verify_plan: warmup tokenization failed\n");
        llama_model_free(model);
        delete[] bounds;
        return;
    }
    prompt_tokens.resize(n_tok);

    // Helper lambda to run one verification test at a given context size
    auto run_verification_test = [&](const char * test_name, uint32_t n_ctx) {
        LLAMA_LOG_PLANNING("\n      verification %s: context size = %u\n", test_name, n_ctx);

        llama_context_params cp = *cparams_base;
        cp.kv_unified = true;
        cp.n_ctx      = n_ctx;
        cp.n_kv_cache = 0;   // unified mode: pool follows n_ctx

        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            LLAMA_LOG_ERROR("kvfit_verify_plan: context init failed for %s\n", test_name);
            return;
        }

        // MTP draft context when measuring it (same pattern as kvfit_measure)
        llama_context * ctx_mtp = nullptr;
        if (measure_mtp && llama_model_n_layer_nextn(model) > 0) {
            llama_context_params cp_mtp = cp;
            cp_mtp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
            cp_mtp.n_rs_seq  = 0;
            cp_mtp.ctx_other = ctx;
            ctx_mtp = llama_init_from_model(model, cp_mtp);
            if (!ctx_mtp) {
                LLAMA_LOG_WARN("kvfit_verify_plan: MTP context init failed\n");
            }
        }

        // Warmup decode to trigger all kernel allocations
        const int decoded = kvfit_warmup_decode(ctx, prompt_tokens);
        if (decoded == 0) {
            LLAMA_LOG_ERROR("kvfit_verify_plan: warmup decode failed for %s\n", test_name);
        }

        // MTP draft decode when present
        if (ctx_mtp && decoded > 0) {
            const int32_t n_embd = llama_model_n_embd(model);
            float * state = (float *)calloc((size_t)n_embd, sizeof(float));

            struct llama_batch_ext * mb = llama_batch_ext_init(ctx_mtp);
            const int32_t idx = llama_batch_ext_add_token(mb, 0, prompt_tokens.back());
            const llama_pos pos = (llama_pos)prompt_tokens.size();
            llama_batch_ext_set_pos(mb, idx, &pos);
            struct llama_embd embd_state = { state, 1, (size_t)n_embd };
            llama_batch_ext_set_embd_state(mb, idx, embd_state);
            llama_batch_ext_set_output_logits(mb, idx, true);

            const int rc = llama_process(ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, mb);
            llama_batch_ext_free(mb);
            free(state);
            if (rc != 0) {
                LLAMA_LOG_ERROR("kvfit_verify_plan: MTP draft decode failed (%d)\n", rc);
            }
        }

        // Show per-GPU free VRAM after everything is loaded + warmed up
        std::vector<kvfit_dev_snapshot> snaps;
        kvfit_read_all_devices(snaps);
        LLAMA_LOG_PLANNING("        free VRAM after %s (post-warmup):\n", test_name);
        for (size_t i = 0; i < snaps.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !snaps[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (t == GGML_BACKEND_DEVICE_TYPE_CPU) continue;   // GPU tool: never show CPU
            LLAMA_LOG_PLANNING("          %-4zu %-6s free=%14llu  %s\n",
                               i, kvfit_dev_type_str(t),
                               (unsigned long long)snaps[i].free_bytes,
                               ggml_backend_dev_name(dev));
        }

        // Cleanup this test's contexts
        if (ctx_mtp) llama_free(ctx_mtp);
        llama_free(ctx);
    };

    // Test 1: at probe context size (16384) — matches the measurement baseline
    run_verification_test("probe", 16384);

    // Test 2: at the planned pool size — this is what will actually deploy
    run_verification_test("planned", n_cells_planned);

    // Cleanup
    llama_model_free(model);
    delete[] bounds;
}

// ============================================================================
// Measurement run
// ============================================================================
// One REAL model load (llama_model's native free-VRAM auto-split — the exact
// provisional placement production would do), a measurement context, and the
// Hamlet warmup decode. The measured device deltas cover: weights + probe KV
// pools + SSM state + kernels/shaders + compute buffers — everything a live
// deployment pays before its own pool exists. Returns false on any failure.
// Results come back through the out parameters; all model/contexts are freed
// before returning (the measurement is self-contained).

// Read llama_get_memory_breakdown(ctx) and return its .compute (temporary
// compute-buffer bytes) per device, indexed by backend device order. This is
// the graph-simulator's own account of the pool-sized compute buffers — the
// ground truth our old kq_mask formula had to guess at. ..._Host bufts are
// printed as their own rows but live in system RAM — they are NEVER folded
// into a GPU rate (that contamination doubled the ROCm0 rate on rocky2).
static std::vector<uint64_t> kvfit_compute_by_device(const llama_context * ctx, const char * label) {
    const size_t ndevs = ggml_backend_dev_count();
    std::vector<uint64_t> compute(ndevs, 0);
    fprintf(stderr, "  breakdown [%s] (from llama_get_memory_breakdown):\n", label);
    for (auto & [buft, data] : llama_get_memory_breakdown(ctx)) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) continue;
        for (size_t i = 0; i < ndevs; ++i) {
            if (ggml_backend_dev_get(i) == dev) {
                fprintf(stderr, "    buft_name='%s'  compute=%llu (%.2f GiB)  context=%llu (%.2f GiB)\n",
                        ggml_backend_buft_name(buft), 
                        (unsigned long long)data.compute, 
                        (double)data.compute / (1024.0 * 1024.0 * 1024.0),
                        (unsigned long long)data.context,
                        (double)data.context / (1024.0 * 1024.0 * 1024.0));
                if (!ggml_backend_buft_is_host(buft)) {   // host RAM never counts against GPU rates
                    compute[i] += data.compute;
                }
                break;
            }
        }
    }
    return compute;
}

struct kvfit_measure_result {
    bool ok = false;
    std::vector<kvfit_dev_snapshot> scan_measure;  // after load + warmup (+ MTP)
    std::vector<kvfit_dev_usage> usage;            // readback: weights/layers per device
    std::vector<kvfit_rate_info> rates;             // KV/SSM/MTP rates per device
    bool mtp_measured = false;                      // nextn KV pool is in the measured delta
    uint32_t n_ctx_measure = 0;                     // measurement pool size (decomposition input)
    // Measured compute-buffer bytes-per-pool-cell rate per device, from
    // llama_get_memory_breakdown() at two pool sizes. Replaces the old kq_mask
    // guess — the rate is context-size-independent, so it holds for any pool.
    std::vector<uint64_t> compute_rate_main;        // main-context rate (per device)
    
    // MTP draft context compute rate per cell (per device), measured with the
    // hi-lo difference method — the same pool-proportional approach as
    // compute_rate_main, on the MTP draft context's own buffers. When a device
    // hosts nextn layers it allocates these SEPARATELY from the main context's
    // compute buffers; both rates are charged in the solver.
    std::vector<uint64_t> mtp_rate;                 // mtp-context rate (per device)
};

static bool kvfit_measure(const char * path,
                          const llama_model_params * mparams,
                          const llama_context_params * cparams,
                          bool measure_mtp,
                          kvfit_measure_result & r) {
    // raw-device scan is the caller's job (Guard 2 + budget denominators); it
    // prints once and its snapshot feeds the delta decomposition below.

    // REAL model load: the caller's params define how production loads (mmap
    // etc.); override placement so llama_model's native auto-split-by-free-VRAM
    // decides the provisional layout — exactly as a from-scratch deployment does.
    llama_model_params mp = *mparams;
    mp.n_gpu_layers  = -1;      // all layers GPU
    mp.tensor_split  = nullptr; // -> native free-VRAM proportional split
    mp.tensor_layer_bounds = nullptr; // provisional load: never a planned placement
    mp.load_mtp      = measure_mtp;

    LLAMA_LOG_PLANNING("\n      loading: %s (auto-split by free VRAM, load_mtp=%d)\n",
                       path, (int)mp.load_mtp);
    const auto t_load0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        LLAMA_LOG_ERROR("kvfit: failed to load model %s\n", path);
        return false;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    kvfit_print_provisional_split(model);

    // measurement context at a common deployment size. n_kv_cache stays 0 with
    // kv_unified: the pool is then PAD(n_ctx) cells — its size is subtracted in
    // the decomposition below, so it no longer skews the shader overhead.
    // Deployment shape (batch sizes, KV types, streams, rollback rows) comes
    // from the caller's production params — measure what deployment will pay.
    llama_context_params cp = *cparams;
    cp.kv_unified = true;
    cp.n_ctx      = 16384;   // common deployment size (warmup prompt fits with room)
    cp.n_kv_cache = 0;
    const uint32_t n_ctx_measure = cp.n_ctx;

    std::vector<int32_t> prompt_tokens(strlen(WARMUP_TEXT) + 8);
    const int n_tok = llama_tokenize(
            llama_model_get_vocab(model), WARMUP_TEXT, strlen(WARMUP_TEXT),
            prompt_tokens.data(), (int)prompt_tokens.size(), true, false);
    if (n_tok <= 0) {
        LLAMA_LOG_ERROR("kvfit: warmup tokenization failed\n");
        llama_model_free(model);
        return false;
    }
    prompt_tokens.resize(n_tok);

    // Guard 3: confirm the warmup KV pool fits in what's left after the load —
    // exact numbers instead of an opaque OOM at context init.
    {
        std::vector<kvfit_dev_usage> usage_now = kvfit_compute_dev_usage(model);
        std::vector<kvfit_rate_info> rates_now = kvfit_compute_rates(model, usage_now, cp.type_k, cp.type_v, cp.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED);

        std::vector<kvfit_dev_snapshot> scan_now;
        kvfit_read_all_devices(scan_now);
        for (size_t i = 0; i < scan_now.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !scan_now[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (!(t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU)) continue;
            const uint64_t mtp_bpc = measure_mtp ? rates_now[i].mtp_kv_bytes_per_cell : 0;
            const uint64_t need_i = (uint64_t)n_ctx_measure * (rates_now[i].kv_bytes_per_cell + mtp_bpc);
            if (need_i > scan_now[i].free_bytes) {
                LLAMA_LOG_ERROR("kvfit: warmup KV pool too large — GPU %zu needs %llu B for %u unified cells but only %llu B free\n",
                                i, (unsigned long long)need_i, n_ctx_measure,
                                (unsigned long long)scan_now[i].free_bytes);
                llama_model_free(model);
                return false;
            }
        }
    }

    LLAMA_LOG_PLANNING("      measuring: n_ctx=%u n_batch=%u n_ubatch=%u seq_max=%u rs_seq=%u mtp=%s (warmup: Hamlet excerpt, %d tokens)\n",
                       cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max, cp.n_rs_seq,
                       measure_mtp ? "yes" : "no", n_tok);

    const auto t_warmup0 = std::chrono::steady_clock::now();
    llama_context * ctx0 = llama_init_from_model(model, cp);
    if (!ctx0) {
        LLAMA_LOG_ERROR("kvfit: measurement context init failed\n");
        llama_model_free(model);
        return false;
    }

    // ---- measure the pool-dependent compute-buffer RATE (per device) -----------
    // The compute buffers grow linearly with the pool (verified on rocky2); the
    // rate is context-size-independent. Read .compute at this pool and at 2x via
    // llama_get_memory_breakdown, free the second, store the rate for the solver.
    {
        const size_t ndevs = ggml_backend_dev_count();
        const uint32_t lo_cells  = n_ctx_measure;
        const uint32_t hi_cells  = 2u * lo_cells;
        std::vector<uint64_t> compute_lo = kvfit_compute_by_device(ctx0, "main ctx, lo pool");   // main ctx at lo
        
        // Create hi-pool context for rate measurement (keep alive for MTP measurement too)
        llama_context_params cp_hi = cp;
        cp_hi.n_ctx = hi_cells;
        llama_context * ctx_hi_main = nullptr;
        if (llama_context * tmp = llama_init_from_model(model, cp_hi)) {
            ctx_hi_main = tmp;  // Keep alive!
        } else {
            LLAMA_LOG_WARN("kvfit: hi-pool context for compute-rate init failed — falling back to 0 rate\n");
        }

        // Measure main context at hi pool
        std::vector<uint64_t> compute_hi;
        if (ctx_hi_main) {
            compute_hi = kvfit_compute_by_device(ctx_hi_main, "main ctx, hi pool");   // main ctx at hi
        }

        // Calculate main rate from hi-lo difference
        r.compute_rate_main.assign(ndevs, 0);
        for (size_t i = 0; i < ndevs && i < compute_lo.size(); ++i) {
            const uint64_t d_bytes = (compute_hi[i] > compute_lo[i]) ? compute_hi[i] - compute_lo[i] : 0;
            r.compute_rate_main[i] = (hi_cells > lo_cells) ? d_bytes / (hi_cells - lo_cells) : 0;
        }

        // MTP draft context gets its OWN compute buffers. Measure using hi-lo difference.
        if (measure_mtp && llama_model_n_layer_nextn(model) > 0 && ctx_hi_main) {
            LLAMA_LOG_PLANNING("      measuring MTP draft context rate (pools %u -> %u):\n", lo_cells, hi_cells);
            
            // Create MTP at lo pool referencing ctx0 (main at lo)
            llama_context_params cp_mtp_lo = cp;  // same as ctx0's params
            cp_mtp_lo.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cp_mtp_lo.n_rs_seq = 0;
            cp_mtp_lo.ctx_other = ctx0;
            
            // Create MTP at hi pool referencing ctx_hi_main (main at hi)
            llama_context_params cp_mtp_hi = cp_hi;  // same as ctx_hi_main's params
            cp_mtp_hi.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cp_mtp_hi.n_rs_seq = 0;
            cp_mtp_hi.ctx_other = ctx_hi_main;
            
            std::vector<uint64_t> mtp_compute_lo, mtp_compute_hi;
            llama_context * ctx_mtp_lo = nullptr, * ctx_mtp_hi = nullptr;
            
            // Measure at lo pool
            ctx_mtp_lo = llama_init_from_model(model, cp_mtp_lo);
            if (ctx_mtp_lo) {
                mtp_compute_lo = kvfit_compute_by_device(ctx_mtp_lo, "mtp ctx, lo pool");
                
                // Measure at hi pool
                ctx_mtp_hi = llama_init_from_model(model, cp_mtp_hi);
                if (ctx_mtp_hi) {
                    mtp_compute_hi = kvfit_compute_by_device(ctx_mtp_hi, "mtp ctx, hi pool");
                } else {
                    LLAMA_LOG_WARN("kvfit: MTP context init at hi pool failed\n");
                }
            } else {
                LLAMA_LOG_WARN("kvfit: MTP context init at lo pool failed — cannot measure MTP rate\n");
            }
            
            // Calculate MTP rate from hi-lo difference (just like main rate)
            r.mtp_rate.assign(ndevs, 0);
            if (!mtp_compute_lo.empty() && !mtp_compute_hi.empty()) {
                for (size_t i = 0; i < ndevs; ++i) {
                    const uint64_t d_bytes = (mtp_compute_hi[i] > mtp_compute_lo[i]) ? 
                                             mtp_compute_hi[i] - mtp_compute_lo[i] : 0;
                    r.mtp_rate[i] = (hi_cells > lo_cells) ? d_bytes / (hi_cells - lo_cells) : 0;
                }
                
                // Log MTP rates
                {
                    for (size_t i = 0; i < ndevs && i < r.mtp_rate.size(); ++i) {
                        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                        if (!dev) continue;
                        const auto t = ggml_backend_dev_type(dev);
                        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
                        if (r.mtp_rate[i] > 0) {
                            LLAMA_LOG_PLANNING("        %s: MTP rate = %llu B/cell\n",
                                               ggml_backend_dev_name(dev),
                                               (unsigned long long)r.mtp_rate[i]);
                        }
                    }
                }
            }
            
            // Cleanup temporary MTP contexts
            if (ctx_mtp_lo) llama_free(ctx_mtp_lo);
            if (ctx_mtp_hi) llama_free(ctx_mtp_hi);
        }

        LLAMA_LOG_PLANNING("      compute-buffer rate (from llama_get_memory_breakdown, pools %u -> %u):\n",
                           lo_cells, hi_cells);
        {
            for (size_t i = 0; i < ndevs && i < r.compute_rate_main.size(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (!dev) continue;
                const auto t = ggml_backend_dev_type(dev);
                if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
                LLAMA_LOG_PLANNING("        %s: main rate = %llu B/cell\n",
                                   ggml_backend_dev_name(dev),
                                   (unsigned long long)r.compute_rate_main[i]);
            }
        }
        
        // Free hi context now that we're done measuring
        if (ctx_hi_main) {
            llama_free(ctx_hi_main);
            ctx_hi_main = nullptr;
        }
    }

    kvfit_warmup_decode(ctx0, prompt_tokens);

    // MTP draft context (speculative.cpp pattern): same n_ctx as the target, no
    // recurrent state of its own, reads h_nextn from the main context. It must
    // still exist when scan_measure runs so its nextn KV pool is part of the
    // measured delta — freed only afterwards.
    llama_context * ctx_mtp = nullptr;
    if (measure_mtp) {
        llama_context_params cp_mtp = cp;
        cp_mtp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        cp_mtp.n_rs_seq  = 0;
        cp_mtp.ctx_other = ctx0;
        ctx_mtp = llama_init_from_model(model, cp_mtp);
        if (ctx_mtp) {
            // one single-token MTP graph run at the first position after the
            // prompt: uploads every kernel an MTP deployment will launch.
            // The nextn graph requires a hidden-state embedding on every
            // ubatch entry (llama-graph.cpp llm_graph_input_embd_h::set_input):
            // ext-batch path with a zeroed state row at model n_embd width —
            // allocation behavior is what the measurement validates, not numerics.
            const int32_t n_embd = llama_model_n_embd(model);
            float * state = (float *)calloc((size_t)n_embd, sizeof(float));

            struct llama_batch_ext * mb = llama_batch_ext_init(ctx_mtp);
            const int32_t idx = llama_batch_ext_add_token(mb, 0, prompt_tokens.back());
            const llama_pos pos = (llama_pos)prompt_tokens.size();
            llama_batch_ext_set_pos(mb, idx, &pos);
            struct llama_embd embd_state = { state, 1, (size_t)n_embd };
            llama_batch_ext_set_embd_state(mb, idx, embd_state);
            llama_batch_ext_set_output_logits(mb, idx, true);

            const int rc = llama_process(ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, mb);
            llama_batch_ext_free(mb);
            free(state);
            if (rc != 0) {
                LLAMA_LOG_ERROR("kvfit: MTP draft decode failed (%d)\n", rc);
            } else {
                r.mtp_measured = true;
                // diagnostic breakdown only — the pool-proportional MTP rate comes
                // exclusively from the hi-lo difference method above. Absolute
                // division (bytes / n_ctx) wrongly folds in the fixed base, so it
                // is never used here as a fallback: a missing hi-lo measurement
                // leaves the rate 0 rather than reporting an inflated value.
                kvfit_compute_by_device(ctx_mtp, "mtp ctx");
            }
        } else {
            LLAMA_LOG_WARN("kvfit: MTP draft context init failed — nextn KV pool NOT in the measured delta\n");
        }
    }

    const double warmup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warmup0).count();
    LLAMA_LOG_PLANNING("      measurement: done; load %.1f s + warmup %.1f s (%.0f tok/s)%s\n",
                       load_s, warmup_s, warmup_s > 0 ? (double)n_tok / warmup_s : 0.0,
                       r.mtp_measured ? " [mtp draft context ready]" : "");

    kvfit_read_all_devices(r.scan_measure);
    r.usage = kvfit_compute_dev_usage(model);
    r.rates = kvfit_compute_rates(model, r.usage, cp.type_k, cp.type_v, cp.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED);
    r.n_ctx_measure = n_ctx_measure;   // the probe pool size for the decomposition (see kvfit_plan_compute)

    if (ctx_mtp) llama_free(ctx_mtp);
    llama_free(ctx0);
    llama_model_free(model);
    r.ok = true;
    return true;
}

// ============================================================================
// Plan: measurement → decomposition → budgets → upper limit → solve → layout
// ============================================================================
// One measurement run (kvfit_measure) prices everything the deployment will
// pay per device; the plan derives the solver inputs from it:
//   delta[i]   = free VRAM consumed by load + warmup on device i
//   kernels[i] = delta - weights(readback) - probe-KV - SSM(state rows x bpr)
//                (shaders + compute buffers: the fixed overhead every
//                 deployment pays — subtracted from each device's budget)
//   budget[i]  = free[i] - kernels[i] - reserve[i],
//   reserve[i] = total[i] x keep_free (+ per-device margin),
//                keep_free = 1 - vram_fraction
// The global ceiling for the pool, over all GPUs together:
//   kv upper limit = (sum budgets - sum weights(readback, incl. MTP)
//                    - state rows x sum SSM bytes/row) / sum KV bytes/cell
// and n_max = kv upper limit / 256 blocks. Greedy bisection then finds the
// largest block count whose layers fit every GPU's budget; the winning
// placement becomes the plan the caller applies to its real load.

struct kvfit_plan {
    bool ok = false;
    uint32_t n_cells = 0;            // pool size (blocks x 256), the solver's result
    uint32_t n_layer_all = 0;        // trunk + nextn layer count (hparams)
    std::vector<int> first_layer;    // per GPU: first layer index of its range
    std::vector<std::string> names;  // per GPU, solver row order
};

// One metadata-only model load for hparams (router/nextn flags, n_layer_all):
// no_alloc LLAMA_LOAD_MODE_NONE reads the GGUF header without touching VRAM.
static bool kvfit_peek_hparams(const char * path,
                               const llama_model_params * mparams,
                               llama_model * & model_out) {
    llama_model_params mp = *mparams;
    mp.no_alloc  = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    // MTP on by default (gpu-split convention): the peek is metadata-only (no VRAM),
    // and load_mtp=false would make the loader print "unused tensor" warnings for
    // every nextn tensor of an MTP model.
    mp.load_mtp  = true;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        LLAMA_LOG_WARN("kvfit: metadata pre-look failed — assuming no MTP block\n");
        return false;
    }
    model_out = model;
    return true;
}

// Model summary header (gpu-split style): identifies the model being planned
// and its capabilities. hparams come from a metadata-only peek (no VRAM).
static void kvfit_print_model_header(const char * path,
                                     const kvfit_gguf_info * gguf,
                                     const llama_model * model,
                                     const llama_context_params * cparams,
                                     const llama_kv_fit_params * params) {
    const auto & hp = model->hparams;
    const uint32_t n_layer_all  = hp.n_layer_all;
    const bool has_moe = hp.n_expert > 0;
    const bool has_mtp = hp.n_layer_nextn > 0;
    bool has_swa = false, has_ssm = false;
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        has_swa |= hp.is_swa(il);
        has_ssm |= hp.is_recr(il);
    }
    const double keep_free = 1.0 - params->vram_fraction;

    LLAMA_LOG_PLANNING("\n      file:       %s\n", path);
    LLAMA_LOG_PLANNING("      name:       %s\n", gguf->name.empty() ? "(unknown)" : gguf->name.c_str());
    LLAMA_LOG_PLANNING("      arch:       %s\n", gguf->arch.empty() ? llm_arch_name(model->arch) : gguf->arch.c_str());
    LLAMA_LOG_PLANNING("      layers:     %u (all)\n", (unsigned)n_layer_all);
    if (has_moe) {
        LLAMA_LOG_PLANNING("      MoE:        yes (%u experts, top-%u)\n",
                           (unsigned)hp.n_expert, (unsigned)hp.n_expert_used_max());
    } else {
        LLAMA_LOG_PLANNING("      MoE:        no\n");
    }
    LLAMA_LOG_PLANNING("      SWA:        %s\n", has_swa ? "yes" : "no");
    LLAMA_LOG_PLANNING("      SSM:        %s\n", has_ssm ? "yes" : "no");
    LLAMA_LOG_PLANNING("      MTP:        %s\n", has_mtp ? "yes" : "no");
    LLAMA_LOG_PLANNING("      kv_type_k:  %s\n", ggml_type_name(cparams->type_k));
    LLAMA_LOG_PLANNING("      kv_type_v:  %s\n", ggml_type_name(cparams->type_v));
    LLAMA_LOG_PLANNING("      keep_free:  %.2f\n", keep_free);
    if (gguf->embed_bytes > 0) {
        // token embedding stays on the CPU (no offloading benefit) — not part
        // of what gets distributed to GPUs, so it is shown outside the budgets.
        LLAMA_LOG_PLANNING("      embed(CPU): %lld\n", (long long)gguf->embed_bytes);
    }
}

static kvfit_plan kvfit_plan_compute(const char * path,
                                     const llama_model_params * mparams,
                                     const llama_context_params * cparams,
                                     const llama_kv_fit_params * params) {
    kvfit_plan plan;
    if (!params) {
        LLAMA_LOG_ERROR("kvfit: params is null — use llama_kv_fit_params_new()\n");
        return plan;
    }

    // ---- Guard 1: GPU deployments only (no CPU-only mode) -------------------
    if (!kvfit_gpus_present()) {
        LLAMA_LOG_ERROR("kvfit: no GPU devices found — planning requires at least one "
                        "GPU (there is no CPU-only mode)\n");
        return plan;
    }

    // ---- GGUF metadata: Guard 2 + the solver's per-layer weights -------------
    kvfit_gguf_info gguf;
    if (!kvfit_read_gguf(path, gguf)) {
        return plan;   // kvfit_read_gguf already reported
    }

    // ---- MTP: a nextn block measured in the same run as the trunk ------------
    // On router models the trunk attends to those layers on EVERY pass (no
    // separate draft context exists) and llama_init_from_model rejects MTP
    // contexts for them — skip measuring there. The metadata-only peek also
    // feeds the model header, so it runs before any device scan or load.
    bool has_nextn = false, is_router = false;
    llama_model * model_peek = nullptr;
    if (kvfit_peek_hparams(path, mparams, model_peek)) {
        kvfit_print_model_header(path, &gguf, model_peek, cparams, params);
        has_nextn  = model_peek->hparams.n_layer_nextn > 0;
        is_router  = model_peek->hparams.router_layer >= 0;
        llama_model_free(model_peek);
    }
    const bool extra_wants_mtp = params->extra && params->extra->shares_model;
    const bool measure_mtp = (extra_wants_mtp || !params->extra) && has_nextn && !is_router;

    // device scan with nothing loaded: Guard 2 and the budget denominators
    std::vector<kvfit_dev_snapshot> scan_raw;
    kvfit_read_all_devices(scan_raw);
    // This is a GPU planning tool — CPU devices are never shown or planned.
    LLAMA_LOG_PLANNING("\n      device scan (raw, nothing loaded):\n");
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU) continue;   // GPU tool: never show CPU
        LLAMA_LOG_PLANNING("        %-4zu %-6s total=%14llu free=%14llu  %s\n",
                           i, kvfit_dev_type_str(t),
                           (unsigned long long)scan_raw[i].total_bytes,
                           (unsigned long long)scan_raw[i].free_bytes,
                           ggml_backend_dev_name(dev));
    }

    // Guard 2: weights alone exceeding total free GPU VRAM is a definite no —
    // report it instead of paying for a load that cannot work.
    uint64_t gpu_free_total = 0, gpu_count = 0;
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        gpu_free_total += scan_raw[i].free_bytes;
        ++gpu_count;
    }

    // Guard 1: no GPUs — this tool plans GPU deployments, there is nothing to do.
    if (gpu_count == 0) {
        LLAMA_LOG_ERROR("kvfit: no GPU devices found — kv-fit requires at least one GPU\n");
        return plan;
    }

    if ((uint64_t)gguf.model_bytes > gpu_free_total) {
        LLAMA_LOG_ERROR("kvfit: model too large — weights (%llu B) exceed total free GPU "
                        "VRAM (%llu B on %zu GPUs)\n",
                        (unsigned long long)gguf.model_bytes,
                        (unsigned long long)gpu_free_total, gpu_count);
        return plan;
    }

    // ---- the measurement run ----------------------------------------------------
    kvfit_measure_result meas;
    if (!kvfit_measure(path, mparams, cparams, measure_mtp, meas)) {
        return plan;   // kvfit_measure already reported
    }
    const bool mtp_measured = meas.mtp_measured;
    const uint32_t n_ctx_measure = meas.n_ctx_measure;

    // deployment shape for the SSM state: rows per recurrent layer the
    // production contexts will allocate (n_seq_max x (live row + rollback rows)).
    // The application alone knows its checkpoint policy, so it passes the final
    // count via params->n_checkpoints; the planner uses it verbatim. 0 = legacy
    // fallback: derive from the context params (n_seq_max * (1 + n_rs_seq)).
    const uint32_t n_seq_max  = (cparams->n_seq_max > 0) ? (uint32_t)cparams->n_seq_max : 1;
    const uint32_t n_rs_seq   = (cparams->n_rs_seq  > 0) ? (uint32_t)cparams->n_rs_seq  : 0;
    const uint32_t state_rows = (params->n_checkpoints > 0)
                              ? params->n_checkpoints
                              : n_seq_max * (1u + n_rs_seq);

    // ---- per-device free VRAM after measurement ----------------------------------
    LLAMA_LOG_PLANNING("\n      device free VRAM after measurement (post-Hamlet decode):\n");
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU) continue;   // GPU tool: never show CPU
        LLAMA_LOG_PLANNING("        %-4zu %-6s free=%14llu  %s\n",
                           i, kvfit_dev_type_str(t),
                           (unsigned long long)meas.scan_measure[i].free_bytes,
                           ggml_backend_dev_name(dev));
    }

    // ---- per-device decomposition of the measured delta -------------------------
    LLAMA_LOG_PLANNING("\n      overhead decomposition: "
                       "kernels = delta - weights - probe-KV - SSM(state rows x bytes/row)\n");
    const uint64_t probe_cells = ((uint64_t)n_ctx_measure + 255u) / 256u * 256u; // GGML_PAD, as the cache does
    size_t nd_gpus = 0;
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok || i >= meas.scan_measure.size() || !meas.scan_measure[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) ++nd_gpus;
    }
    std::vector<uint64_t> d_delta(nd_gpus), d_weights(nd_gpus), d_probe_kv(nd_gpus), d_kernels(nd_gpus);
    std::vector<uint64_t> d_ssm_bpr(nd_gpus, 0), d_kv_bpc(nd_gpus, 0), d_mtp_bpc(nd_gpus, 0);
    std::vector<uint64_t> d_base(nd_gpus, 0);   // base cost per GPU (width-0 overhead: shaders + compute-buffer intercept)

    // Compute-buffer rates per pool cell: main context + MTP draft context as
    // separate vectors. bufs_bpc combines them — that is what the solver
    // charges; the split exists for the cost-breakdown display only.
    std::vector<uint64_t> buf_bpc(nd_gpus, 0), mtp_buf_bpc(nd_gpus, 0), bufs_bpc(nd_gpus, 0);
    {
        size_t gi = 0;
        for (size_t i = 0; i < scan_raw.size() && gi < nd_gpus; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !scan_raw[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            buf_bpc[gi]     = (i < meas.compute_rate_main.size()) ? meas.compute_rate_main[i] : 0;
            mtp_buf_bpc[gi] = (i < meas.mtp_rate.size())          ? meas.mtp_rate[i] : 0;
            bufs_bpc[gi]    = buf_bpc[gi] + mtp_buf_bpc[gi];
            ++gi;
        }
    }
    {
        size_t gi = 0;
        for (size_t i = 0; i < scan_raw.size() && gi < nd_gpus; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !scan_raw[i].ok || i >= meas.scan_measure.size() || !meas.scan_measure[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;

            const uint64_t delta   = scan_raw[i].free_bytes > meas.scan_measure[i].free_bytes
                                   ? scan_raw[i].free_bytes - meas.scan_measure[i].free_bytes : 0;
            const uint64_t weights = i < meas.usage.size() ? meas.usage[i].bytes : 0;
            const uint64_t kv_bpc  = i < meas.rates.size() ? meas.rates[i].kv_bytes_per_cell : 0;
            const uint64_t mtp_bpc = (mtp_measured && i < meas.rates.size()) ? meas.rates[i].mtp_kv_bytes_per_cell : 0;
            const uint64_t ssm_bpr = i < meas.rates.size() ? meas.rates[i].ssm_bytes_per_row : 0;

            // probe-KV: the measurement context's own pools (main + MTP draft when
            // measured). Unified pools are shared across streams — streams do not
            // multiply the size. SSM state rows: charged here AND in the upper
            // limit, at the deployment's planned row count.
            d_probe_kv[gi] = probe_cells * (kv_bpc + mtp_bpc);
            const uint64_t ssm_alloc = (uint64_t)state_rows * ssm_bpr;
            d_kernels[gi]  = (delta > weights + d_probe_kv[gi] + ssm_alloc)
                           ? delta - weights - d_probe_kv[gi] - ssm_alloc : 0;
            d_delta[gi]   = delta;
            d_weights[gi] = weights;
            d_ssm_bpr[gi] = ssm_bpr;
            d_kv_bpc[gi]  = kv_bpc;
            d_mtp_bpc[gi] = mtp_bpc;

            // Base-buffer derivation for the cost breakdown: at probe width the
            // compute buffers are base + rate*probe_cells. Subtracting the rate slice
            // (main + MTP draft rates) isolates the width-0 intercept (shaders + base
            // combined — llama.cpp does not report them separately).
            const uint64_t probe_buf_slice = (uint64_t)bufs_bpc[gi] * probe_cells;
            d_base[gi] = (d_kernels[gi] > probe_buf_slice) ? d_kernels[gi] - probe_buf_slice : 0;

            LLAMA_LOG_PLANNING("        %s: delta=%llu weights=%llu probe-KV=%llu "
                               "(cells(%llu) x bpc(%llu)) SSM=%llu kernels=%llu\n",
                               ggml_backend_dev_name(dev),
                               (unsigned long long)d_delta[gi], (unsigned long long)d_weights[gi],
                               (unsigned long long)d_probe_kv[gi],
                               (unsigned long long)probe_cells, (unsigned long long)(kv_bpc + mtp_bpc),
                               (unsigned long long)ssm_alloc,
                               (unsigned long long)d_kernels[gi]);
            ++gi;
        }
    }

    // ---- budgets: what each GPU leaves for weights + KV -------------------------
    // Each device's measured 'kernels' is a probe-width residual that already
    // includes its pool-proportional compute buffers (rate x probe_cells). The
    // solver will re-charge those at the target pool width, so we strip that
    // slice out and only charge the fixed remainder — the base-buffer column:
    //     overhead  = kernels - (bufs_bpc x probe_cells)   (= d_base, base-buffer)
    //     budget    = free - overhead - reserve
    const double keep_free = 1.0 - params->vram_fraction;

    LLAMA_LOG_PLANNING("\n      split budget: reserve = total x keep_free(%.2f)%s\n"
                       "                    budget  = free - base-buffer - reserve\n"
                       "   (base-buffer = the fixed cost subtracted now; the solver charges the\n"
                       "   pool-proportional compute buffers at target width, not here)\n",
                       keep_free, (params->margins ? " + margin" : ""));
    std::vector<uint64_t> budgets(nd_gpus);
    std::vector<uint64_t> reserves(nd_gpus);   // solver header: per-GPU reserve kept back
    std::vector<std::string> names;
    int64_t total_budget = 0, total_weights = 0;
    uint64_t ssm_per_row_total = 0, cell_size_total = 0;
    {
        size_t gi = 0;
        for (size_t i = 0; i < scan_raw.size() && gi < nd_gpus; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !scan_raw[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;

            int64_t total   = (int64_t)scan_raw[i].total_bytes;
            int64_t free    = (int64_t)scan_raw[i].free_bytes;
            // d_base[gi] was derived in the decomposition step as
            // kernels - bufs_bpc*probe_cells (the probe-width rate slice is
            // re-charged by the solver at target width). This IS the fixed
            // overhead we subtract from free — shaders + base-buffer. Using
            // it directly guarantees: the base-buffer column value == what's
            // subtracted here, no divergence between display and math.
            int64_t overhead = (int64_t)d_base[gi];
            int64_t reserve  = (int64_t)((double)total * keep_free);
            if (params->margins) reserve += (int64_t)params->margins[i]; // indexed by backend device order
            int64_t budget     = free - overhead - reserve;

            budgets[gi]  = budget > 0 ? (uint64_t)budget : 0;  // negative = overcommitted GPU
            reserves[gi] = (uint64_t)reserve;                  // what was kept back from this device's total
            names.push_back(ggml_backend_dev_name(dev));
            total_budget  += budget;
            total_weights += (int64_t)d_weights[gi];
            ssm_per_row_total += d_ssm_bpr[gi];
            cell_size_total   += d_kv_bpc[gi] + d_mtp_bpc[gi] + bufs_bpc[gi];

            LLAMA_LOG_PLANNING("        %s: budget = free(%lld) - base-buffer(%lld, "
                               "was kernels-probe(%lld), -probe-rate-slice(%lld)) - "
                               "reserve(total(%lld) x keep_free(%.2f)%s) = %lld\n",
                               names.back().c_str(),
                               (long long)free, (long long)overhead,
                               (long long)d_kernels[gi],
                               (long long)((uint64_t)bufs_bpc[gi] * probe_cells),
                               (long long)total, keep_free,
                               (params->margins ? " + margin" : ""),
                               (long long)budgets[gi]);
            ++gi;
        }
    }

    // ---- cost breakdown of the provisional split (before the bisection) -----------
    {
        std::vector<uint64_t> ssm_states(nd_gpus, 0);
        for (size_t g = 0; g < nd_gpus; ++g) {
            ssm_states[g] = d_ssm_bpr[g] * (uint64_t)state_rows;
        }
        kvfit_print_cost_breakdown("cost breakdown [provisional split]", names, d_weights, d_base,
                                   ssm_states, d_kv_bpc, d_mtp_bpc, buf_bpc, mtp_buf_bpc);
    }

    // ---- theoretical upper limit: the perfect-split ceiling ----------------------
    const uint64_t ssm_state_total = (uint64_t)state_rows * ssm_per_row_total;
    const int64_t upper_num = total_budget - total_weights - (int64_t)ssm_state_total;
    LLAMA_LOG_PLANNING("\n      kv cache upper limit: (total budget(%lld) - tensors(%lld)"
                       " - SSM state(%u x %llu)) / cell size(%llu)\n",
                       (long long)total_budget, (long long)total_weights,
                       (unsigned)state_rows, (unsigned long long)ssm_per_row_total,
                       (unsigned long long)cell_size_total);
    if (upper_num < 0 || cell_size_total == 0) {
        LLAMA_LOG_ERROR("kvfit: even a perfect split cannot host the KV pool — "
                        "numerator(%lld), total cell size(%llu B)\n",
                        (long long)upper_num, (unsigned long long)cell_size_total);
        return plan;
    }
    const uint64_t kv_upper = (uint64_t)upper_num / cell_size_total;
    const uint32_t n_max = (uint32_t)(kv_upper / 256u);
    LLAMA_LOG_PLANNING("        kv upper limit = %llu cells -> n_max = %7llu / 256 = %3u blocks\n",
                       (unsigned long long)kv_upper, (unsigned long long)kv_upper, (unsigned)n_max);
    if (n_max == 0) {
        LLAMA_LOG_ERROR("kvfit: even the perfect split fits fewer than 256 KV cells — "
                        "not a viable GPU deployment\n");
        return plan;
    }

    // ---- layer costs: GGUF weights + hparams-derived KV/SSM rates ---------------
    // (reuses the function-level peek pointer — freed after the header above)
    if (!kvfit_peek_hparams(path, mparams, model_peek)) {
        LLAMA_LOG_ERROR("kvfit: cannot read model hyperparameters for the solver\n");
        return plan;
    }
    auto layer_costs = kvfit_compute_layer_costs(model_peek, gguf.layer_weights,
                                                 (ggml_type)cparams->type_k, (ggml_type)cparams->type_v,
                                                 cparams->flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED, state_rows);
    plan.n_layer_all = model_peek->hparams.n_layer_all;
    llama_model_free(model_peek);

    // (compute-buffer rates computed earlier in the budgets section above;
    // bufs_bpc = main + MTP draft — what the solver charges per cell)
    const uint64_t last_gpu_extra = (uint64_t)gguf.norm_bytes + (uint64_t)gguf.output_bytes;
    const uint32_t n_blocks = kvfit_solve(n_max, layer_costs, budgets, reserves, names, last_gpu_extra,
                                          bufs_bpc);
    if (n_blocks == 0) {
        LLAMA_LOG_ERROR("kvfit: solver found no placement — even the perfect split cannot "
                        "hold all layers on the GPUs\n");
        return plan;
    }

    const uint32_t n_cells = n_blocks * 256u;
    if (n_cells < params->min_pool_cells) {
        LLAMA_LOG_WARN("kvfit: planned pool %u cells is below the viability floor "
                       "(min_pool_cells=%u) — a tiny KV cache for this deployment\n",
                       (unsigned)n_cells, (unsigned)params->min_pool_cells);
    }

    // ---- winning layout: re-run the greedy walk at n_blocks ---------------------
    // llama's loader assigns layer il to the first GPU whose CUMULATIVE split
    // fraction exceeds il/(N+1) (llama-model.cpp get_layer_buft_list, with
    // n_gpu_layers = -1). To make it reproduce the solver's ranges exactly,
    // each boundary a_g (first layer on GPU g) becomes cum[g-1] = a_g / (N+1);
    // llama normalizes splits, so tensor_split holds those cumulative values
    // directly. The output pseudo-layer N lands on the last GPU with any share.
    const uint32_t N = plan.n_layer_all;
    std::vector<int> first_layer(budgets.size(), 0);
    {
        // same walk as the solver: gated on each GPU's compute-buffer rate so
        // the final layout matches the winning candidate's checked placement
        const std::vector<size_t> walk = kvfit_greedy_walk(n_blocks, layer_costs, budgets, bufs_bpc);
        for (size_t g = 0; g < walk.size() && g < first_layer.size(); ++g) {
            first_layer[g] = (int)walk[g];
        }
    }

    plan.ok          = true;
    plan.n_cells     = n_cells;
    plan.first_layer = std::move(first_layer);
    plan.names       = names;

    LLAMA_LOG_PLANNING("\n      ==> plan: pool %u cells, placement:\n", (unsigned)n_cells);
    for (size_t g = 0; g < plan.first_layer.size(); ++g) {
        const int a = plan.first_layer[g];
        const int b = (g + 1 < plan.first_layer.size()) ? plan.first_layer[g + 1] : (int)N;
        LLAMA_LOG_PLANNING("        %s: layers %d..%d%s\n", plan.names[g].c_str(), a, b - 1,
                           (a >= b) ? " (empty)" : "");
    }

    // ---- cost breakdown of the final plan ------------------------------------
    // Recomputed from the solver's per-layer costs over the winning ranges;
    // the placement-independent terms (base-buffer, compute-buffer rates) are
    // shared with the provisional table.
    {
        const size_t nd = plan.first_layer.size();
        std::vector<uint64_t> p_wts(nd, 0), p_ssm(nd, 0), p_kv(nd, 0), p_mtp(nd, 0);
        for (size_t g = 0; g < nd; ++g) {
            const int a = plan.first_layer[g];
            const int b = (g + 1 < nd) ? plan.first_layer[g + 1] : (int)N;
            for (int l = a; l < b; ++l) {
                p_wts[g] += layer_costs[l].weights;
                p_ssm[g] += layer_costs[l].ssm_state;
                p_kv[g]  += layer_costs[l].kv_bpc;
                p_mtp[g] += layer_costs[l].mtp_kv_bpc;
            }
        }
        // the last GPU also takes the global tensors (norm + output / lm_head)
        if (nd > 0) p_wts[nd - 1] += last_gpu_extra;
        kvfit_print_cost_breakdown("cost breakdown [final plan]", plan.names, p_wts, d_base,
                                   p_ssm, p_kv, p_mtp, buf_bpc, mtp_buf_bpc);
    }

    // ---- Verification run: prove the plan works at runtime ---------------------
    kvfit_verify_plan(path, mparams, cparams, measure_mtp, 
                      plan.first_layer, n_cells);

    return plan;
}


// ============================================================================
// Public API
// ============================================================================
// Measurement + plan are expensive (a real load plus warmup decode); both
// entry points need the same result, so the last computation is cached and
// reused when the inputs match: same model path plus the same params object
// addresses (fllama passes identical structs to both calls). Any change ->
// a fresh measurement; stale results are never served.

namespace {

struct kvfit_cache_entry {
    bool valid = false;
    char path[1024] = "";
    const void * mp  = nullptr;   // mparams address
    const void * cp  = nullptr;   // cparams address
    const void * pp  = nullptr;   // planning params address
    kvfit_plan plan;
};

kvfit_cache_entry g_kvfit_cache;
std::mutex g_kvfit_cache_mu;

} // anonymous namespace

static bool kvfit_cache_match(const char * path, const void * mp, const void * cp, const void * pp) {
    return g_kvfit_cache.valid
        && strcmp(g_kvfit_cache.path, path) == 0
        && g_kvfit_cache.mp == mp
        && g_kvfit_cache.cp == cp
        && g_kvfit_cache.pp == pp;
}

static kvfit_plan kvfit_get_plan(const char * path,
                                 const llama_model_params * mparams,
                                 const llama_context_params * cparams,
                                 const llama_kv_fit_params * params) {
    std::lock_guard<std::mutex> lock(g_kvfit_cache_mu);
    if (!kvfit_cache_match(path, mparams, cparams, params)) {
        LLAMA_LOG_PLANNING("kvfit: new plan computation for %s\n", path);
        g_kvfit_cache = kvfit_cache_entry{};
        strncpy(g_kvfit_cache.path, path, sizeof(g_kvfit_cache.path) - 1);
        g_kvfit_cache.mp = mparams;
        g_kvfit_cache.cp = cparams;
        g_kvfit_cache.pp = params;
        g_kvfit_cache.plan = kvfit_plan_compute(path, mparams, cparams, params);
        g_kvfit_cache.valid = true;
    } else {
        LLAMA_LOG_PLANNING("kvfit: reusing computed plan for %s\n", path);
    }
    return g_kvfit_cache.plan;   // copied out under the lock
}

LLAMA_API bool llama_kv_fit_auto_split(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        int32_t * layer_bounds,
        const struct llama_kv_fit_params * params,
        int32_t * n_devices_planned) {
    kvfit_plan plan = kvfit_get_plan(path_model, mparams, cparams, params);
    if (!plan.ok || layer_bounds == nullptr) {
        LLAMA_LOG_ERROR("kvfit: auto split failed or no layer_bounds buffer — "
                        "the caller must not load this model as-is\n");
        return false;
    }

    // the plan's ranges ARE the loader's boundaries — write them verbatim,
    // and report how many of them there are so the caller knows exactly
    // which part of its buffer the plan occupies
    const size_t nd = plan.first_layer.size();
    for (size_t g = 0; g < nd; ++g) {
        layer_bounds[g] = plan.first_layer[g];
    }
    if (n_devices_planned) {
        *n_devices_planned = (int32_t)nd;
    }

    mparams->n_gpu_layers       = -1;   // all layers on GPUs (loader's standard idiom)
    mparams->tensor_layer_bounds = layer_bounds;
    mparams->tensor_split       = nullptr;   // the boundaries supersede ratio logic

    LLAMA_LOG_PLANNING("kvfit: applied placement to mparams (pool %u cells) — "
                       "reuse these params unchanged for the real load\n",
                       (unsigned)plan.n_cells);
    return true;
}

LLAMA_API int32_t llama_kv_fit_max_pool(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const struct llama_kv_fit_params * params) {
    kvfit_plan plan = kvfit_get_plan(path_model, mparams, cparams, params);
    if (!plan.ok) {
        return 0;   // failure already reported by the plan computation
    }
    return (int32_t)plan.n_cells;   // 256-aligned; below min_pool_cells is warned above
}

// ============================================================================
// Explain: solver terms for an explicit placement and pool size
// ============================================================================
// Pure metadata math for deviation analysis (no measurement load): given the
// placement a caller actually loads — layer_bounds[g] = first layer on GPU g,
// in backend order — and the unified pool size, print what the solver charges
// each device: tensors / swa/ssm / cache / mtp-cache / buffers. Same per-layer
// cost model as the fit calculation (GGUF weights + hparams rates); mask terms
// from the context parameters. The measured stage deltas are compared against
// these numbers one-for-one: tensors <-> load, swa/ssm + cache <-> main ctx,
// mtp-cache <-> MTP ctx, buffers (+ runtime kernels) <-> warmup decode.

static void kvfit_exprint_terms(const char * path,
                                const llama_context_params * cparams,
                                const llama_kv_fit_params * params,
                                const int32_t * bounds, size_t n_dev,
                                uint32_t pool_cells) {
    if (!params || !bounds || n_dev == 0 || pool_cells == 0) {
        LLAMA_LOG_ERROR("kvfit: explain requires params, layer bounds and a pool size\n");
        return;
    }

    kvfit_gguf_info gguf;
    if (!kvfit_read_gguf(path, gguf)) {
        return;   // kvfit_read_gguf already reported
    }

    llama_model_params mp_peek = llama_model_default_params();
    llama_model * model = nullptr;
    if (!kvfit_peek_hparams(path, &mp_peek, model)) {
        LLAMA_LOG_ERROR("kvfit: explain cannot read the model hyperparameters\n");
        return;
    }

    // SSM state rows: same derivation as the planner (the application's policy,
    // passed verbatim; legacy fallback from the context params)
    const uint32_t n_seq_max  = (cparams->n_seq_max > 0) ? (uint32_t)cparams->n_seq_max : 1;
    const uint32_t n_rs_seq   = (cparams->n_rs_seq  > 0) ? (uint32_t)cparams->n_rs_seq  : 0;
    const uint32_t state_rows = (params->n_checkpoints > 0)
                              ? params->n_checkpoints
                              : n_seq_max * (1u + n_rs_seq);

    // same per-layer costs the solver charges with
    auto layers = kvfit_compute_layer_costs(model, gguf.layer_weights,
                                            (ggml_type)cparams->type_k, (ggml_type)cparams->type_v,
                                            cparams->flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED,
                                            state_rows);
    const uint32_t n_ctx_train = model->hparams.n_ctx_train;
    llama_model_free(model);

    // pool-sized kq_mask graph inputs: bytes per cell per context type, at the
    // deployment's reserve width (same rule as the plan computation)
    const uint32_t n_ctx_eff = (cparams->n_ctx > 0) ? (uint32_t)cparams->n_ctx : n_ctx_train;
    const uint64_t mask_bpc_main = kvfit_mask_bpc_per_ctx(cparams, n_ctx_eff);
    const uint64_t mask_bpc_mtp  = kvfit_mask_bpc_per_ctx(cparams, n_ctx_eff);

    LLAMA_LOG_PLANNING("\n      explain: solver terms for bounds=");
    for (size_t g = 0; g < n_dev; ++g) {
        LLAMA_LOG_PLANNING("%s%d", g ? "," : "", (int)bounds[g]);
    }
    LLAMA_LOG_PLANNING(" pool=%u cells:\n", (unsigned)pool_cells);

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;

        // this GPU's index in the bounds array: count of GPU devices before it
        size_t seen = 0;
        for (size_t j = 0; j < i; ++j) {
            ggml_backend_dev_t d2 = ggml_backend_dev_get(j);
            if (!d2) continue;
            const auto t2 = ggml_backend_dev_type(d2);
            if (t2 == GGML_BACKEND_DEVICE_TYPE_GPU || t2 == GGML_BACKEND_DEVICE_TYPE_IGPU) ++seen;
        }
        if (seen >= n_dev) continue;

        const int64_t first_layer = bounds[seen];
        const int64_t last_layer  = (seen + 1 < n_dev) ? bounds[seen + 1] : (int64_t)layers.size();
        const bool has_layers = (last_layer > first_layer);

        // range sums over the layers this GPU hosts (solver-style accounting)
        uint64_t wts = 0, ssm = 0, kv_bpc = 0, mtp_bpc = 0;
        if (has_layers) {
            for (size_t il = (size_t)first_layer; il < (size_t)last_layer && il < layers.size(); ++il) {
                wts      += layers[il].weights;
                ssm      += layers[il].ssm_state;
                kv_bpc   += layers[il].kv_bpc;
                mtp_bpc  += layers[il].mtp_kv_bpc;
            }
        }
        const bool is_last = (seen + 1 == n_dev);
        if (is_last) {   // global tensors (norm/output) ride the last GPU
            wts += (uint64_t)gguf.norm_bytes + (uint64_t)gguf.output_bytes;
        }

        // buffers: one shared kq_mask per context type the device computes for:
        // main where trunk attention layers landed, MTP where nextn layers did
        bool main_ctx = false, mtp_here = false;
        if (has_layers) {
            for (size_t il = (size_t)first_layer; il < (size_t)last_layer && il < layers.size(); ++il) {
                main_ctx |= (layers[il].kv_bpc > 0);
                mtp_here |= (layers[il].mtp_kv_bpc > 0);
            }
        }
        const uint64_t buffers_bpc = (main_ctx ? mask_bpc_main : 0) + (mtp_here ? mask_bpc_mtp : 0);

        const uint64_t cells   = pool_cells;
        const uint64_t cache   = cells * kv_bpc;
        const uint64_t mtp_kv  = cells * mtp_bpc;
        const uint64_t buffers = cells * buffers_bpc;

        char range[32] = "(none)";
        if (has_layers) {
            snprintf(range, sizeof range, "%lld..%lld", (long long)first_layer, (long long)(last_layer - 1));
        }
        LLAMA_LOG_PLANNING("        %s: layers = %s\n"
                           "          tensors   = %llu\n"
                           "          swa/ssm   = %llu (rows(%u) x bytes/row)\n"
                           "          cache     = cells(%llu) x bpc(%llu) = %llu\n"
                           "          mtp-cache = cells(%llu) x bpc(%llu) = %llu\n"
                           "          buffers   = cells(%llu) x bpc(%llu) = %llu\n",
                           ggml_backend_dev_name(dev), range,
                           (unsigned long long)wts, (unsigned long long)ssm, (unsigned)state_rows,
                           (unsigned long long)cells, (unsigned long long)kv_bpc, (unsigned long long)cache,
                           (unsigned long long)cells, (unsigned long long)mtp_bpc, (unsigned long long)mtp_kv,
                           (unsigned long long)cells, (unsigned long long)buffers_bpc, (unsigned long long)buffers);
    }
}

LLAMA_API void llama_kv_fit_explain(
        const char * path_model,
        const struct llama_context_params * cparams,
        const struct llama_kv_fit_params * params,
        const int32_t * layer_bounds,
        int32_t n_devices,
        uint32_t pool_cells) {
    kvfit_exprint_terms(path_model, cparams, params, layer_bounds,
                        (size_t)n_devices, pool_cells);
}
