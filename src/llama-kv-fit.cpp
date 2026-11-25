// [FORK] Unified KV pool planning — the gpu-split algorithm, library-side.
//
// The application supplies its deployment parameters (model params, context
// params, an optional extra/MTP context); this library SOLVES the optimal
// tensor split and the maximum unified KV cache pool size against live
// device VRAM. All C++ lives here; callers speak the public C API in
// llama-ext.h only.
//
// Measurement-based: one REAL model load with llama_model's native free-VRAM
// auto-split as the provisional placement, a measurement context at a common
// deployment size, and a Hamlet warmup decode so every kernel the deployment will launch is
// uploaded to the devices — kernels only materialize on their first execution
// (no preload/prewarm API exists on any backend). Readback of the actual
// placement plus architecture-derived KV/SSM rates and a greedy bisection
// solver derive the plan from that single measurement: what is measured is
// exactly what the deployment will pay.
//
// Logging: detailed traces use LLAMA_LOG_PLANNING (visible without all of
// llama.cpp's debug logs); failures and conclusions use the standard levels.
// The consumer decides visibility. Deployment report blocks (model header,
// layer table, cost breakdowns) leave as ONE indivisible unit each — see
// kvfit_block_emit: into the caller's collection stream when one is set, or
// as a single log record otherwise, so live-log consumers never get records
// interleaving inside a block.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // open_memstream (block emission) — must precede includes
#endif

#include "llama-impl.h"   // LLAMA_LOG_* macros
#include "llama-ext.h"    // public planning API declarations
#include "llama-model.h"  // llama_internal_get_tensor_map, llama_hparams
#include "model-info.h"   // deployment report: model header + per-layer table

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

#include "warmup-text.h"   // Hamlet excerpt (warmup prompt for the verification load)

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
    p->options        = 0;      // v5: all behavior options off by default
    p->report_stream  = nullptr;   // v6: deployment tables go to stderr unless the caller collects them
    return p;
}

LLAMA_API void llama_kv_fit_params_free(struct llama_kv_fit_params * p) {
    delete p;
}

// The deployment-report stream: caller-collected (params->report_stream — the
// kv-fit tool assembles the whole report and writes it out atomically) or a
// single log record by default.
static FILE * kvfit_report(const struct llama_kv_fit_params * p) {
    return (p && p->report_stream) ? p->report_stream : stderr;
}

// ---- unified block emission -------------------------------------------------
// Every report block (model header + layer table, cost breakdowns) leaves as
// ONE indivisible unit: into the caller's collection stream when one is set
// (the kv-fit tool assembles the full report, written atomically at exit), or
// as a SINGLE log record otherwise — so consumers with live logs (the server,
// fllama) can never have records interleave inside a block, no matter how
// their transports buffer. Blocks start on a fresh line: the leading newline
// is guaranteed here.
static bool kvfit_block_emit(const struct llama_kv_fit_params * p, const char * text, size_t len) {
    if (len == 0) return true;
    FILE * rep = (p && p->report_stream) ? p->report_stream : nullptr;
    if (rep) {
        return fwrite(text, 1, len, rep) == len;
    }
    // log path: pad a leading newline when the block does not already start
    // with one, then emit as one atomic record.
    if (text[0] != '\n') {
        std::string s("\n", 1);
        s.append(text, len);
        llama_log_internal(GGML_LOG_LEVEL_PLANNING, "%s", s.c_str());
    } else {
        llama_log_internal(GGML_LOG_LEVEL_PLANNING, "%s", text);
    }
    return true;
}

// Render helper: call render() with a temporary stream, then emit everything
// written into it as one block.
static bool kvfit_emit_block(const struct llama_kv_fit_params * p,
                             void (*render)(FILE *, void *), void * ctx) {
    char * buf = nullptr;
    size_t len = 0;
    FILE * tmp = open_memstream(&buf, &len);
    if (!tmp) return false;
    render(tmp, ctx);
    fflush(tmp);                    // open_memstream updates pointers on flush
    if (fclose(tmp) != 0) { free(buf); return false; }
    bool ok = kvfit_block_emit(p, buf ? buf : "", len);
    free(buf);
    return ok;
}

namespace {
// Context for the cost-breakdown render callback (kvfit_emit_block).
struct kvfit_cb_ctx {
    const char * title;
    const std::vector<std::string> & names;
    const std::vector<uint64_t> & free, & kernels, & base, & reserve, & budget;
    const std::vector<uint64_t> & wts, & ssm_state;
    const std::vector<uint64_t> & kv_bpc, & mtp_bpc, & buf_bpc, & mtp_buf_bpc;
};

// render() target for the breakdown: title + box into one stream. The table is
// written by print_table as a single atomic block against the same stream.
void kvfit_cb_render(FILE * out, void * vctx) {
    const auto & c = *static_cast<kvfit_cb_ctx *>(vctx);
    fprintf(out, "\n      %s\n", c.title);

    // columns: split-budget ledger (independent of the placement), then the
    // placement-dependent fixed costs, then the per-cell rates (buffers kept
    // separate) and their sum
    static const char * hdrs[12] = { "free", "kernels", "base-buffer", "reserve", "budget",
                                     "tensors", "ssm-state", "kv/c", "mtp/c",
                                     "buf/c", "mbuf/c", "total/c" };
    const int n_rows = (int)c.names.size() + 1;   // one per GPU + TOTAL

    std::vector<std::string> keys;
    keys.reserve(n_rows);
    for (const auto & n : c.names) keys.push_back(n);
    keys.push_back("TOTAL");

    std::vector<long> flat((size_t)n_rows * 12, 0);
    const auto get = [](const std::vector<uint64_t> & v, size_t g) -> long {
        return (g < v.size()) ? (long)v[g] : 0;
    };
    for (size_t g = 0; g < c.names.size(); ++g) {
        const size_t o = g * 12;
        flat[o + 0] = get(c.free, g);
        flat[o + 1] = get(c.kernels, g);
        flat[o + 2] = get(c.base, g);
        flat[o + 3] = get(c.reserve, g);
        flat[o + 4] = get(c.budget, g);
        flat[o + 5] = get(c.wts, g);
        flat[o + 6] = get(c.ssm_state, g);
        flat[o + 7] = get(c.kv_bpc, g);
        flat[o + 8] = get(c.mtp_bpc, g);
        flat[o + 9] = get(c.buf_bpc, g);
        flat[o + 10] = get(c.mtp_buf_bpc, g);
        flat[o + 11] = flat[o + 7] + flat[o + 8] + flat[o + 9] + flat[o + 10];
    }
    // TOTAL row: column sums (total/c = sum of the four per-cell columns)
    for (int col = 0; col < 12; ++col) {
        long t = 0;
        for (size_t g = 0; g < c.names.size(); ++g) t += flat[g * 12 + col];
        flat[(size_t)(c.names.size()) * 12 + col] = t;
    }

    std::vector<const long *> row_ptrs(n_rows);
    for (int r = 0; r < n_rows; ++r) row_ptrs[r] = flat.data() + (size_t)r * 12;
    std::vector<const char *> keys_arr(n_rows);
    for (int r = 0; r < n_rows; ++r) keys_arr[r] = keys[r].c_str();

    print_table(out, n_rows, 12, hdrs, keys_arr.data(), "device", row_ptrs.data(), nullptr, true);
}
} // anonymous namespace

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

// Planning devices: GPU/iGPU whenever present; on a CPU-only machine the KV pool
// is planned against system RAM through the CPU backend instead.
static bool kvfit_is_plan_device(enum ggml_backend_dev_type t, bool cpu_only) {
    if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return true;
    }
    return cpu_only && t == GGML_BACKEND_DEVICE_TYPE_CPU;
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
// this is a GPU planning tool, CPU placement is noise for planning. Rendered
// from the stored readback so kvfit_plan_compute places it right in front of
// the cost breakdown table.
static void kvfit_print_provisional_split(const std::vector<kvfit_dev_usage> & usage) {
    if (!kvfit_gpus_present()) return;

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

// Print the per-GPU cost breakdown: the split-budget ledger first (free -
// kernels - base-buffer - reserve = budget), then the placement costs
// (tensors, SSM) and the per-cell rates with their sum. Used for both the
// provisional split (before the bisection) and the final plan — the caller
// supplies per-GPU arrays in its own placement. Title + box leave as ONE
// indivisible block through kvfit_block_emit (collection stream or log).
static bool kvfit_print_cost_breakdown(const struct llama_kv_fit_params * params,
                                       const char * title,
                                       const std::vector<std::string> & names,
                                       const std::vector<uint64_t> & free,
                                       const std::vector<uint64_t> & kernels,
                                       const std::vector<uint64_t> & base,
                                       const std::vector<uint64_t> & reserve,
                                       const std::vector<uint64_t> & budget,
                                       const std::vector<uint64_t> & wts,
                                       const std::vector<uint64_t> & ssm_state,
                                       const std::vector<uint64_t> & kv_bpc,
                                       const std::vector<uint64_t> & mtp_bpc,
                                       const std::vector<uint64_t> & buf_bpc,
                                       const std::vector<uint64_t> & mtp_buf_bpc) {
    kvfit_cb_ctx ctx = { title, names, free, kernels, base, reserve, budget,
                         wts, ssm_state, kv_bpc, mtp_bpc, buf_bpc, mtp_buf_bpc };
    return kvfit_emit_block(params, kvfit_cb_render, &ctx);
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

    // The whole step as ONE log record: the solver pass is pure computation
    // (microseconds), so splitting a step across records just scatters log
    // stamps inside what is a single unit of work. Accumulate, emit once.
    std::string rec;
    { char b[128];
      snprintf(b, sizeof b, "\n        testing with n = %-5u blocks (%llu cells)\n",
               n_blocks, (unsigned long long)cells);
      rec += b; }

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

        char b[1024];
        snprintf(b, sizeof b, "          %s: layers = %s\n"
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
        rec += b;
        if (fits) {
            snprintf(b, sizeof b, "            used      = tensors(%llu) + swa/ssm(%llu) + cache(%llu) + mtp(%llu) + buffers(%llu) = %llu\n"
                                  "            not used  = budget(%llu) - used(%llu) = %llu\n",
                     (unsigned long long)wts, (unsigned long long)ssm,
                     (unsigned long long)cache, (unsigned long long)mtp,
                     (unsigned long long)buffers, (unsigned long long)used,
                     (unsigned long long)budgets[g], (unsigned long long)used,
                     (unsigned long long)(budgets[g] - used));
        } else {
            snprintf(b, sizeof b, "            used      = tensors(%llu) + swa/ssm(%llu) + cache(%llu) + mtp(%llu) + buffers(%llu) = %llu\n"
                                  "            over limit = used(%llu) - budget(%llu) = %llu\n",
                     (unsigned long long)wts, (unsigned long long)ssm,
                     (unsigned long long)cache, (unsigned long long)mtp,
                     (unsigned long long)buffers, (unsigned long long)used,
                     (unsigned long long)used, (unsigned long long)budgets[g],
                     (unsigned long long)(used - budgets[g]));
        }
        rec += b;

        if (is_last) {
            LLAMA_LOG_PLANNING("%s", rec.c_str());   // the step as one record
            return fits;
        }
    }
    LLAMA_LOG_PLANNING("%s", rec.c_str());   // the step as one record
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

// Run the verification load: model with the COMPUTED split + context at the
// planned pool + warmup decode (forces every lazy kernel allocation). Also
// proves the loader honored the plan's boundaries (per-layer readback).
// Returns true only when every stage succeeded.
static bool kvfit_verify_plan(const char * path,
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
        return false;
    }

    // placement readback: prove the loader applied the plan's boundaries
    // (expected device = last whose bound does not exceed il; the output
    // pseudo-layer lands on the last device; -1 = CPU = a failure)
    const uint32_t n_layer_all = llama_model_n_layer_all(model);
    for (uint32_t il = 0; il <= n_layer_all; ++il) {
        const int dev = llama_model_layer_device_index(model, il);
        int expected = 0;
        for (size_t g = nd; g > 0; --g) {
            if ((int32_t)il >= first_layer[g - 1]) { expected = (int)(g - 1); break; }
        }
        if (dev != expected) {
            char what[32];
            if (il == n_layer_all) strcpy(what, "output");
            else snprintf(what, sizeof what, "layer %u", il);
            LLAMA_LOG_ERROR("kvfit_verify_plan: MISMATCH at %s: loaded on device %d, plan says %d\n",
                            what, dev, expected);
            llama_model_free(model);
            delete[] bounds;
            return false;   // the loader ignored the planned boundaries
        }
    }
    LLAMA_LOG_PLANNING("      placement verified: all layers on their planned devices\n");

    // Hamlet warmup text (reuse the same prompt as the measurement)
    std::vector<int32_t> prompt_tokens(strlen(WARMUP_TEXT) + 8);
    const int n_tok = llama_tokenize(
            llama_model_get_vocab(model), WARMUP_TEXT, strlen(WARMUP_TEXT),
            prompt_tokens.data(), (int)prompt_tokens.size(), true, false);
    if (n_tok <= 0) {
        LLAMA_LOG_ERROR("kvfit_verify_plan: warmup tokenization failed\n");
        llama_model_free(model);
        delete[] bounds;
        return false;
    }
    prompt_tokens.resize(n_tok);

    // Single verification run at the planned pool size — what will actually
    // deploy; the warmup decode forces every lazy kernel allocation
    LLAMA_LOG_PLANNING("\n      verification run: context size = %u\n", (unsigned)n_cells_planned);

    llama_context_params cp = *cparams_base;
    cp.kv_unified = true;
    cp.n_ctx      = n_cells_planned;
    cp.n_kv_cache = 0;   // unified mode: pool follows n_ctx

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        LLAMA_LOG_ERROR("kvfit_verify_plan: context init failed — plan over-committed VRAM\n");
        llama_model_free(model);
        delete[] bounds;
        return false;
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
    bool verified = true;
    const int decoded = kvfit_warmup_decode(ctx, prompt_tokens);
    if (decoded == 0) {
        LLAMA_LOG_ERROR("kvfit_verify_plan: warmup decode failed\n");
        verified = false;
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
            verified = false;
        }
    }

    // Show per-GPU free VRAM after everything is loaded + warmed up
    std::vector<kvfit_dev_snapshot> snaps;
    kvfit_read_all_devices(snaps);
    LLAMA_LOG_PLANNING("        free VRAM after planned context (post-warmup):\n");
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

    // Cleanup this run's contexts
    if (ctx_mtp) llama_free(ctx_mtp);
    llama_free(ctx);

    // Cleanup
    llama_model_free(model);
    delete[] bounds;
    return verified;
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
// ground truth our old kq_mask formula had to guess at. GPU-side host bufts
// (..._Host pinned staging memory) are printed as their own rows but live in
// system RAM — they are NEVER folded into a GPU rate (that contamination
// doubled the ROCm0 rate on rocky2). The CPU backend's own buffers (also host
// bufts) ARE charged to the CPU device: real RAM used by the deployment.
static std::vector<uint64_t> kvfit_compute_by_device(const llama_context * ctx, const char * label, bool debug, FILE * report) {
    const size_t ndevs = ggml_backend_dev_count();
    std::vector<uint64_t> compute(ndevs, 0);
    if (debug) fprintf(report, "  breakdown [%s] (from llama_get_memory_breakdown):\n", label);
    for (auto & [buft, data] : llama_get_memory_breakdown(ctx)) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) continue;
        for (size_t i = 0; i < ndevs; ++i) {
            if (ggml_backend_dev_get(i) == dev) {
                if (debug) {
                    fprintf(report, "    buft_name='%s'  compute=%llu (%.2f GiB)  context=%llu (%.2f GiB)\n",
                            ggml_backend_buft_name(buft), 
                            (unsigned long long)data.compute, 
                            (double)data.compute / (1024.0 * 1024.0 * 1024.0),
                            (unsigned long long)data.context,
                            (double)data.context / (1024.0 * 1024.0 * 1024.0));
                }
                const bool is_cpu_dev = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
                if (is_cpu_dev || !ggml_backend_buft_is_host(buft)) {
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

    // Per-device .compute readbacks at lo and hi pool width (main and MTP ctx).
    // Plan decomposition derives from these: base-buffer = the width-0 intercept
    // (lo - (hi - lo)) of each measured context, and the kernels residual
    // subtracts what was live at probe width (the lo values).
    std::vector<uint64_t> compute_lo_main;          // main ctx .compute at lo pool
    std::vector<uint64_t> compute_hi_main;          // main ctx .compute at hi pool
    std::vector<uint64_t> compute_lo_mtp;           // MTP ctx .compute at lo pool
    std::vector<uint64_t> compute_hi_mtp;           // MTP ctx .compute at hi pool
};

static bool kvfit_measure(const char * path,
                          const llama_model_params * mparams,
                          const llama_context_params * cparams,
                          bool measure_mtp,
                          bool debug,
                          FILE * report,
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
        std::vector<uint64_t> compute_lo = kvfit_compute_by_device(ctx0, "main ctx, lo pool", debug, report);   // main ctx at lo
        
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
            compute_hi = kvfit_compute_by_device(ctx_hi_main, "main ctx, hi pool", debug, report);   // main ctx at hi
        }

        // Calculate main rate from hi-lo difference
        r.compute_rate_main.assign(ndevs, 0);
        for (size_t i = 0; i < ndevs && i < compute_lo.size() && i < compute_hi.size(); ++i) {
            const uint64_t d_bytes = (compute_hi[i] > compute_lo[i]) ? compute_hi[i] - compute_lo[i] : 0;
            r.compute_rate_main[i] = (hi_cells > lo_cells) ? d_bytes / (hi_cells - lo_cells) : 0;
        }
        // readbacks for the plan decomposition: base-buffer intercepts and the
        // live-at-probe compute total both derive from these lo/hi values
        r.compute_lo_main = compute_lo;
        r.compute_hi_main = compute_hi;

        // MTP draft context gets its OWN compute buffers. Measure using hi-lo difference.
        if (measure_mtp && llama_model_n_layer_nextn(model) > 0 && ctx_hi_main) {
            if (debug) LLAMA_LOG_PLANNING("      measuring MTP draft context rate (pools %u -> %u):\n", lo_cells, hi_cells);
            
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
                mtp_compute_lo = kvfit_compute_by_device(ctx_mtp_lo, "mtp ctx, lo pool", debug, report);
                
                // Measure at hi pool
                ctx_mtp_hi = llama_init_from_model(model, cp_mtp_hi);
                if (ctx_mtp_hi) {
                    mtp_compute_hi = kvfit_compute_by_device(ctx_mtp_hi, "mtp ctx, hi pool", debug, report);
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
                // readbacks for the plan decomposition (main ctx stores its own above)
                r.compute_lo_mtp = mtp_compute_lo;
                r.compute_hi_mtp = mtp_compute_hi;
                
                // Log MTP rates (diagnostic only — the rate values flow into the solver regardless)
                if (debug) {
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

        if (debug) LLAMA_LOG_PLANNING("      compute-buffer rate (from llama_get_memory_breakdown, pools %u -> %u):\n",
                           lo_cells, hi_cells);
        if (debug) {
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
                kvfit_compute_by_device(ctx_mtp, "mtp ctx", debug, report);
            }
        } else {
            LLAMA_LOG_WARN("kvfit: MTP draft context init failed — nextn KV pool NOT in the measured delta\n");
        }
    }

    const double warmup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_warmup0).count();
    if (debug) LLAMA_LOG_PLANNING("      measurement: done; load %.1f s + warmup %.1f s (%.0f tok/s)%s\n",
                       load_s, warmup_s, warmup_s > 0 ? (double)n_tok / warmup_s : 0.0,
                       r.mtp_measured ? " [mtp draft context ready]" : "");

    kvfit_read_all_devices(r.scan_measure);
    r.usage = kvfit_compute_dev_usage(model);
    // [FORK] CPU-only deployment: mmap'd weights live in the CPU_Mapped buft,
    // which carries no device pointer — the readback attributes no layers to
    // the CPU device. Every layer IS on it by construction, so synthesize that
    // placement: the architecture-derived KV/SSM/MTP rates need the layer list.
    if (!kvfit_gpus_present()) {
        for (size_t i = 0; i < r.usage.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            if (!r.usage[i].layers.empty()) continue;
            r.usage[i].first_layer = 0;
            r.usage[i].last_layer  = (int)model->hparams.n_layer_all - 1;
            for (uint32_t il = 0; il < model->hparams.n_layer_all; ++il) {
                r.usage[i].layers.push_back(il);
            }
        }
    }
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
//   delta[i]    = free VRAM consumed by load + warmup on device i
//   kernels[i]  = delta - weights(readback) - probe-KV - SSM(state rows x bpr)
//                 - compute(lo, measured): the shader binary residual
//   base-buffer[i] = width-0 intercept of the measured compute buffers:
//                 compute-lo - (compute-hi - compute-lo), per context
//   budget[i]  = free[i] - kernels[i] - base-buffer[i] - reserve[i],
//   reserve[i] = total[i] x keep_free (+ per-device margin),
//                keep_free = 1 - vram_fraction
// The global ceiling for the pool, over all GPUs together:
//   kv upper limit = (sum budgets - sum weights(readback, incl. MTP)
//                    - state rows x sum SSM bytes/row) / sum KV bytes/cell
// and n_max = kv upper limit / 256 blocks. Greedy bisection then finds the
// largest block count whose layers fit every GPU's budget; the winning
// placement becomes the plan the caller applies to its real load.

struct kvfit_plan {
    bool ok = false;                // a viable deployment exists
    uint32_t n_cells = 0;           // maximum viable pool (cells, 256-aligned)
    uint32_t n_layer_all = 0;       // trunk + nextn layer count (hparams)
    uint32_t n_ctx_train = 0;       // context size the model was trained on
    std::vector<int> first_layer;    // per GPU: first layer index of its range
    std::vector<std::string> names;  // per GPU, solver row order
    bool cpu_only = false;           // placement is trivial (all on CPU); no GPU split to apply
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

    // ---- GGUF metadata: the model-too-large guard + the solver's per-layer weights --
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

    // device scan with nothing loaded: guard and budget denominators.
    // Planning devices: GPU/iGPU whenever present; on a CPU-only machine the
    // pool is planned against system RAM through the CPU backend instead.
    std::vector<kvfit_dev_snapshot> scan_raw;
    kvfit_read_all_devices(scan_raw);
    const bool cpu_only = !kvfit_gpus_present();

    LLAMA_LOG_PLANNING("\n      device scan (raw, nothing loaded):\n");
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (!kvfit_is_plan_device(t, cpu_only)) continue;
        LLAMA_LOG_PLANNING("        %-4zu %-6s total=%14llu free=%14llu  %s\n",
                           i, kvfit_dev_type_str(t),
                           (unsigned long long)scan_raw[i].total_bytes,
                           (unsigned long long)scan_raw[i].free_bytes,
                           ggml_backend_dev_name(dev));
    }

    // Guard: weights alone exceeding total free planning-device memory is a
    // definite no — report it instead of paying for a load that cannot work.
    uint64_t dev_free_total = 0, dev_count = 0;
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (!kvfit_is_plan_device(t, cpu_only)) continue;
        dev_free_total += scan_raw[i].free_bytes;
        ++dev_count;
    }

    if (dev_count == 0) {
        LLAMA_LOG_ERROR("kvfit: no usable planning device found — cannot plan a deployment\n");
        return plan;
    }

    if ((uint64_t)gguf.model_bytes > dev_free_total) {
        LLAMA_LOG_ERROR("kvfit: model too large — weights (%llu B) exceed total free "
                        "%s memory (%llu B on %zu devices)\n",
                        (unsigned long long)gguf.model_bytes,
                        cpu_only ? "RAM" : "VRAM",
                        (unsigned long long)dev_free_total, dev_count);
        return plan;
    }

    // ---- the measurement run ----------------------------------------------------
    kvfit_measure_result meas;
    if (!kvfit_measure(path, mparams, cparams, measure_mtp, params->options & LLAMA_KV_FIT_DEBUG,
                                 kvfit_report(params), meas)) {
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

    // ---- per-device free memory after measurement (diagnostic, DEBUG only) -----
    if (params->options & LLAMA_KV_FIT_DEBUG) {
        LLAMA_LOG_PLANNING("\n      device free %s after measurement (post-Hamlet decode):\n",
                           cpu_only ? "RAM" : "VRAM");
        for (size_t i = 0; i < scan_raw.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !scan_raw[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (!kvfit_is_plan_device(t, cpu_only)) continue;
            LLAMA_LOG_PLANNING("        %-4zu %-6s free=%14llu  %s\n",
                               i, kvfit_dev_type_str(t),
                               (unsigned long long)meas.scan_measure[i].free_bytes,
                               ggml_backend_dev_name(dev));
        }
    }

    // ---- per-device decomposition of the measured delta -------------------------
    // The free-VRAM delta covers: weights + probe-KV + SSM + compute buffers (at
    // probe width) + kernels (shader binaries etc.). Compute buffers are read
    // directly from llama_get_memory_breakdown at lo/hi pool — everything else
    // left over is the kernels residual. The base-buffer is the width-0
    // intercept of those measured compute buffers (lo - (hi - lo)); it is a
    // separate quantity, never derived from the kernels residual.
    if (params->options & LLAMA_KV_FIT_DEBUG) LLAMA_LOG_PLANNING("\n      overhead decomposition: "
                        "kernels = delta - weights - probe-KV - SSM(state rows x bytes/row) - compute(lo)\n"
                        "        base-buffer = compute-lo - (compute-hi - compute-lo), per measured context\n");
    const uint64_t probe_cells = ((uint64_t)n_ctx_measure + 255u) / 256u * 256u; // GGML_PAD, as the cache does
    size_t nd_gpus = 0;
    for (size_t i = 0; i < scan_raw.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !scan_raw[i].ok || i >= meas.scan_measure.size() || !meas.scan_measure[i].ok) continue;
        const auto t = ggml_backend_dev_type(dev);
        if (kvfit_is_plan_device(t, cpu_only)) ++nd_gpus;
    }
    std::vector<uint64_t> d_delta(nd_gpus), d_weights(nd_gpus), d_probe_kv(nd_gpus), d_kernels(nd_gpus);
    std::vector<uint64_t> d_ssm_bpr(nd_gpus, 0), d_kv_bpc(nd_gpus, 0), d_mtp_bpc(nd_gpus, 0);
    std::vector<uint64_t> d_base(nd_gpus, 0);   // base-buffer per GPU (width-0 intercept of the measured compute buffers)

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
            if (!kvfit_is_plan_device(t, cpu_only)) continue;
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
            if (!kvfit_is_plan_device(t, cpu_only)) continue;

            const uint64_t delta   = scan_raw[i].free_bytes > meas.scan_measure[i].free_bytes
                                   ? scan_raw[i].free_bytes - meas.scan_measure[i].free_bytes : 0;
            const uint64_t weights = i < meas.usage.size() ? meas.usage[i].bytes : 0;
            const uint64_t kv_bpc  = i < meas.rates.size() ? meas.rates[i].kv_bytes_per_cell : 0;
            const uint64_t mtp_bpc = (mtp_measured && i < meas.rates.size()) ? meas.rates[i].mtp_kv_bytes_per_cell : 0;
            const uint64_t ssm_bpr = i < meas.rates.size() ? meas.rates[i].ssm_bytes_per_row : 0;

            // Compute buffers live at probe width, read directly from the
            // llama_get_memory_breakdown lo/hi measurements (main + MTP ctx).
            // These are what the deployment's graph reserves before its own pool.
            const uint64_t compute_lo_main = (i < meas.compute_lo_main.size()) ? meas.compute_lo_main[i] : 0;
            const uint64_t compute_hi_main = (i < meas.compute_hi_main.size()) ? meas.compute_hi_main[i] : 0;
            const uint64_t compute_lo_mtp  = (i < meas.compute_lo_mtp.size())  ? meas.compute_lo_mtp[i]  : 0;
            const uint64_t compute_hi_mtp  = (i < meas.compute_hi_mtp.size())  ? meas.compute_hi_mtp[i]  : 0;
            const uint64_t compute_lo_probe = compute_lo_main + compute_lo_mtp;

            // probe-KV: the measurement context's own pools (main + MTP draft when
            // measured). Unified pools are shared across streams — streams do not
            // multiply the size. SSM state rows: charged here AND in the upper
            // limit, at the deployment's planned row count.
            d_probe_kv[gi] = probe_cells * (kv_bpc + mtp_bpc);
            const uint64_t ssm_alloc = (uint64_t)state_rows * ssm_bpr;

            // kernels = residual overhead after subtracting everything else that
            // the measured delta accounted for. This is the shader/binaries fixed
            // cost, independent of the compute buffers.
            const uint64_t accounted_so_far = weights + d_probe_kv[gi] + ssm_alloc + compute_lo_probe;
            d_kernels[gi] = (delta > accounted_so_far) ? delta - accounted_so_far : 0;

            // base-buffer = width-0 intercept of the measured compute buffers,
            // per context: lo - (hi - lo). Combined across main + MTP ctx.
            // For compute(N) = base + rate*N with hi = 2*lo:
            //   lo - (hi - lo) = 2*lo - hi = 2*(base+rate*lo) - (base+rate*hi) = base
            const uint64_t d_bytes_main = (compute_hi_main > compute_lo_main) ? compute_hi_main - compute_lo_main : 0;
            const uint64_t base_main = (compute_lo_main >= d_bytes_main) ? compute_lo_main - d_bytes_main : 0;

            const uint64_t d_bytes_mtp = (compute_hi_mtp > compute_lo_mtp) ? compute_hi_mtp - compute_lo_mtp : 0;
            const uint64_t base_mtp = (compute_lo_mtp >= d_bytes_mtp) ? compute_lo_mtp - d_bytes_mtp : 0;

            d_base[gi] = base_main + base_mtp;

            d_delta[gi]   = delta;
            d_weights[gi] = weights;
            d_ssm_bpr[gi] = ssm_bpr;
            d_kv_bpc[gi]  = kv_bpc;
            d_mtp_bpc[gi] = mtp_bpc;

            if (params->options & LLAMA_KV_FIT_DEBUG) {
                LLAMA_LOG_PLANNING("        %s: delta=%llu weights=%llu probe-KV=%llu "
                                   "(cells(%llu) x bpc(%llu)) SSM=%llu compute-lo=%llu kernels=%llu\n",
                                   ggml_backend_dev_name(dev),
                                   (unsigned long long)d_delta[gi], (unsigned long long)d_weights[gi],
                                   (unsigned long long)d_probe_kv[gi],
                                   (unsigned long long)probe_cells, (unsigned long long)(kv_bpc + mtp_bpc),
                                   (unsigned long long)ssm_alloc,
                                   (unsigned long long)compute_lo_probe,
                                   (unsigned long long)d_kernels[gi]);
                LLAMA_LOG_PLANNING("        %s: base-buffer = main[lo(%llu) - (hi(%llu)-lo(%llu))] "
                                   "+ mtp[lo(%llu) - (hi(%llu)-lo(%llu))] = %llu\n",
                                   ggml_backend_dev_name(dev),
                                   (unsigned long long)compute_lo_main, (unsigned long long)compute_hi_main,
                                   (unsigned long long)compute_lo_main,
                                   (unsigned long long)compute_lo_mtp, (unsigned long long)compute_hi_mtp,
                                   (unsigned long long)compute_lo_mtp,
                                   (unsigned long long)d_base[gi]);
            }
            ++gi;
        }
    }

    // ---- budgets: what each GPU leaves for tensors + KV pool ---------------------
    // Independent fixed costs per device, measured separately in the decomposition
    // step: kernels (shader binary residual) and base-buffer (width-0 intercept of
    // the measured compute buffers), plus the reserve (kept-free slice). The budget
    // is what's left; the solver then charges tensors, SSM state, pool cells and the
    // pool-proportional compute-buffer rate at target width against it:
    //     budget = free - kernels - base-buffer - reserve
    const double keep_free = 1.0 - params->vram_fraction;

    LLAMA_LOG_PLANNING("\n      split budget: reserve = total x keep_free(%.2f)%s\n"
                       "                    budget  = free - kernels - base-buffer - reserve\n",
                       keep_free, (params->margins ? " + margin" : ""));
    std::vector<uint64_t> budgets(nd_gpus);
    std::vector<uint64_t> reserves(nd_gpus);   // solver header: per-GPU reserve kept back
    std::vector<uint64_t> frees(nd_gpus);      // pre-load free VRAM (cost-breakdown ledger)
    std::vector<std::string> names;
    int64_t total_budget = 0, total_weights = 0;
    uint64_t ssm_per_row_total = 0, cell_size_total = 0;
    {
        size_t gi = 0;
        for (size_t i = 0; i < scan_raw.size() && gi < nd_gpus; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || !scan_raw[i].ok) continue;
            const auto t = ggml_backend_dev_type(dev);
            if (!kvfit_is_plan_device(t, cpu_only)) continue;

            int64_t total   = (int64_t)scan_raw[i].total_bytes;
            int64_t free    = (int64_t)scan_raw[i].free_bytes;
            int64_t reserve  = (int64_t)((double)total * keep_free);
            if (params->margins) reserve += (int64_t)params->margins[i]; // indexed by backend device order
            // kernels and base-buffer are independently measured in the
            // decomposition step above; the cost-breakdown table shows both.
            int64_t budget   = free - (int64_t)d_kernels[gi] - (int64_t)d_base[gi] - reserve;

            budgets[gi]  = budget > 0 ? (uint64_t)budget : 0;  // negative = overcommitted GPU
            reserves[gi] = (uint64_t)reserve;                  // what was kept back from this device's total
            frees[gi]    = free > 0 ? (uint64_t)free : 0;      // pre-load free VRAM for the ledger
            names.push_back(ggml_backend_dev_name(dev));
            total_budget  += budget;
            total_weights += (int64_t)d_weights[gi];
            ssm_per_row_total += d_ssm_bpr[gi];
            cell_size_total   += d_kv_bpc[gi] + d_mtp_bpc[gi] + bufs_bpc[gi];

            LLAMA_LOG_PLANNING("        %s:\n", names.back().c_str());
            if (params->margins) {
                LLAMA_LOG_PLANNING("          reserve     = total(%llu) × keep_free(%.2f) + margin(%llu) = %llu\n",
                                   (unsigned long long)total, keep_free,
                                   (unsigned long long)params->margins[i],
                                   (unsigned long long)reserves[gi]);
            } else {
                LLAMA_LOG_PLANNING("          reserve     = total(%llu) × keep_free(%.2f) = %llu\n",
                                   (unsigned long long)total, keep_free,
                                   (unsigned long long)reserves[gi]);
            }
            LLAMA_LOG_PLANNING("          budget      = free(%llu) - kernels(%llu) - base-buffer(%llu) "
                               "- reserve(%llu) = %llu\n",
                               (unsigned long long)free, (unsigned long long)d_kernels[gi],
                               (unsigned long long)d_base[gi], (unsigned long long)reserves[gi],
                               (unsigned long long)budgets[gi]);
            ++gi;
        }
    }

    // ---- cost breakdown of the provisional split (before the bisection) -----------
    // Multi-GPU only: with a single planning device there is one placement — no
    // provisional layout exists to compare the plan against, so there is nothing
    // provisional to show.
    if (nd_gpus >= 2) {
        kvfit_print_provisional_split(meas.usage);   // where the measurement load actually placed them
        std::vector<uint64_t> ssm_states(nd_gpus, 0);
        for (size_t g = 0; g < nd_gpus; ++g) {
            ssm_states[g] = d_ssm_bpr[g] * (uint64_t)state_rows;
        }
        kvfit_print_cost_breakdown(params, "cost breakdown [provisional split]", names, frees, d_kernels, d_base,
                                   reserves, budgets, d_weights, ssm_states,
                                   d_kv_bpc, d_mtp_bpc, buf_bpc, mtp_buf_bpc);
    }

    // ---- theoretical upper limit: the perfect-split ceiling ----------------------
    // CPU-only deployment: the tensor readback cannot attribute mmap'd weights to
    // the CPU device (CPU_Mapped buft carries no device pointer), so charge the
    // GGUF metadata weight sum instead — a placement-independent logical cost.
    if (cpu_only) {
        total_weights = (int64_t)gguf.model_bytes;
        LLAMA_LOG_PLANNING("      weights (CPU-only): from GGUF metadata = %llu bytes\n",
                           (unsigned long long)gguf.model_bytes);
    }
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
        LLAMA_LOG_ERROR("kvfit: even a perfect split fits fewer than 256 KV cells — "
                        "not a viable deployment\n");
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
    plan.n_ctx_train = model_peek->hparams.n_ctx_train;   // native context: the pool usage cap
    llama_model_free(model_peek);

    // ---- solve for the pool size and layout -------------------------------------
    uint32_t n_blocks = 0;
    std::vector<int> first_layer(budgets.size(), 0);
    // (compute-buffer rates computed earlier in the budgets section above;
    // bufs_bpc = main + MTP draft — what the solver charges per cell)
    const uint64_t last_gpu_extra = (uint64_t)gguf.norm_bytes + (uint64_t)gguf.output_bytes;
    if (nd_gpus <= 1) {
        // Single planning device (one GPU or CPU-only): there is no partition to
        // find — every layer sits on it. The perfect-split ceiling above is then
        // the exact answer: budget minus tensors minus SSM state, divided by the
        // per-cell cost. The bisection solver exists to search for a placement
        // that fits the per-device budgets and has nothing left to do here.
        n_blocks = n_max;
    } else {
        n_blocks = kvfit_solve(n_max, layer_costs, budgets, reserves, names, last_gpu_extra,
                               bufs_bpc);
        if (n_blocks == 0) {
            LLAMA_LOG_ERROR("kvfit: solver found no placement — even the perfect split cannot "
                            "hold all layers on the GPUs\n");
            return plan;
        }

        // ---- winning layout: re-run the greedy walk at n_blocks -----------------
        // llama's loader assigns layer il to the first GPU whose CUMULATIVE split
        // fraction exceeds il/(N+1) (llama-model.cpp get_layer_buft_list, with
        // n_gpu_layers = -1). To make it reproduce the solver's ranges exactly,
        // each boundary a_g (first layer on GPU g) becomes cum[g-1] = a_g / (N+1);
        // llama normalizes splits, so tensor_split holds those cumulative values
        // directly. The output pseudo-layer N lands on the last GPU with any share.
        const std::vector<size_t> walk = kvfit_greedy_walk(n_blocks, layer_costs, budgets, bufs_bpc);
        for (size_t g = 0; g < walk.size() && g < first_layer.size(); ++g) {
            first_layer[g] = (int)walk[g];
        }
    }

    const uint32_t n_cells = n_blocks * 256u;
    if (n_cells < params->min_pool_cells) {
        LLAMA_LOG_WARN("kvfit: planned pool %u cells is below the viability floor "
                       "(min_pool_cells=%u) — a tiny KV cache for this deployment\n",
                       (unsigned)n_cells, (unsigned)params->min_pool_cells);
    }

    plan.ok          = true;
    plan.n_cells     = n_cells;
    plan.first_layer = std::move(first_layer);
    plan.names       = names;
    plan.cpu_only    = cpu_only;

    LLAMA_LOG_PLANNING("\n      ==> plan: pool %u cells, placement:\n", (unsigned)n_cells);
    for (size_t g = 0; g < plan.first_layer.size(); ++g) {
        const int a = plan.first_layer[g];
        const int b = (g + 1 < plan.first_layer.size()) ? plan.first_layer[g + 1] : (int)plan.n_layer_all;
        LLAMA_LOG_PLANNING("        %s: layers %d..%d%s\n", plan.names[g].c_str(), a, b - 1,
                           (a >= b) ? " (empty)" : "");
    }

    // ---- cost breakdown of the plan ------------------------------------------
    // Recomputed from the solver's per-layer costs over the chosen ranges;
    // the placement-independent terms (base-buffer, compute-buffer rates) are
    // shared with the provisional table when one exists (multi-GPU). For a
    // single planning device this is the only table: that is the deployment.
    {
        const size_t nd = plan.first_layer.size();
        std::vector<uint64_t> p_wts(nd, 0), p_ssm(nd, 0), p_kv(nd, 0), p_mtp(nd, 0);
        for (size_t g = 0; g < nd; ++g) {
            const int a = plan.first_layer[g];
            const int b = (g + 1 < nd) ? plan.first_layer[g + 1] : (int)plan.n_layer_all;
            for (int l = a; l < b; ++l) {
                p_wts[g] += layer_costs[l].weights;
                p_ssm[g] += layer_costs[l].ssm_state;
                p_kv[g]  += layer_costs[l].kv_bpc;
                p_mtp[g] += layer_costs[l].mtp_kv_bpc;
            }
        }
        // the last GPU also takes the global tensors (norm + output / lm_head);
        // the budget ledger columns are placement-independent — same for both tables
        if (nd > 0) p_wts[nd - 1] += last_gpu_extra;
        kvfit_print_cost_breakdown(params, (nd_gpus >= 2) ? "cost breakdown [final plan]" : "cost breakdown [plan]",
                                   plan.names, frees, d_kernels, d_base,
                                   reserves, budgets, p_wts, p_ssm,
                                   p_kv, p_mtp, buf_bpc, mtp_buf_bpc);
    }

    // ---- Verification run: prove the plan works at runtime (opt-in) ----------
    if ((params->options & LLAMA_KV_FIT_VERIFY) &&
        !kvfit_verify_plan(path, mparams, cparams, measure_mtp,
                           plan.first_layer, n_cells)) {
        LLAMA_LOG_ERROR("kvfit: verification failed — this plan does not deploy, "
                        "no plan reported\n");
        plan.ok = false;
    }

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

namespace {

// The single result of llama_kv_split() — a plain global owned by the library;
// there are never two concurrent planning calls. layer_bounds stores the
// planned placement; it is valid until the next call overwrites it.
llama_kv_fit_result g_kvfit_result;
std::vector<int>    g_kvfit_layer_bounds;

} // anonymous namespace

// The deployment report (model header + per-layer table) — what the kv-split
// tool used to print itself; printed before plan computation, like the tool
// did. One metadata-only peek load: no_alloc keeps VRAM untouched.
// Header + table are rendered OFFLINE into one buffer and emitted as ONE
// indivisible block (kvfit_block_emit) — nothing can interleave inside it.
static void kvfit_print_deployment_report(const char * path_model,
                                          const llama_context_params * cparams,
                                          const struct llama_kv_fit_params * params) {
    char * rep_buf = nullptr;
    size_t rep_len = 0;

    llama_model_params mp_info = llama_model_default_params();
    mp_info.no_alloc  = true;
    mp_info.load_mode = LLAMA_LOAD_MODE_NONE;
    mp_info.load_mtp  = true;   // deployment shape; avoids "unused tensor" noise

    llama_model * m_info = llama_model_load_from_file(path_model, mp_info);
    if (!m_info) {
        LLAMA_LOG_ERROR("kvfit: could not open %s for the model-info table\n", path_model);
        return;   // the planning load below will report the real error
    }

    char kf[24], nb[24], nu[24], sm[24], rs[24];
    snprintf(kf, sizeof(kf), "%.2f", 1.0 - params->vram_fraction);   // keep_free = 1 - vram_fraction
    snprintf(nb, sizeof(nb), "%u", cparams->n_batch);
    snprintf(nu, sizeof(nu), "%u", cparams->n_ubatch);
    snprintf(sm, sizeof(sm), "%u", cparams->n_seq_max);
    snprintf(rs, sizeof(rs), "%u", cparams->n_rs_seq);
    struct mi_header_opts opts = { kf, nb, nu, sm, rs };

    const bool flash_attn = cparams->flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mi_model_info_t info = mi_model_info_new(path_model, m_info);
    if (info) {
        FILE * tmp = open_memstream(&rep_buf, &rep_len);
        if (tmp) {
            if (!mi_print_header(tmp, info,
                                 (enum ggml_type)cparams->type_k, (enum ggml_type)cparams->type_v, &opts)
                || !mi_print_layer_table(tmp, info,
                                         (enum ggml_type)cparams->type_k, (enum ggml_type)cparams->type_v,
                                         flash_attn)) {
                LLAMA_LOG_WARN("kvfit: model-info table print failed for %s\n", path_model);
            }
            fflush(tmp);
            fclose(tmp);
        } else {
            LLAMA_LOG_ERROR("kvfit: report buffer allocation failed for %s\n", path_model);
        }
        mi_model_info_free(info);
    } else {
        LLAMA_LOG_ERROR("kvfit: could not open %s for the model-info table\n", path_model);
    }
    llama_model_free(m_info);

    if (rep_buf) {
        kvfit_block_emit(params, rep_buf, rep_len);
        free(rep_buf);
    }
}

LLAMA_API const struct llama_kv_fit_result * llama_kv_split(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const struct llama_kv_fit_params * params,
        int32_t target_cells) {
    memset(&g_kvfit_result, 0, sizeof(g_kvfit_result));
    g_kvfit_layer_bounds.clear();

    if (!path_model || !mparams || !cparams || !params) {
        snprintf(g_kvfit_result.error_msg, sizeof(g_kvfit_result.error_msg),
                 "kv-split: missing arguments");
        return &g_kvfit_result;
    }

    // the deployment report (model header + per-layer table), opt-in via options
    if (params->options & LLAMA_KV_FIT_PRINT_TABLES) {
        kvfit_print_deployment_report(path_model, cparams, params);
    }

    // 1. placement + capacity — ONE measurement run, computed independently of
    //    target_cells (the split never depends on the supplied cache size):
    kvfit_plan plan = kvfit_get_plan(path_model, mparams, cparams, params);

    if (!plan.ok) {
        // the planner already reported why nothing fits; carry a short note too
        snprintf(g_kvfit_result.error_msg, sizeof(g_kvfit_result.error_msg),
                 "kv-split: no viable deployment (see the reports above)");
        return &g_kvfit_result;
    }

    // 2. THE single pool decision — the resource is never claimed twice:
    uint32_t pool;
    if (target_cells > 0) {
        if ((uint32_t)target_cells > plan.n_cells) {
            snprintf(g_kvfit_result.error_msg, sizeof(g_kvfit_result.error_msg),
                     "kv-split: requested pool of %d cells does not fit — max viable is %u",
                     target_cells, (unsigned)plan.n_cells);
            return &g_kvfit_result;
        }
        pool = (uint32_t)target_cells;   // exactly what the user asked
    } else {
        pool = plan.n_cells;             // everything that fits
        // ... but never more than the parallel sequences can use: threads x the
        // model's native context size (auto-fit only — an explicit request is
        // validated against physical capacity, not this usage cap):
        if (plan.n_ctx_train > 0 && cparams->n_seq_max > 0) {
            const uint64_t use_cap = ((uint64_t)cparams->n_seq_max * plan.n_ctx_train) / 256u * 256u;
            if (use_cap < pool) {
                LLAMA_LOG_PLANNING("kvfit: capping pool %u -> %llu cells "
                                   "(%u sequences x %u native context)\n",
                                   (unsigned)pool, (unsigned long long)use_cap,
                                   (unsigned)cparams->n_seq_max, plan.n_ctx_train);
                pool = (uint32_t)use_cap;
            }
        }
    }

    // 3. fill THE decided result. The planned placement is returned in
    //    layer_bounds (library-owned, valid until the next call); the caller
    //    applies it to its model params when the plan's split — not a user-set
    //    one — is to be deployed:
    //        mparams->n_gpu_layers       = -1;   // all layers on devices (loader idiom)
    //        mparams->tensor_layer_bounds = result->layer_bounds;
    //        mparams->tensor_split       = nullptr;   // the boundaries supersede ratio logic
    // CPU-only deployments return no placement: everything trivially runs there.
    if (!plan.cpu_only) {
        g_kvfit_layer_bounds.assign(plan.first_layer.begin(), plan.first_layer.end());
    }

    g_kvfit_result.ok                = true;
    g_kvfit_result.n_cells           = (int32_t)pool;
    g_kvfit_result.n_devices_planned = plan.cpu_only ? 0u : (uint32_t)g_kvfit_layer_bounds.size();
    g_kvfit_result.layer_bounds      = g_kvfit_layer_bounds.empty() ? nullptr : g_kvfit_layer_bounds.data();

    LLAMA_LOG_PLANNING("kv-split: pool %d cells%s — apply layer_bounds to the real load's params\n",
                       (int)pool, plan.cpu_only ? " (CPU-only deployment)" : "");
    return &g_kvfit_result;
}
