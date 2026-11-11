// [FORK] Unified KV pool fitting for the main library.
//
// Provides the memory-probe + affine-solve machinery that computes how large a
// physical unified KV cache pool (n_kv_cache) can be while still fitting on the
// devices' VRAM, given a model's layer placement and any extra context that
// competes for the same memory (e.g. an MTP draft context).
//
// External tools such as fllama call these through llama-ext.h without linking
// libllama-common; llama-server's common_fit_params() reuses them too.

#include "llama-impl.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Query VRAM on every GPU-like device at fit time. Historical note: an earlier
// rocky2 run suggested ggml_backend_dev_memory() returns 0/0 before any
// backend allocation ("lazy hipInit"), so this did a dummy 4 KB alloc+free first.
// On 2026-11-10 the pre-allocation read already returned real values, and
// Michael's CUDA experience matches (no allocation needed); the dummy alloc is
// therefore dropped. If a future run shows total=0 again, restore it.
static void llama_kv_fit_warmup_devices() {
    static bool done = false;
    if (done) {
        return;
    }

    const size_t ndevs = ggml_backend_dev_count();
    fprintf(stderr, "[KV-FIT-DBG] warmup: ggml_backend_dev_count() = %zu\n", ndevs);
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue; // only GPU-like backends report VRAM worth printing
        }

        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);

        fprintf(stderr, "[KV-FIT-DBG] warmup: dev %s (type=%d) VRAM total %lld / free %lld\n",
                ggml_backend_dev_name(dev), (int)type,
                (long long)total, (long long)free);
    }

    done = true;
}

// [KV-FIT-DBG] instrumentation: dump raw total/free VRAM of every device, as-is.
static void llama_kv_fit_dbg_vram_all(const char * tag) {
    fprintf(stderr, "[KV-FIT-DBG] vram@%s:\n", tag);
    const size_t ndevs = ggml_backend_dev_count();
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        fprintf(stderr, "[KV-FIT-DBG]   %s: total=%lld free=%lld\n",
                ggml_backend_dev_name(dev), (long long)total, (long long)free);
    }
}

// Load a model + context with no_alloc and return the per-device memory
// breakdown plus each device's total/free memory. The last entry is the host.
static std::vector<llama_device_memory_data> llama_kv_fit_probe(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert) {
    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    fprintf(stderr, "[KV-FIT-DBG] probe enter: mparams=%p no_alloc=%d load_mode=%d n_ctx=%llu kv_cache=%u\n",
            reinterpret_cast<const void*>(mparams), (int)mparams_copy.no_alloc, (int)mparams_copy.load_mode,
            (unsigned long long)cparams->n_ctx, (unsigned)cparams->n_kv_cache);

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }
    llama_kv_fit_dbg_vram_all("after model load");

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("failed to create llama_context from model");
    }
    llama_kv_fit_dbg_vram_all("after ctx init");

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("no CPU backend found");
        }
        size_t free, total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends (BLAS etc.) report 0/0 and rely on the
        // host-memory fallback. For GPU-like backends keep 0/0 so fitting does not
        // assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LLAMA_LOG_WARN("%s: device %s did not report memory; it will not be used\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl     += llama_model_n_layer_nextn(model);
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    llama_free(ctx);
    llama_model_free(model);
    llama_kv_fit_dbg_vram_all("after free");

    return ret;
}

// Measure per-device VRAM consumed by everything a real context needs BEYOND what
// the no_alloc probes already account for (weights via `model`, KV rows + recurrent
// state via `rate`/`fixed`, sched graph buffers via `compute`). Evidence (rocky2
// 2026-11-11, u256 OOM run): the output/sampling buffers and the scheduler compute
// buffers are allocated DURING llama_init_from_model - after a ctx-init-anchored
// baseline they are invisible, which left ~0.9 GiB uncounted on dev1 while the pool
// was sized to the cap and OOMed at the first decode.
//
// Design: free-memory snapshot right after the real model load (F0) - this is the
// exact landscape a real run will face, probe residuals included. Then init both
// contexts (tiny pools) and decode each once; F_last is the post-everything state.
// Returns:
//   overhead[g] = F0[g] - F_last[g]           (raw bytes of ctx init + decodes)
//   baseline[g] = F0[g]                       (caller logs it; pool_room math still
//                                              uses the no_alloc probe's free values)
// The solver subtracts the measured contexts' own KV rows (fixed + rate*n_probe_tok)
// from overhead before using it, so nothing is counted twice.
struct llama_kv_fit_runtime_result {
    bool               ok = false;       // probe actually ran (model + ctxs + decodes)
    uint32_t           probe_cells = 0;  // pool size used by the measurement contexts
    std::vector<uint64_t> overhead; // per-device, host entry 0
    std::vector<size_t>   baseline_free; // per-device free at model-load point, host 0
};

static llama_kv_fit_runtime_result llama_kv_fit_runtime_overhead(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const llama_kv_fit_extra * extra,
        std::vector<ggml_backend_dev_t> & devs) {
    const size_t nd = devs.size(); // includes the host entry
    llama_kv_fit_runtime_result ret;
    std::vector<uint64_t> overhead(nd, 0);
    std::vector<size_t> baseline_free(nd, 0);

    const llama_model_params mp_real = *mparams; // real weights: no no_alloc override
    llama_model * model = llama_model_load_from_file(path_model, mp_real);
    if (model == nullptr) {
        LLAMA_LOG_WARN("%s: runtime overhead probe failed to load model\n", __func__);
        ret.overhead      = std::move(overhead);
        ret.baseline_free = std::move(baseline_free);
        return ret; // ok = false
    }

    // Baseline: everything already resident here (probe leftovers via the caching
    // allocator included) is what a real run must live with.
    for (size_t g = 0; g < devs.size() - 1; g++) {
        size_t f, t;
        ggml_backend_dev_memory(devs[g], &f, &t);
        baseline_free[g] = f;
    }

    // Effective worst-case ubatch: what the scheduler reserves/launches at runtime is
    // min(n_ctx, n_ubatch). Decode exactly that many tokens so batch-dependent lazy
    // buffers (cuBLAS/HIPBLASLt workspaces) allocate at their real size. Fall back to a
    // safe default when the caller leaves it unset.
    uint32_t n_probe_tok = 4096;
    if (cparams->n_ubatch > 0) {
        n_probe_tok = cparams->n_ubatch;
        if (cparams->n_ctx > 0 && cparams->n_ctx < n_probe_tok) {
            n_probe_tok = cparams->n_ctx;
        }
    }

    llama_context_params cp = *cparams;
    // OOM-proof: the measurement pool only needs to hold one worst-case ubatch. It stays
    // tiny compared to the real pool (hundreds of thousands of cells) even at large ubatch.
    cp.n_kv_cache = std::max<uint32_t>(n_probe_tok, 256);
    if (cp.n_batch < n_probe_tok) { // the probe decode below wants all tokens in one batch
        cp.n_batch = n_probe_tok;
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        LLAMA_LOG_WARN("%s: runtime overhead probe failed to init context\n", __func__);
        llama_model_free(model);
        ret.overhead      = std::move(overhead);
        ret.baseline_free = std::move(baseline_free);
        return ret; // ok = false
    }
    ret.probe_cells = cp.n_kv_cache;

    // One worst-case decode: content is irrelevant; the point is to make every kernel
    // type run at full ubatch size so all lazy buffers are allocated before the deltas.
    {
        std::vector<llama_token> toks(n_probe_tok, 1); // any in-vocab ids work
        llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) n_probe_tok);
        llama_decode(ctx, batch);
    }

    // Mirror fllama's deployment: when an MTP context shares this model it adds its own
    // contexts and kernels (nextn layers), so measure those first-use allocations too.
    if (extra != nullptr && extra->shares_model) {
        llama_context_params cpe = *extra->cparams;
        cpe.n_kv_cache  = std::max<uint32_t>(n_probe_tok, 256);
        cpe.ctx_other   = ctx;
        if (cpe.n_batch < n_probe_tok) { // keep the probe decode in one batch
            cpe.n_batch = n_probe_tok;
        }
        llama_context * mctx = llama_init_from_model(model, cpe);
        if (mctx != nullptr) {
            // The MTP graph takes a hidden-state input per token (llama-graph.cpp:
            // llm_graph_input_embd_h::set_input asserts ubatch->embd_state), so this
            // decode must use the batch_ext API with embd_state set, like fllama does.
            // Content is irrelevant for the overhead measurement; only the layout matters
            // (contiguous n_embd_out floats per token).
            const int32_t n_state = llama_model_n_embd_out(model);

            std::vector<llama_token> toks(n_probe_tok, 1);
            std::vector<float> state((size_t) n_probe_tok * (size_t) n_state, 0.0f);

            llama_batch_ext * bext = llama_batch_ext_init(mctx);
            for (uint32_t t = 0; t < n_probe_tok; ++t) {
                const int32_t idx = llama_batch_ext_add_token(bext, 0, toks[t]);
                if (idx < 0) {
                    break; // n_tokens_max reached: still a valid (shorter) worst-case probe
                }
                const llama_pos pos = (llama_pos) t;
                llama_batch_ext_set_pos(bext, idx, &pos);

                struct llama_embd embd;
                embd.data   = state.data() + (size_t) t * n_state;
                embd.n_rows = 1;
                embd.n_embd = (size_t) n_state;
                llama_batch_ext_set_embd_state(bext, idx, embd);

                llama_batch_ext_set_output_logits(bext, idx, true);
            }
            llama_process(mctx, LLAMA_PROCESS_TYPE_DECODE, bext);
            llama_batch_ext_free(bext);
            llama_free(mctx);
        } else {
            LLAMA_LOG_WARN("%s: runtime overhead probe could not init the extra context\n", __func__);
        }
    }

    for (size_t g = 0; g < devs.size() - 1; g++) {
        size_t f, t;
        ggml_backend_dev_memory(devs[g], &f, &t);
        if (baseline_free[g] > f) {
            overhead[g] = baseline_free[g] - f;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    ret.ok          = true; // main ctx measured; the production config that matters is covered
    ret.overhead    = std::move(overhead);
    ret.baseline_free = std::move(baseline_free);
    return ret;
}

// Pure computation: solve the maximum unified KV pool size from two no_alloc
// probes taken at N0 and N1 cells. Has no device/model dependencies of its own;
// everything is measured into the dmds vectors by the caller.
// runtime_overhead[id] is the RAW measured bytes (ctx init + decodes, see
// llama_kv_fit_runtime_overhead): it already contains the sched compute buffers and
// the measurement contexts' own KV rows, so the solver adds only model + the residual
// (raw - fixed - rate*n_probe_cells). n_probe_cells = pool size of the measurement
// contexts (0 when no overhead data is available).
static uint32_t llama_kv_fit_max_pool_solve(
        const std::vector<llama_device_memory_data> & dmds_low,
        const std::vector<llama_device_memory_data> & dmds_high,
                                uint32_t N0, uint32_t N1,
                                const size_t * margins,      // per-device margin to keep free (bytes)
                                const uint64_t * runtime_overhead, // raw measured ctx-init+decode bytes per device
                                uint32_t n_probe_cells,        // pool size of the measurement contexts
                                double vram_fraction) {      // hard cap: max usage of each device's total memory
    fprintf(stderr, "[KV-FIT-DBG] solve: %zu device entries (last = host)\n", dmds_low.size());
    for (size_t id = 0; id < dmds_low.size(); id++) {
        fprintf(stderr, "[KV-FIT-DBG] solve entry %zu: model=%lld compute=%lld context(lo/hi)=%lld/%lld total=%lld free=%lld\n",
                id,
                (long long)dmds_low[id].mb.model,   (long long)dmds_low[id].mb.compute,
                (long long)dmds_low[id].mb.context, (long long)dmds_high[id].mb.context,
                (long long)dmds_low[id].total,      (long long)dmds_low[id].free);
    }
    if (dmds_low.empty() || dmds_low.size() != dmds_high.size()) {
        fprintf(stderr, "[KV-FIT-DBG] solve: giving up - probe vectors empty or size mismatch (%zu vs %zu)\n",
                dmds_low.size(), dmds_high.size());
        return 0;
    }

    const size_t nd = dmds_low.size() - 1; // devices only; the host entry (last) is excluded on purpose:
                                           // auto-maximizing into system RAM stays an explicit option
    uint32_t n_max = UINT32_MAX;
    bool have_constraint = false;

    for (size_t id = 0; id < nd; id++) {
        const auto & lo = dmds_low[id].mb;
        const auto & hi = dmds_high[id].mb;

        if (lo.model != hi.model) {
            fprintf(stderr, "[KV-FIT-DBG] solve: giving up - model bytes differ between probes on device %zu (%lld vs %lld)\n",
                    id, (long long)lo.model, (long long)hi.model);
            return 0;
        }

        // Model bytes are a real allocation measurement and MUST match.
        const uint64_t compute = std::max(lo.compute, hi.compute);

        const int64_t delta_ctx = (int64_t) hi.context - (int64_t) lo.context;
        const int64_t dns       = (int64_t) (N1 - N0);
        if (delta_ctx < 0 || delta_ctx % dns != 0) {
            fprintf(stderr, "[KV-FIT-DBG] solve: giving up - non-affine context size on device %zu (%lld -> %lld)\n",
                    id, (long long)lo.context, (long long)hi.context);
            return 0;
        }
        const int64_t rate  = delta_ctx / dns;
        const int64_t fixed = rate > 0 ? (int64_t) lo.context - rate * (int64_t) N0 : (int64_t) lo.context;

        if (rate == 0) {
            fprintf(stderr, "[KV-FIT-DBG] solve: device %zu hosts no KV layers (rate=0), no constraint\n", id);
            continue; // this device hosts no KV layers: no pool constraint
        }

        const llama_device_memory_data & dmd = dmds_low[id];
        if (dmd.total <= 0 || dmd.free <= 0 || dmd.free > dmd.total) {
            fprintf(stderr, "[KV-FIT-DBG] solve: giving up - device %zu invalid memory but hosts KV layers (total=%lld free=%lld)\n",
                    id, (long long)dmd.total, (long long)dmd.free);
            return 0;
        }

        // what this device may still allocate for the pool:
        //   - hard cap so total usage never exceeds vram_fraction of the device (driver headroom)
        //   - limited by measured free memory, minus the margins to keep for other purposes
        const int64_t used       = dmd.total > dmd.free ? dmd.total - dmd.free : 0;
        const int64_t cap_driver = (int64_t) ((double) dmd.total * vram_fraction) - used;
        const int64_t margin_i   = margins != nullptr ? (int64_t) margins[id] : 0;
        const int64_t cap_free   = (int64_t) dmd.free - margin_i;
        const int64_t pool_room  = std::min(cap_driver, cap_free);

        // non-pool bytes. Two accounting modes:
        // (a) measured window available (llama_kv_fit_max_pool): model + (raw - rate*n_pc).
        //     The window carries everything a real run pays before the pool grows -
        //     output/sampling buffers, recurrent state, sched compute buffers, kernels -
        //     so no separate fixed/compute terms (they would double-count).
        // (b) no measurement (Phase D split-consistency check): model + compute + fixed from
        //     the no_alloc breakdowns, same basis Phase C fits against.
        int64_t rt_oh_used = 0;
        int64_t non_pool;
        if (runtime_overhead != nullptr) {
            if (runtime_overhead[id] > 0) {
                rt_oh_used = (int64_t) runtime_overhead[id] - rate * (int64_t) n_probe_cells;
                if (rt_oh_used < 0) {
                    rt_oh_used = 0; // allocator recycling edge: keep raw measurement, never negative
                }
            }
            non_pool = (int64_t) lo.model + rt_oh_used;
        } else {
            non_pool = (int64_t) lo.model + (int64_t) compute + fixed;
        }

        uint32_t n_dev = 0;
        if (non_pool < pool_room) {
            const int64_t cells = (pool_room - non_pool) / rate;
            n_dev = (uint32_t) (cells / 256u * 256u);
        }

        fprintf(stderr, "[KV-FIT-DBG] solve: device %zu: total=%lld free=%lld used=%lld | cap_driver=%lld cap_free=%lld pool_room=%lld non_pool=%lld fixed=%lld rt_raw=%lld rt_used=%lld rate=%lld B/cell -> max %9u cells\n",
                id,
                (long long)dmd.total,  (long long)dmd.free, (long long)used,
                (long long)cap_driver, (long long)cap_free, (long long)pool_room,
                (long long)non_pool,   (long long)fixed,
                (long long)(runtime_overhead != nullptr ? runtime_overhead[id] : 0),
                (long long)rt_oh_used, (long long)rate, n_dev);

        have_constraint = true;
        n_max = std::min(n_max, n_dev);
    }

    if (!have_constraint) {
        fprintf(stderr, "[KV-FIT-DBG] solve: no device with a VRAM constraint hosts KV layers -> 0\n");
        return 0;
    }
    if (n_max < N0) {
        return 0; // not even one padding unit of pool fits anywhere
    }

    return n_max;
}

// See llama-ext.h.
LLAMA_API int32_t llama_kv_fit_max_pool(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const size_t * margins,
        const llama_kv_fit_extra * extra,
        double vram_fraction) {
    if (mparams == nullptr || cparams == nullptr || !cparams->kv_unified) {
        return 0; // pool math is only defined for unified mode
    }

    llama_kv_fit_warmup_devices();

    // two probes one padding unit apart decompose the per-device context bytes into
    // fixed + rate * n_cells; what they measure is what create_memory() allocates,
    // so SWA/MLA/hybrid quirks are handled by construction
    const uint32_t N0 = 256;
    const uint32_t N1 = 2 * 256;

    // shared between probe() and the runtime-overhead measurement below: the real-load
    // context must use exactly the devices the no_alloc probes measured
    std::vector<ggml_backend_dev_t> devs_main;

    auto probe = [&](uint32_t n_pool) -> std::vector<llama_device_memory_data> {
        llama_context_params cp_copy = *cparams;
        cp_copy.n_kv_cache = n_pool;

        devs_main.clear();
        uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;

        std::vector<llama_device_memory_data> dmds;
        fprintf(stderr, "[KV-FIT-DBG] probe(n_pool=%u): MAIN cparams n_seq_max=%u n_ctx=%llu n_kv_cache=%u kv_unified=%d ctx_type=%d\n",
                (unsigned)n_pool, (unsigned)cp_copy.n_seq_max,
                (unsigned long long)cp_copy.n_ctx, (unsigned)cp_copy.n_kv_cache,
                (int)cp_copy.kv_unified, (int)cp_copy.ctx_type);
        try {
            dmds = llama_kv_fit_probe(path_model, mparams, &cp_copy, devs_main, hp_ngl, hp_nct, hp_nex);
        } catch (const std::runtime_error & e) {
            fprintf(stderr, "[KV-FIT-DBG] MAIN probe threw: %s\n", e.what());
            LLAMA_LOG_ERROR("%s: probing failed for the main model: %s\n", __func__, e.what());
            return {};
        }

        if (extra != nullptr && !devs_main.empty()) {
            // the extra (e.g. MTP) context competes for the same devices; its pool follows
            // the probed size so its KV rows are part of the measured rate
            const uint32_t n_pool_saved = extra->cparams->n_kv_cache;
            extra->cparams->n_kv_cache  = n_pool;

            fprintf(stderr, "[KV-FIT-DBG] probe(n_pool=%u): EXTRA/MTP cparams ptr=%p n_seq_max=%u n_ctx=%llu n_kv_cache=%u kv_unified=%d ctx_type=%d\n",
                    (unsigned)n_pool, (void*)extra->cparams, (unsigned)extra->cparams->n_seq_max,
                    (unsigned long long)extra->cparams->n_ctx, (unsigned)extra->cparams->n_kv_cache,
                    (int)extra->cparams->kv_unified, (int)extra->cparams->ctx_type);
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_e = 0, nct_e = 0, nex_e = 0;
            std::vector<llama_device_memory_data> measured;
            try {
                measured = llama_kv_fit_probe(extra->path_model, extra->mparams, extra->cparams, devs_extra, ngl_e, nct_e, nex_e);
            } catch (const std::runtime_error & e) {
                fprintf(stderr, "[KV-FIT-DBG] EXTRA/MTP probe threw: %s\n", e.what());
                LLAMA_LOG_WARN("%s: probing failed for the extra model, continuing without it: %s\n", __func__, e.what());
            }
            extra->cparams->n_kv_cache = n_pool_saved;

            if (!measured.empty()) {
                // fold the extra model's memory into the matching devices (same mapping as common)
                std::vector<llama_device_memory_data> merged(devs_main.size() + 1);
                merged.back().mb = measured.back().mb;
                for (size_t je = 0; je < devs_extra.size(); je++) {
                    for (size_t id = 0; id < devs_main.size(); id++) {
                        if (devs_extra[je] == devs_main[id]) {
                            merged[id].mb.model   += measured[je].mb.model;
                            merged[id].mb.context += measured[je].mb.context;
                            merged[id].mb.compute += measured[je].mb.compute;
                            break;
                        }
                    }
                }
                if (extra->shares_model) {
                    for (llama_device_memory_data & dmd : merged) {
                        dmd.mb.model = 0; // weights already counted in the main model
                    }
                }
                for (size_t id = 0; id < dmds.size() && id < merged.size(); id++) {
                    dmds[id].mb.model   += merged[id].mb.model;
                    dmds[id].mb.context += merged[id].mb.context;
                    dmds[id].mb.compute += merged[id].mb.compute;
                }
            }
        }

        return dmds;
    };

    LLAMA_LOG_DEBUG("%s: probing for the maximum KV pool size (n_kv_cache = %u / %u)\n", __func__, N0, N1);
    const std::vector<llama_device_memory_data> dmds_low  = probe(N0);
    if (dmds_low.empty()) {
        return 0;
    }
    const std::vector<llama_device_memory_data> dmds_high = probe(N1);
    if (dmds_high.empty() || dmds_high.size() != dmds_low.size()) {
        LLAMA_LOG_ERROR("%s: inconsistent probe sizes (%zu vs %zu devices), giving up\n", __func__,
                dmds_high.size(), dmds_low.size());
        return 0;
    }

    // The no_alloc probes cannot see what a real context allocates beyond weight bytes:
    // output/sampling buffers, recurrent state, scheduler compute buffers and decode-time
    // kernels all land in VRAM during llama_init_from_model() / llama_decode(). Measure
    // them with one real load + tiny-pool ctxs + worst-case ubatch decodes at the solved
    // placement (F0 baseline right after model load). If the measurement itself fails we
    // must NOT size optimistically on incomplete data - return 0 and let the caller fall
    // back to an explicit --kv-cache-size or the conservative default.
    const llama_kv_fit_runtime_result rt =
            llama_kv_fit_runtime_overhead(path_model, mparams, cparams, extra, devs_main);
    fprintf(stderr, "[KV-FIT-DBG] runtime overhead probe: ok=%d probe_cells=%u\n", (int)rt.ok, rt.probe_cells);
    for (size_t g = 0; g + 1 < rt.overhead.size(); g++) {
        fprintf(stderr, "[KV-FIT-DBG]   - device %zu: raw=%8.2f MiB (baseline free %8.2f MiB)\n",
                g, rt.overhead[g] / 1048576.0, rt.baseline_free[g] / 1048576.0);
    }
    if (!rt.ok) {
        fprintf(stderr, "[KV-FIT-DBG] solve: giving up - runtime overhead probe failed (no unsafe sizing)\n");
        return 0;
    }

    const uint32_t n_max = llama_kv_fit_max_pool_solve(
            dmds_low, dmds_high, N0, N1, margins, rt.overhead.data(), rt.probe_cells, vram_fraction);
    if (n_max > 0) {
        LLAMA_LOG_INFO("%s: maximum unified KV pool size is %u cells\n", __func__, n_max);
    }
    return (int32_t) n_max;
}

// ============================================================================
// Automatic tensor-split optimizer — measured, not heuristic.
//
// Algorithm outline (see concept-auto-split-optimizer.txt):
//   Phase A: Place weights with N+1 different splits, probing each at 256/512 cells.
//   Phase B: Fit affine cost models per device from the sweep measurements.
//   Phase C: Water-fill solve for the split that maximizes the minimum per-device pool.
//   Phase D: Verify with one final probe pair at the solved split; log predicted vs measured.
// ============================================================================

namespace {

// Data structure to accumulate all probe results during Phase A.
struct PlacementSweepData {
    // Per-sweep probe results (low/high pairs for each placement)
    struct SweepResult {
        float shares[16];           // tensor_split used for this sweep
        std::vector<llama_device_memory_data> dmds_low;  // at N0 cells
        std::vector<llama_device_memory_data> dmds_high; // at N1 cells
    };

    std::vector<SweepResult> sweeps;
    size_t nd; // number of GPU devices

    void add_sweep(float * split,
                   std::vector<llama_device_memory_data> lo,
                   std::vector<llama_device_memory_data> hi) {
        SweepResult r;
        for (size_t i = 0; i < nd && i < llama_max_devices(); i++) {
            r.shares[i] = split[i];
        }
        r.dmds_low  = std::move(lo);
        r.dmds_high = std::move(hi);
        sweeps.push_back(std::move(r));
    }
};

// Phase B: Fit affine model overhead_g(m_g) = a_g + b_g * m_g per device from sweep points.
struct DeviceCostModel {
    double a_overhead; // intercept
    double b_overhead; // slope w.r.t. model bytes on this device
    double c_rate;     // KV rate: intercept
    double d_rate;     // KV rate: slope w.r.t. model bytes on this device

    // Layer-index placement: model bytes on this device as a function of its split
    // share x (fitted from the sweep; layers have non-uniform sizes, so bytes are not
    // proportional to the share — llama-model.cpp assigns by layer index ranges).
    double p_bytes; // slope w.r.t. share
    double q_bytes; // intercept

    bool valid = false;
};

// Least squares for y = a + b*x given (x, y) pairs. Returns a, b if at least 2 points.
bool fit_line(const std::vector<std::pair<double, double>> & pts,
              double & a_out, double & b_out,
              double * max_rel_residual = nullptr) {
    if (pts.size() < 2) {
        return false;
    }

    const size_t n = pts.size();
    double sum_x  = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;

    for (const auto & [x, y] : pts) {
        sum_x  += x;
        sum_y  += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-9) {
        // degenerate: constant or nearly constant
        a_out = sum_y / n;
        b_out = 0.0;
        return true;
    }

    b_out = (n * sum_xy - sum_x * sum_y) / denom;
    a_out = (sum_y - b_out * sum_x) / n;

    if (max_rel_residual != nullptr) {
        double max_err = 0.0;
        for (const auto & [x, y] : pts) {
            double pred = a_out + b_out * x;
            double ref  = std::max(std::abs(y), 1e-9);
            max_err = std::max(max_err, std::abs(pred - y) / ref);
        }
        *max_rel_residual = max_err;
    }

    return true;
}

// Collect (m_g, overhead_g), (m_g, rate_g) and (x_g, m_g) pairs for device g across sweeps.
void collect_device_points(
        const PlacementSweepData & data,
        size_t g,
        std::vector<std::pair<double, double>> & overhead_pts,
        std::vector<std::pair<double, double>> & rate_pts,
        std::vector<std::pair<double, double>> & bytes_pts) {

    overhead_pts.clear();
    rate_pts.clear();
    bytes_pts.clear();

    for (const auto & sw : data.sweeps) {
        if (g >= sw.dmds_low.size() || g >= sw.dmds_high.size()) {
            continue;
        }
        const auto & lo = sw.dmds_low[g];
        const auto & hi = sw.dmds_high[g];

        // model bytes on device g in this sweep (same for low/high)
        double m_g = static_cast<double>(lo.mb.model);
        // this sweep's split share for device g
        double x_g = static_cast<double>(sw.shares[g]);

        // KV rate: bytes per cell (from the delta between high and low probes).
        // N0=256, N1=512 are the probe pool sizes.
        const double N0 = 256.0, N1 = 512.0;
        double ctx_low  = static_cast<double>(lo.mb.context);
        double ctx_high = static_cast<double>(hi.mb.context);
        double rate     = (ctx_high - ctx_low) / (N1 - N0); // bytes per cell

        // overhead = compute + the cell-independent ("fixed") part of context.
        // The low-pool context measurement still includes N0 pool cells; subtract
        // them so the affine model fits only what the water-fill solver expects
        // (pool size P enters Phase C as an independent variable).
        double fixed    = ctx_low - rate * N0;
        double overhead = static_cast<double>(lo.mb.compute) + fixed;

        overhead_pts.push_back({m_g, overhead});
        rate_pts.push_back({m_g, rate});
        bytes_pts.push_back({x_g, m_g});
    }
}

// Phase C: water-filling solve for optimal tensor_split and predicted pool size.
// Returns (shares, predicted_pool_cells); shares.empty() means "no feasible solution".
std::pair<std::vector<double>, double> solve_split(
        const std::vector<DeviceCostModel> & models,
        const std::vector<uint64_t> & cap_bytes) { // max allocatable bytes per device (VRAM budget)

    const size_t nd = models.size();
    if (nd == 0) return {std::vector<double>(), 0.0};

    // Per-device usage model at pool size P and split share x_g (all from measurements):
    //   m_g(x_g)        = p_g*x_g + q_g                  (layer-index byte placement)
    //   overhead_g(m_g) = a_g + b_g*m_g
    //   rate_g(m_g)     = c_g + d_g*m_g
    //   U_g(P, x_g)     = model bytes + overhead + pool KV
    //                    = [a_g + (1+b_g)*q_g + P*(c_g + d_g*q_g)] + x_g*(1+b_g+P*d_g)*p_g
    //
    // The constraint U_g <= cap_g solves for the maximum share device g can take at pool P:
    //   ub_g(P) = (cap_g - A_g(P)) / B_g(P), where
    //     A_g(P) = a_g + (1+b_g)*q_g + P*(c_g + d_g*q_g)
    //     B_g(P) = (1 + b_g + P*d_g)*p_g      (the 1s = the device's own model bytes)
    //
    // A pool size P is feasible iff the device shares that exactly fill every budget still
    // sum to at least 1 (i.e. the whole model can be placed): Feasible(P) = sum ub_g(P) >= 1.
    auto ub = [&](size_t g, double P) -> double {
        const auto & m = models[g];
        // U_g(P,x) = oh(m(x)) + P*rt(m(x)) + m(x),  with m(x) = p*x + q:
        //   A_g(P) collects the x-independent parts: overhead intercept a,
        //           the fixed model-byte offset q (itself!), and pool KV terms.
        double A = m.a_overhead + m.q_bytes * (1.0 + m.b_overhead)
                 + P * (m.c_rate + m.d_rate * m.q_bytes);
        //   B_g multiplies the share x: device's own bytes (1) + overhead/
        //           rate sensitivity to its model bytes (b, d).
        double B = m.p_bytes * (1.0 + m.b_overhead + P * m.d_rate);
        if (B <= 0) return 0.0;
        return std::max(0.0, (static_cast<double>(cap_bytes[g]) - A) / B);
    };

    auto sum_ub_positive = [&](double P) -> double {
        double s = 0.0;
        for (size_t g = 0; g < nd; g++) {
            s += ub(g, P);
        }
        return s;
    };

    // Binary search for the largest feasible pool size. Monotone: as P grows each device's
    // affordable share shrinks, so sum(ub) crosses 1 exactly once.
    double lo_pool = 0, hi_pool = 4 * 1e6; // up to ~4M cells max
    if (sum_ub_positive(hi_pool) >= 1.0) {
        return {std::vector<double>(), 0.0}; // pathological: unbounded, let the caller verify catch it
    }

    for (int iter = 0; iter < 60; iter++) {
        double mid = (lo_pool + hi_pool) / 2.0;
        if (sum_ub_positive(mid) >= 1.0) {
            lo_pool = mid;
        } else {
            hi_pool = mid;
        }
    }

    double P_star = lo_pool;
    if (P_star < 256.0 / 2.0) {
        return {std::vector<double>(), 0.0}; // not even a minimum pool fits anywhere
    }

    // Compute shares at P_star: use each device exactly up to its budget (water-filling).
    double sum_shares = 0;
    std::vector<double> shares(nd, 0.0);
    for (size_t g = 0; g < nd; g++) {
        shares[g] = ub(g, P_star);
        if (shares[g] > 0) sum_shares += shares[g];
    }

    if (sum_shares <= 0) {
        return {std::vector<double>(), 0.0}; // no feasible split found
    }

    for (size_t g = 0; g < nd; g++) {
        shares[g] /= sum_shares;
    }

    return std::pair(std::move(shares), P_star);
}

} // anonymous namespace

// See llama-ext.h.
LLAMA_API bool llama_kv_fit_auto_split(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split) {

    llama_kv_fit_warmup_devices();

    // Step 1: Get initial device memory data to find out how many devices we have
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;
    std::vector<llama_device_memory_data> dmds_full;
    try {
        dmds_full = llama_kv_fit_probe(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex);
    } catch (const std::runtime_error & e) {
        LLAMA_LOG_ERROR("%s: probing failed: %s\n", __func__, e.what());
        return false;
    }

    const size_t nd = devs.size();
    if (nd == 0 || nd > llama_max_devices()) {
        LLAMA_LOG_WARN("%s: no GPU devices found or too many devices (%zu), aborting auto split\n", __func__, nd);
        return false;
    }

    // For N=1, there's no split to optimize. Return success with the single device getting everything.
    if (nd == 1) {
        tensor_split[0] = 1.0f;
        mparams->n_gpu_layers = static_cast<int32_t>(hp_ngl);
        LLAMA_LOG_INFO("%s: single GPU device, no split optimization needed\n", __func__);
        return true;
    }

    // Phase A: Run placement sweep with N+1 different splits.
    // For each candidate placement, probe at 256/512 cells to measure per-device costs.
    const double eps = 0.05; // minimum share for "concentrated on k" placements

    PlacementSweepData sweep_data;
    sweep_data.nd = nd;

    auto run_probe_pair = [&](float * split, const char * phase) -> std::pair<std::vector<llama_device_memory_data>, std::vector<llama_device_memory_data>> {
        // Set the split for this probe
        mparams->tensor_split = split;

        LLAMA_LOG_INFO("%s: %s - probing tensor_split [", __func__, phase);
        for (size_t i = 0; i < nd; i++) {
            fprintf(stderr, "%s", i > 0 ? ", " : "");
            fprintf(stderr, "%.4f", split[i]);
        }
        fprintf(stderr, "]\n");

        // Low pool probe (256 cells)
        llama_context_params cp_low  = *cparams;
        cp_low.n_kv_cache  = 256;
        std::vector<ggml_backend_dev_t> devs_lo;
        uint32_t ngl_lo = 0, nct_lo = 0, nex_lo = 0;
        std::vector<llama_device_memory_data> dmds_lo;
        try {
            dmds_lo = llama_kv_fit_probe(path_model, mparams, &cp_low, devs_lo, ngl_lo, nct_lo, nex_lo);
        } catch (const std::runtime_error & e) {
            LLAMA_LOG_WARN("%s: low pool probe failed: %s\n", __func__, e.what());
            return {{}, {}};
        }

        // High pool probe (512 cells)
        llama_context_params cp_hi  = *cparams;
        cp_hi.n_kv_cache  = 512;
        std::vector<ggml_backend_dev_t> devs_hi;
        uint32_t ngl_hi = 0, nct_hi = 0, nex_hi = 0;
        std::vector<llama_device_memory_data> dmds_hi;
        try {
            dmds_hi = llama_kv_fit_probe(path_model, mparams, &cp_hi, devs_hi, ngl_hi, nct_hi, nex_hi);
        } catch (const std::runtime_error & e) {
            LLAMA_LOG_WARN("%s: high pool probe failed: %s\n", __func__, e.what());
            return {{}, {}};
        }

        // Log the per-device measurements so the decisions are auditable.
        for (size_t i = 0; i < nd && i + 1 < dmds_lo.size(); i++) {
            const auto & lo = dmds_lo[i].mb;
            const auto & hi = dmds_hi[i].mb;
            double rate = ((double)hi.context - (double)lo.context) / 256.0;
            double fixed = (double)lo.context - rate * 256.0;
            LLAMA_LOG_INFO("%s: %s -   dev %zu: model=%.3f GiB compute+fixed=%.3f GiB rate=%.1f B/cell\n",
                    __func__, phase, i,
                    lo.model / 1073741824.0,
                    ((double)lo.compute + fixed) / 1073741824.0,
                    rate);
        }

        return {std::move(dmds_lo), std::move(dmds_hi)};
    };

    // Generate placements: for each k in 0..N-1, concentrate on device k.
    // Then one equal split.
    for (size_t k = 0; k < nd; k++) {
        float split[16] = {0};
        for (size_t g = 0; g < nd; g++) {
            if (g == k) {
                split[g] = static_cast<float>(1.0 - eps * (nd - 1));
            } else {
                split[g] = static_cast<float>(eps);
            }
        }

        LLAMA_LOG_DEBUG("%s: Phase A - probing concentrated placement on device %zu\n", __func__, k);

        auto [lo, hi] = run_probe_pair(split, "Phase A");
        if (!lo.empty() && !hi.empty()) {
            sweep_data.add_sweep(split, std::move(lo), std::move(hi));
        } else {
            LLAMA_LOG_WARN("%s: probe pair failed for concentrated placement on device %zu\n", __func__, k);
        }
    }

    // Add equal split.
    {
        float split[16] = {0};
        for (size_t g = 0; g < nd; g++) {
            split[g] = static_cast<float>(1.0 / nd);
        }

        LLAMA_LOG_DEBUG("%s: Phase A - probing equal split placement\n", __func__);

        auto [lo, hi] = run_probe_pair(split, "Phase A");
        if (!lo.empty() && !hi.empty()) {
            sweep_data.add_sweep(split, std::move(lo), std::move(hi));
        } else {
            LLAMA_LOG_WARN("%s: probe pair failed for equal split placement\n", __func__);
        }
    }

    if (sweep_data.sweeps.empty()) {
        LLAMA_LOG_ERROR("%s: no successful probes in Phase A, aborting auto split\n", __func__);
        return false;
    }

    // Phase B: Fit per-device cost models from sweep data.
    std::vector<DeviceCostModel> models(nd);

    for (size_t g = 0; g < nd; g++) {
        std::vector<std::pair<double, double>> overhead_pts;
        std::vector<std::pair<double, double>> rate_pts;
        std::vector<std::pair<double, double>> bytes_pts;

        collect_device_points(sweep_data, g, overhead_pts, rate_pts, bytes_pts);

        DeviceCostModel & model = models[g];
        model.valid = true;

        double res_overhead = 0, res_rate = 0, res_bytes = 0;

        if (!fit_line(overhead_pts, model.a_overhead, model.b_overhead, &res_overhead)) {
            model.valid = false;
        }
        if (model.valid && !fit_line(rate_pts, model.c_rate, model.d_rate, &res_rate)) {
            model.valid = false;
        }
        if (model.valid && !fit_line(bytes_pts, model.q_bytes, model.p_bytes, &res_bytes)) {
            model.valid = false;
        }

        // Log model quality so rocky2 verification is one grep: residuals reveal drift
        // on exotic models before it shows up in the Phase D verification.
        LLAMA_LOG_INFO("%s: Phase B - device %zu fit: overhead a=%.3f GiB b=%.6f rate c=%.1f B/cell d=%.6f bytes/share p=%.3f GiB q=%.3f GiB residuals o=%.2f%% r=%.2f%% b=%.2f%% valid=%d\n",
                __func__, g,
                model.a_overhead / 1073741824.0, model.b_overhead,
                model.c_rate, model.d_rate,
                model.p_bytes / 1073741824.0, model.q_bytes / 1073741824.0,
                100.0 * res_overhead, 100.0 * res_rate, 100.0 * res_bytes, (int)model.valid);
    }

    // Check if all devices have valid models. If not, fall back to VRAM-proportional split.
    bool all_valid = true;
    for (size_t g = 0; g < nd; g++) {
        if (!models[g].valid) {
            all_valid = false;
            break;
        }
    }

    if (!all_valid) {
        LLAMA_LOG_WARN("%s: Phase B models incomplete, falling back to proportional VRAM split\n", __func__);
        double sum_vram = 0.0;
        std::vector<double> vram_nd(nd);
        for (size_t id = 0; id < nd; id++) {
            size_t free, total;
            ggml_backend_dev_memory(devs[id], &free, &total);
            vram_nd[id] = static_cast<double>(total);
            sum_vram += vram_nd[id];
        }

        if (sum_vram <= 0) {
            LLAMA_LOG_WARN("%s: total VRAM is zero or negative, aborting auto split\n", __func__);
            return false;
        }

        for (size_t id = 0; id < nd; id++) {
            tensor_split[id] = static_cast<float>(vram_nd[id] / sum_vram);
        }

        LLAMA_LOG_INFO("%s: computing proportional VRAM tensor split (fallback):\n", __func__);
        for (size_t id = 0; id < nd; id++) {
            LLAMA_LOG_INFO("%s:   - device %zu: %6.1f MiB (%5.2f%%)\n",
                    __func__, id, vram_nd[id]/(1024*1024), 100.0*tensor_split[id]);
        }

        // For fallback, keep all layers on GPU; the split is the proportional one.
        mparams->n_gpu_layers = static_cast<int32_t>(hp_ngl);
        if (nd > 1) {
            mparams->tensor_split = tensor_split;
        }

        LLAMA_LOG_INFO("%s: fallback optimized n_gpu_layers=%u with tensor_split for %zu devices\n",
                __func__, hp_ngl, nd);
        return true;
    }

    // Phase C: Water-filling solve. Use a slightly more conservative VRAM fraction
    // (0.95 vs the standard 0.97) for extra safety margin while validating on rocky2.
    std::vector<uint64_t> cap_bytes(nd);
    for (size_t g = 0; g < nd; g++) {
        size_t free, total;
        ggml_backend_dev_memory(devs[g], &free, &total);
        uint64_t vram_cap   = static_cast<uint64_t>(0.95 * total);
        uint64_t free_cap   = free;
        cap_bytes[g] = std::min(vram_cap, free_cap);

        // Device 0 usually doubles as the display GPU (X server / Wayland live in its
        // VRAM). Reserve extra headroom there so it never ends up tighter than the
        // other devices: when the split cannot be even, the surplus goes to dev 0.
        if (g == 0) {
            const uint64_t display_headroom = 512u * 1024u * 1024u;
            if (cap_bytes[0] > display_headroom) {
                cap_bytes[0] -= display_headroom;
            }
        }
    }

    auto [solved_shares, predicted_pool] = solve_split(models, cap_bytes);

    if (!solved_shares.empty()) {
        for (size_t g = 0; g < nd; g++) {
            tensor_split[g] = static_cast<float>(solved_shares[g]);
        }
        LLAMA_LOG_INFO("%s: Phase C - solved tensor split:\n", __func__);
        for (size_t g = 0; g < nd; g++) {
            LLAMA_LOG_INFO("%s:   - device %zu: %.4f (%.1f%%)\n",
                    __func__, g, tensor_split[g], 100.0 * tensor_split[g]);
        }
        LLAMA_LOG_INFO("%s: predicted pool size = %.1f cells\n", __func__, predicted_pool);
    } else {
        // If solve failed, fall back to VRAM-proportional split.
        LLAMA_LOG_WARN("%s: Phase C solve failed, falling back to proportional VRAM split\n", __func__);
        double sum_vram = 0.0;
        std::vector<double> vram_nd(nd);
        for (size_t id = 0; id < nd; id++) {
            size_t free, total;
            ggml_backend_dev_memory(devs[id], &free, &total);
            vram_nd[id] = static_cast<double>(total);
            sum_vram += vram_nd[id];
        }

        if (sum_vram <= 0) {
            LLAMA_LOG_WARN("%s: total VRAM is zero or negative, aborting auto split\n", __func__);
            return false;
        }

        for (size_t id = 0; id < nd; id++) {
            tensor_split[id] = static_cast<float>(vram_nd[id] / sum_vram);
        }
    }

    // Set mparams to use all layers on GPU with the solved split.
    mparams->n_gpu_layers = static_cast<int32_t>(hp_ngl);
    if (nd > 1) {
        mparams->tensor_split = tensor_split;
    }

    LLAMA_LOG_INFO("%s: optimized n_gpu_layers=%u with tensor_split for %zu devices\n",
            __func__, hp_ngl, nd);

    // Phase D: verify the predicted pool against a real probe pair at the solved split.
    // Uses the same 256/512 probe + affine solve as llama_kv_fit_max_pool() so the
    // measured pool is what fllama will actually get; rocky2 verification greps one line.
    {
        const double vram_fraction = 0.95; // conservative margin (see Phase C)
        auto [lo_ver, hi_ver] = run_probe_pair(tensor_split, "Phase D");

        if (!lo_ver.empty() && !hi_ver.empty()) {
            // runtime (lazy) overhead is not passed here: it is measured later, inside
            // llama_kv_fit_max_pool(), which produces the binding pool decision. Phase D
            // stays slightly optimistic on purpose - it checks split consistency, not the
            // final budget.
            const uint32_t n_ver = llama_kv_fit_max_pool_solve(
                    lo_ver, hi_ver, 256, 512, nullptr, nullptr, 0, vram_fraction);
            LLAMA_LOG_INFO("%s: Phase D verification - solved split gives measured pool %u cells (predicted %.1f)\n",
                    __func__, n_ver, predicted_pool);
        } else {
            LLAMA_LOG_WARN("%s: Phase D verification probes failed; relying on prediction only\n", __func__);
        }
    }

    return true;
}
