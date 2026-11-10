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
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Force GPU backends to fully initialize (e.g. hipInit) before we query VRAM.
// llama_backend_init() only loads backend *registries*; ROCm/HIP does not report real
// device memory until a buffer is first allocated on that backend. That first allocation
// normally happens during model load — but the fit probes run BEFORE any load, so their
// ggml_backend_dev_memory() reads hit an uninitialized backend and come back 0/0. Allocate
// and free a tiny buffer on every GPU-like device to bring the runtime up; it stays up for
// the process lifetime, so this is effectively one-shot.
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
            continue; // only GPU-like backends need warming up for VRAM reporting
        }

        size_t total_before = 0, free_before = 0;
        ggml_backend_dev_memory(dev, &free_before, &total_before);

        const char * name  = ggml_backend_dev_name(dev);
        const size_t warm  = 4096; // small probe allocation to trigger backend runtime init
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        ggml_backend_buffer_t buf       = (buft != nullptr) ? ggml_backend_buft_alloc_buffer(buft, warm) : nullptr;
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }

        size_t total_after = 0, free_after = 0;
        ggml_backend_dev_memory(dev, &free_after, &total_after);

        fprintf(stderr, "[KV-FIT-DBG] warmup: dev %s (type=%d) alloc_rc=%d VRAM total %lld / free before %lld / after %lld\n",
                name, (int)type, buf != nullptr,
                (long long)total_before, (long long)free_before, (long long)free_after);
    }

    done = true;
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

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("failed to create llama_context from model");
    }

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

    return ret;
}

// Pure computation: solve the maximum unified KV pool size from two no_alloc
// probes taken at N0 and N1 cells. Has no device/model dependencies of its own;
// everything is measured into the dmds vectors by the caller.
static uint32_t llama_kv_fit_max_pool_solve(
        const std::vector<llama_device_memory_data> & dmds_low,
        const std::vector<llama_device_memory_data> & dmds_high,
                                uint32_t N0, uint32_t N1,
                                const size_t * margins,      // per-device margin to keep free (bytes)
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

        // compute (the scheduler's graph-workspace estimate) may legitimately grow with the
        // pool size: attention workspaces follow KV tensor length. It is a deterministic
        // function of the graph, never an external-allocation drift, so take the max instead
        // of demanding equality. Model bytes are a real allocation measurement and MUST match.
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

        // model weights + compute buffers + cell-independent state are consumed before the pool
        const int64_t non_pool   = (int64_t) lo.model + (int64_t) compute + fixed;

        uint32_t n_dev = 0;
        if (non_pool < pool_room) {
            const int64_t cells = (pool_room - non_pool) / rate;
            n_dev = (uint32_t) (cells / 256u * 256u);
        }

        fprintf(stderr, "[KV-FIT-DBG] solve: device %zu: total=%lld free=%lld used=%lld | cap_driver=%lld cap_free=%lld pool_room=%lld non_pool=%lld fixed=%lld rate=%lld B/cell -> max %9u cells\n",
                id,
                (long long)dmd.total,  (long long)dmd.free, (long long)used,
                (long long)cap_driver, (long long)cap_free, (long long)pool_room,
                (long long)non_pool,   (long long)fixed, (long long)rate, n_dev);

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

    auto probe = [&](uint32_t n_pool) -> std::vector<llama_device_memory_data> {
        llama_context_params cp_copy = *cparams;
        cp_copy.n_kv_cache = n_pool;

        std::vector<ggml_backend_dev_t> devs_main;
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

    const uint32_t n_max = llama_kv_fit_max_pool_solve(dmds_low, dmds_high, N0, N1, margins, vram_fraction);
    if (n_max > 0) {
        LLAMA_LOG_INFO("%s: maximum unified KV pool size is %u cells\n", __func__, n_max);
    }
    return (int32_t) n_max;
}

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

    // Step 2: Compute tensor_split proportional to VRAM
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

    LLAMA_LOG_INFO("%s: computing proportional VRAM tensor split:\n", __func__);
    for (size_t id = 0; id < nd; id++) {
        LLAMA_LOG_INFO("%s:   - device %zu: %6.1f MiB (%5.2f%%)\n",
                __func__, id, vram_nd[id]/(1024*1024), 100.0*tensor_split[id]);
    }

    // Step 3: Analytic calculation to find maximum n_gpu_layers that allows KV pool >= 256 cells

    // Model bytes per layer from metadata (sum across all devices to get total)
    size_t model_bytes_total = 0;
    for (size_t id = 0; id < nd; id++) {
        model_bytes_total += dmds_full[id].mb.model;
    }
    size_t model_bytes_per_layer = hp_ngl > 0 ? model_bytes_total / hp_ngl : 0;

    // Compute overhead: use a conservative fraction of model memory (typical for inference)
    const double COMPUTE_FRACTION = 0.10;  // 10% overhead estimate

    LLAMA_LOG_INFO("%s: analytic calculation with model_bytes_per_layer=%zu bytes, compute_fraction=%.2f\n",
            __func__, model_bytes_per_layer, COMPUTE_FRACTION);

    // Get the context data from our probe (sum across all devices to get total)
    size_t context_bytes_total = 0;
    for (size_t id = 0; id < nd; id++) {
        context_bytes_total += dmds_full[id].mb.context;
    }
    uint32_t current_n_ctx = cparams->n_ctx > 0 ? cparams->n_ctx : hp_nct;

    // Estimate KV rate per cell (total across all devices)
    size_t kv_rate_per_cell_total = (current_n_ctx > 0 && context_bytes_total > 0) ? context_bytes_total / current_n_ctx : 0;

    LLAMA_LOG_INFO("%s: estimated kv_rate_per_cell_total=%zu bytes\n", __func__, kv_rate_per_cell_total);

    // For device i, its share of KV rate is proportional to its tensor_split weight
    std::vector<size_t> kv_rate_per_device(nd);
    for (size_t id = 0; id < nd; id++) {
        kv_rate_per_device[id] = (kv_rate_per_cell_total * vram_nd[id]) / sum_vram;
    }

    // Now solve for maximum g (n_gpu_layers) such that all devices can support N >= 256 cells
    uint32_t max_ngl = hp_ngl;

    for (uint32_t g = max_ngl; g >= 1; g--) {
        bool fits = true;

        for (size_t id = 0; id < nd && fits; id++) {
            double layers_on_dev   = g * vram_nd[id] / sum_vram;  // proportional distribution
            size_t model_bytes_i   = (size_t)(model_bytes_per_layer * layers_on_dev);
            size_t compute_i       = (size_t)(COMPUTE_FRACTION * model_bytes_i);

            size_t free_mem, total_mem;
            ggml_backend_dev_memory(devs[id], &free_mem, &total_mem);
            size_t cap = std::min(free_mem, (size_t)(0.97 * total_mem));

            size_t pool_bytes_needed = kv_rate_per_device[id] * 256;
            size_t total_needed      = model_bytes_i + compute_i + pool_bytes_needed;

            LLAMA_LOG_DEBUG("%s:   - testing g=%u on device %zu: needed=%zu vs cap=%zu\n",
                    __func__, g, id, total_needed, cap);

            if (total_needed > cap) {
                fits = false;
            }
        }

        if (fits) {
            LLAMA_LOG_INFO("%s: found n_gpu_layers=%u that allows pool >= 256 cells\n", __func__, g);
            break;
        } else {
            max_ngl = g - 1;  // try fewer layers next
        }
    }

    if (max_ngl == 0) {
        LLAMA_LOG_WARN("%s: cannot fit any GPU layers with a usable KV pool, aborting auto split\n", __func__);
        return false;
    }

    // Step 4: Update mparams with the optimal settings
    mparams->n_gpu_layers = static_cast<int32_t>(max_ngl);
    if (nd > 1) {
        mparams->tensor_split = tensor_split;
    }

    LLAMA_LOG_INFO("%s: optimized n_gpu_layers=%u with tensor_split for %zu devices\n",
            __func__, max_ngl, nd);

    return true;
}
