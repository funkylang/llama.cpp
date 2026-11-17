// kv-mask-diag: diagnose whether the 2x compute-buffer charge on MTP devices is real
//
// Usage: llama-kv-mask-diag <model.gguf> [--n-ctx N] [--ubatch-size N]
//
// For each pool size (X, 2X): create main + MTP contexts, read per-device
// compute-buffer bytes from llama_get_memory_breakdown(). The rate of change
// reveals the true per-cell cost vs. what the solver charges.

#include "llama.h"
#include "llama-ext.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_compute(const char * label, const llama_context * ctx) {
    const llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
    std::printf("%-24s", label);
    for (const auto & [buft, data] : mb) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) continue;
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        std::printf(" %s:%" PRIu64, ggml_backend_dev_name(dev), data.compute);
    }
    std::printf("\n");
}

static uint64_t read_compute(const llama_context * ctx, size_t dev_idx) {
    const llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
    uint64_t total = 0;
    for (const auto & [buft, data] : mb) {
        ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
        if (!buft_dev) continue;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            if (ggml_backend_dev_get(i) == buft_dev && i == dev_idx) {
                total += data.compute;
                break;
            }
        }
    }
    return total;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--n-ctx N] [--ubatch-size N]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int n_ctx    = 16384;   // measurement pool (matches kvfit default)
    int n_ubatch = 512;

    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--n-ctx")       && i + 1 < argc) n_ctx    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ubatch-size") && i + 1 < argc) n_ubatch = atoi(argv[++i]);
    }

    std::printf("=== KV Mask Diagnostic ===\n");
    std::printf("model: %s\n", model_path);
    std::printf("lo pool: %d, hi pool: %d\n\n", n_ctx, 2 * n_ctx);

    // Load model with MTP (like fllama)
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.tensor_split = nullptr;   // native auto-split by free VRAM
    mp.load_mtp     = true;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "failed to load model\n"); return 1; }

    const int n_layer_all = llama_model_n_layer(model) + llama_model_n_layer_nextn(model);
    const int n_nextn     = llama_model_n_layer_nextn(model);
    std::printf("layers: total=%d nextn=%d trunk=%d\n\n",
                n_layer_all, n_nextn, n_layer_all - n_nextn);

    size_t ndevs = ggml_backend_dev_count();

    // ---- lo pool ----
    llama_context_params cp_lo = llama_context_default_params();
    cp_lo.n_ctx       = (uint32_t)n_ctx;
    cp_lo.kv_unified  = true;
    cp_lo.n_kv_cache  = 0;               // unified pool = PAD(n_ctx) cells
    cp_lo.n_ubatch    = n_ubatch;

    llama_context * ctx_main_lo = llama_init_from_model(model, cp_lo);
    if (!ctx_main_lo) { std::fprintf(stderr, "main ctx lo init failed\n"); return 1; }

    llama_context_params cpm_lo = cp_lo;
    cpm_lo.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cpm_lo.n_rs_seq  = 0;
    cpm_lo.ctx_other = ctx_main_lo;
    llama_context * ctx_mtp_lo = llama_init_from_model(model, cpm_lo);
    if (!ctx_mtp_lo) { std::fprintf(stderr, "MTP ctx lo init failed\n"); return 1; }

    std::printf("=== lo pool (%d cells) ===\n", n_ctx);
    print_compute("main compute:", ctx_main_lo);
    print_compute("mtp  compute:", ctx_mtp_lo);

    uint64_t main_lo[32] = {0}, mtp_lo[32] = {0};
    for (size_t i = 0; i < ndevs && i < 32; ++i) {
        main_lo[i] = read_compute(ctx_main_lo, i);
        mtp_lo[i]  = read_compute(ctx_mtp_lo,  i);
    }

    llama_free(ctx_mtp_lo);
    llama_free(ctx_main_lo);

    // ---- hi pool ----
    llama_context_params cp_hi = cp_lo;
    cp_hi.n_ctx = (uint32_t)(2 * n_ctx);

    llama_context * ctx_main_hi = llama_init_from_model(model, cp_hi);
    if (!ctx_main_hi) { std::fprintf(stderr, "main ctx hi init failed\n"); return 1; }

    llama_context_params cpm_hi = cp_hi;
    cpm_hi.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cpm_hi.n_rs_seq  = 0;
    cpm_hi.ctx_other = ctx_main_hi;
    llama_context * ctx_mtp_hi = llama_init_from_model(model, cpm_hi);
    if (!ctx_mtp_hi) { std::fprintf(stderr, "MTP ctx hi init failed\n"); return 1; }

    std::printf("=== hi pool (%d cells) ===\n", 2 * n_ctx);
    print_compute("main compute:", ctx_main_hi);
    print_compute("mtp  compute:", ctx_mtp_hi);

    uint64_t main_hi[32] = {0}, mtp_hi[32] = {0};
    for (size_t i = 0; i < ndevs && i < 32; ++i) {
        main_hi[i] = read_compute(ctx_main_hi, i);
        mtp_hi[i]  = read_compute(ctx_mtp_hi,  i);
    }

    llama_free(ctx_mtp_hi);
    llama_free(ctx_main_hi);
    llama_model_free(model);

    // ---- rates ----
    std::printf("\n=== Per-device compute-buffer rate (B/cell) ===\n");
    for (size_t i = 0; i < ndevs && i < 32; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;

        const uint64_t main_rate = (main_hi[i] > main_lo[i]) ?
            (main_hi[i] - main_lo[i]) / (uint64_t)n_ctx : 0;
        const uint64_t mtp_rate  = (mtp_hi[i]  > mtp_lo[i])  ?
            (mtp_hi[i]  - mtp_lo[i])  / (uint64_t)n_ctx : 0;

        std::printf("%s:\n", ggml_backend_dev_name(dev));
        std::printf("    lo:  main=%" PRIu64 "  mtp=%" PRIu64 "\n",
                    main_lo[i], mtp_lo[i]);
        std::printf("    hi:  main=%" PRIu64 "  mtp=%" PRIu64 "\n",
                    main_hi[i], mtp_hi[i]);
        std::printf("    rate(main)=%" PRIu64 " B/cell  rate(mtp)=%" PRIu64
                    " B/cell  combined=%" PRIu64 " B/cell\n",
                    main_rate, mtp_rate, main_rate + mtp_rate);
    }

    std::printf("\nCompare with the solver's 'buffers' bpc charge:\n");
    std::printf("If rate(main) ~= rate(mtp) and both are large -> 2x on MTP devices is REAL\n");
    std::printf("If rate(mtp) ~= 0 or << rate(main)          -> solver OVERCHARGES (double-count)\n");

    return 0;
}
