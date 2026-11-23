// [FORK] KV-fit planner — end-to-end client of libllama's KV-split planning API.
//
// Pure C against the public interfaces, exactly what fllama will do at startup:
//   1. llama_kv_fit_params_new()  — planning options (library-allocated)
//   2. llama_kv_split()           — placement AND pool size in ONE call
//
// This tool is the deployment report itself: it sets LLAMA_KV_FIT_PRINT_TABLES
// so the library prints the model header + per-layer table before planning.
// With --verify the library additionally runs ONE verification load with the
// computed plan (model at the planned split + context at the planned pool +
// warmup decode) and proves it: per-layer placement readback against the plan,
// plus free VRAM after everything is loaded and warmed up. Without --verify
// there is no verification run — planning only.
//
// Planning traces flow through the standard log callback at the PLANNING level.

#include "llama.h"
#include "../../src/llama-ext-c.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * g_path = NULL;



static void log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level == GGML_LOG_LEVEL_PLANNING || level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static int kv_type_from_str(const char * s) {
    if (!strcmp(s, "f16"))  return GGML_TYPE_F16;
    if (!strcmp(s, "bf16")) return GGML_TYPE_BF16;
    if (!strcmp(s, "f32"))  return GGML_TYPE_F32;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q4_0")) return GGML_TYPE_Q4_0;
    fprintf(stderr, "error: unknown cache type '%s' (f16|bf16|f32|q8_0|q4_0)\n", s);
    exit(1);
}

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s MODEL [options]\n"
        "\n"
        "KV-fit planner: plans a deployment through libllama's KV-fit API and\n"
        "validates the result by loading with it (see the file header).\n"
        "\n"
        "options:\n"
        "  --keep-free F           reserve fraction of each GPU's VRAM   [0.03]\n"
        "  --parallel P            parallel streams (plans 9 SSM rows each)\n"
        "  --checkpoints N         SSM state rows, single stream          [1]\n"
        "  --flash-attn / --no-flash-attn\n"
        "  --cache-type-k T        KV cache type for K    [f16]\n"
        "  --cache-type-v T        KV cache type for V    [f16]\n"
        "  --ubatch-size N         physical ubatch size     [512]\n"
        "  --verify                run one verification load with the computed plan\n"
        "                          (split + pool + warmup; proves it fits at runtime)\n"
        "  --debug                 print planning internals (breakdown readbacks,\n"
        "                          rate measurements, decomposition formulas)\n",
        prog);
}

int main(int argc, char ** argv) {
    double keep_free = 0.03;
    int parallel_arg = -1, checkpoints_arg = -1;
    enum ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    bool flash_attn = true;
    bool verify = false;
    bool debug = false;
    uint32_t ubatch_size = 512;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        if (!strcmp(a, "--keep-free") || !strcmp(a, "--parallel") || !strcmp(a, "--checkpoints") ||
            !strcmp(a, "--cache-type-k") || !strcmp(a, "--cache-type-v") || !strcmp(a, "--ubatch-size")) {
            if (++i >= argc) { fprintf(stderr, "error: '%s' requires a value\n", a); return 1; }
        }
        if      (!strcmp(a, "--keep-free"))    keep_free   = atof(argv[i]);
        else if (!strcmp(a, "--parallel"))     parallel_arg    = atoi(argv[i]);
        else if (!strcmp(a, "--checkpoints"))  checkpoints_arg = atoi(argv[i]);
        else if (!strcmp(a, "--flash-attn"))   flash_attn      = true;
        else if (!strcmp(a, "--no-flash-attn"))flash_attn      = false;
        else if (!strcmp(a, "--cache-type-k")) type_k          = kv_type_from_str(argv[i]);
        else if (!strcmp(a, "--cache-type-v")) type_v          = kv_type_from_str(argv[i]);
        else if (!strcmp(a, "--ubatch-size"))  ubatch_size     = (uint32_t)atoi(argv[i]);
        else if (!strcmp(a, "--verify"))       verify          = true;
        else if (!strcmp(a, "--debug"))        debug           = true;
        else if (!g_path)                      g_path          = argv[i];
        else { fprintf(stderr, "error: unexpected argument '%s'\n", a); return 1; }
    }
    if (!g_path) { usage(argv[0]); return 1; }

    // deployment-shape constraints
    if ((parallel_arg >= 0 && parallel_arg < 1) || (checkpoints_arg >= 0 && checkpoints_arg < 1)) {
        fprintf(stderr, "error: --parallel / --checkpoints must be >= 1\n");
        return 1;
    }
    if (parallel_arg >= 0 && checkpoints_arg >= 0) {
        fprintf(stderr, "error: --parallel and --checkpoints cannot be combined\n");
        return 1;
    }
    if (!(keep_free >= 0.0 && keep_free < 1.0)) {
        fprintf(stderr, "error: --keep-free must be in [0, 1)\n");
        return 1;
    }

    llama_log_set(log_callback, NULL);
    llama_backend_init();

    // the deployment's params, as fllama would supply them
    struct llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true;   // inert without a nextn block; the plan decides MTP measurement

    struct llama_context_params cparams = llama_context_default_params();
    cparams.kv_unified = true;  // the deployment model
    cparams.n_ctx      = 0;     // natural context (library default) — the pool extends the cache, never the context
    cparams.n_batch    = 2048;
    cparams.n_ubatch   = ubatch_size;
    cparams.type_k     = type_k;
    cparams.type_v     = type_v;
    cparams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (parallel_arg >= 0) {          // P streams x (8 rollback + 1 live) SSM rows
        cparams.n_seq_max = (uint32_t)parallel_arg;
        cparams.n_rs_seq  = 8;
    } else {                           // 1 stream, checkpoints-1 rollback rows
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = (checkpoints_arg >= 0) ? (uint32_t)checkpoints_arg - 1 : 0;
    }

    // planning options — allocated by the library (stale-header safe)
    struct llama_kv_fit_params * params = llama_kv_fit_params_new(LLAMA_KV_FIT_PARAMS_VERSION);
    params->vram_fraction = 1.0 - keep_free;
    // the app knows its own checkpoint policy: SSM rows per recurrent layer it
    // will allocate (= n_seq_max x (live + rollback rows)). Pass it explicitly —
    // the planner uses it verbatim rather than re-deriving from context params.
    params->n_checkpoints = cparams.n_seq_max * (1u + cparams.n_rs_seq);
    // this tool is the deployment report: PRINT_TABLES prints the model header
    // + per-layer table before planning; verify/debug stay caller-controlled.
    params->options = LLAMA_KV_FIT_PRINT_TABLES
                    | (verify ? LLAMA_KV_FIT_VERIFY : 0u)
                    | (debug  ? LLAMA_KV_FIT_DEBUG  : 0u);

    // ---- plan placement AND pool in one call ---------------------------------
    fprintf(stderr, "\n== kv-fit clone: planning %s ==\n", g_path);

    const struct llama_kv_fit_result * result =
        llama_kv_split(g_path, &mparams, &cparams, params, 0 /* maximize the pool */);

    if (!result || !result->ok) {
        fprintf(stderr, "error: no viable deployment — %s\n",
                result ? result->error_msg : "(no result)");
        llama_kv_fit_params_free(params);
        llama_backend_free();
        return 1;
    }
    if (result->n_cells < (int32_t)params->min_pool_cells) {
        fprintf(stderr, "error: planned pool %d cells is below the viability floor (%u) — aborting\n",
                result->n_cells, params->min_pool_cells);
        llama_kv_fit_params_free(params);
        llama_backend_free();
        return 1;
    }

    // Verification (load + context + warmup) lives in libllama: it ran inside
    // llama_kv_split when --verify was given (params->options).
    llama_kv_fit_params_free(params);
    llama_backend_free();

    fprintf(stderr, "\nkv-fit clone: planning complete — pool %d cells%s\n",
            result->n_cells, verify ? ", verification run passed" : "");
    return 0;
}
