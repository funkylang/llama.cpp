// [FORK] gpu-split clone — end-to-end test of libllama's KV-fit planning API.
//
// Pure C against the public interfaces, exactly what fllama will do at startup:
//   1. llama_kv_fit_params_new()    — planning options (library-allocated)
//   2. llama_kv_fit_auto_split()    — planned placement (tensor_layer_bounds)
//   3. llama_kv_fit_max_pool()      — pool size (reuses step 2's measurement)
//   4. real model load with the planned params; per-layer placement verified
//      against the plan (proves the loader applies the boundaries verbatim)
//   5. unified context at the planned pool — proves the plan fits on VRAM;
//      plus the MTP draft context when the model has a nextn block (fllama shape)
//   6. Shakespeare warmup decode on that context — forces every lazy
//      allocation a decode pays (init alone can pass while decoding OOMs),
//      including one single-token MTP draft decode when measured in planning
//
// Planning traces flow through the standard log callback at the PLANNING level.
// Run next to tools/gpu-split with the same options on the same machine: the
// planning numbers must match (that tool is the verified reference).

#include "llama.h"
#include "../../src/llama-ext.h"
#include "../../src/warmup-text.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * g_path = NULL;

// Device memory snapshot at one stage of the test — raw bytes, nothing
// subtracted. Printed before the context (weights only) AND after it, so the
// planned pool's actual landing in VRAM is visible instead of assumed.
static void print_free_vram(const struct llama_model * model, int ndev, const char * header) {
    fprintf(stderr, "\n%s:\n", header);
    for (int i = 0; i < ndev; ++i) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        fprintf(stderr, "  %d    %-12s total=%14zu free=%14zu\n", i,
                ggml_backend_dev_name(dev), total_mem, free_mem);
    }
}

// Re-run the planning excerpt on the production context: every lazy
// allocation a decode pays (kernels materializing, graph re-reservation)
// happens here. Context init alone proves nothing — fllama's OOM appeared
// only at the first decode. Batch construction mirrors fllama.c / gpu-split:
// raw mallocs, one shared seq_id pointer, manual frees.
// Returns the decoded token count, or -1 on a failed decode (e.g. VRAM).
static int warmup_decode(struct llama_context * ctx, const llama_token * tokens, int n_tok) {
    if (n_tok <= 0) return 0;

    int seq_id = 0;   // non-const: batch.seq_id pointers must stay valid during llama_decode

    // prefill: n_batch-sized chunks; positions contiguous across chunks
    const int32_t n_batch = llama_n_batch(ctx);
    for (int32_t start = 0; start < n_tok; start += n_batch) {
        const int32_t end   = start + n_batch < n_tok ? start + n_batch : n_tok;
        const int32_t n     = end - start;
        llama_batch batch;
        batch.n_tokens = n;
        batch.token    = malloc(n * sizeof(llama_token));
        batch.embd     = NULL;
        batch.pos      = malloc(n * sizeof(llama_pos));
        batch.n_seq_id = malloc(n * sizeof(int32_t));
        batch.seq_id   = malloc(n * sizeof(llama_seq_id *));
        batch.logits   = malloc(n);
        for (int32_t i = 0; i < n; ++i) {
            batch.token[i]    = tokens[start + i];
            batch.pos[i]      = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = &seq_id;
            // output only at the final chunk's last position
            batch.logits[i]   = (end == n_tok && i == n - 1) ? 1 : 0;
        }

        const int rc = llama_decode(ctx, batch);
        free(batch.token); free(batch.pos); free(batch.n_seq_id); free(batch.seq_id); free(batch.logits);
        if (rc != 0) {
            fprintf(stderr, "  warmup: prefill decode failed at token %d\n", start);
            return -1;
        }
    }

    // a few generate steps so the single-token decode path is exercised too
    for (int g = 0; g < 4; ++g) {
        llama_batch batch;
        batch.n_tokens = 1;
        batch.token    = malloc(sizeof(llama_token));
        batch.embd     = NULL;
        batch.pos      = malloc(sizeof(llama_pos));
        batch.n_seq_id = malloc(sizeof(int32_t));
        batch.seq_id   = malloc(sizeof(llama_seq_id *));
        batch.logits   = malloc(1);
        batch.token[0]    = tokens[n_tok - 1];
        batch.pos[0]      = n_tok + g;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0]   = &seq_id;
        batch.logits[0]   = 1;

        const int rc = llama_decode(ctx, batch);
        free(batch.token); free(batch.pos); free(batch.n_seq_id); free(batch.seq_id); free(batch.logits);
        if (rc != 0) {
            fprintf(stderr, "  warmup: generate decode %d failed\n", g);
            return -1;
        }
    }

    return n_tok;
}

// planning traces + warnings + errors are shown; everything else stays quiet
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
        "gpu-split clone: plans a deployment through libllama's KV-fit API and\n"
        "validates the result by loading with it (see the file header).\n"
        "\n"
        "options (mirroring tools/gpu-split):\n"
        "  --keep-free F           reserve fraction of each GPU's VRAM   [0.03]\n"
        "  --parallel P            parallel streams (plans 9 SSM rows each)\n"
        "  --checkpoints N         SSM state rows, single stream          [1]\n"
        "  --flash-attn / --no-flash-attn\n"
        "  --cache-type-k T        KV cache type for K    [f16]\n"
        "  --cache-type-v T        KV cache type for V    [f16]\n"
        "  --ubatch-size N         physical ubatch size     [512]\n",
        prog);
}

int main(int argc, char ** argv) {
    double keep_free = 0.03;
    int parallel_arg = -1, checkpoints_arg = -1;
    enum ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    bool flash_attn = true;
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
        else if (!g_path)                      g_path          = argv[i];
        else { fprintf(stderr, "error: unexpected argument '%s'\n", a); return 1; }
    }
    if (!g_path) { usage(argv[0]); return 1; }

    // same deployment-shape rules as tools/gpu-split
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

    int32_t layer_bounds[16];   // llama_max_devices(); auto_split fills the used entries
    memset(layer_bounds, 0, sizeof(layer_bounds));

    // ---- 1+2: plan the placement ---------------------------------------------
    fprintf(stderr, "\n== kv-fit clone: planning %s ==\n", g_path);
    if (!llama_kv_fit_auto_split(g_path, &mparams, &cparams, layer_bounds, params, 0)) {
        fprintf(stderr, "error: no viable plan — aborting (see reports above)\n");
        return 1;
    }

    // ---- 3: pool size (reuses the measurement — one load, two answers) -------
    const int32_t n_cells = llama_kv_fit_max_pool(g_path, &mparams, &cparams, params);
    if (n_cells <= 0) {
        fprintf(stderr, "error: no KV pool fits this plan — aborting\n");
        return 1;
    }
    if (n_cells < (int32_t)params->min_pool_cells) {
        fprintf(stderr, "error: planned pool %d cells is below the viability floor (%u) — aborting\n",
                n_cells, params->min_pool_cells);
        return 1;
    }

    // ---- 4: load with the plan and verify the placement -----------------------
    struct llama_model * model = llama_model_load_from_file(g_path, mparams);
    if (!model) {
        fprintf(stderr, "error: planned load failed\n");
        return 1;
    }

    const uint32_t n_layer_all = llama_model_n_layer_all(model);
    const int32_t ndev = (int32_t)llama_model_n_devices(model);
    fprintf(stderr, "\nplacement verification: %u layers on %d devices, bounds=[", n_layer_all, ndev);
    for (int g = 0; g < ndev; ++g) {
        fprintf(stderr, "%s%d", g ? "," : "", layer_bounds[g]);
    }
    fprintf(stderr, "]\n");

    // expected device: the last whose boundary does not exceed il (loader rule);
    // output pseudo-layer lands on the last device. dev == -1 = CPU = a failure.
    for (uint32_t il = 0; il <= n_layer_all; ++il) {
        const int dev = llama_model_layer_device_index(model, il);
        int exp = 0;
        for (int g = ndev - 1; g >= 0; --g) {
            if ((int32_t)il >= layer_bounds[g]) { exp = g; break; }
        }
        if (dev != exp) {
            char what[32];
            if (il == n_layer_all) strcpy(what, "output");
            else snprintf(what, sizeof what, "layer %u", il);
            fprintf(stderr, "  MISMATCH at %s: loaded on device %d, plan says %d\n",
                    what, dev, exp);
            return 1;   // the loader ignored the planned boundaries
        }
    }
    fprintf(stderr, "  all layers on their planned devices\n");

    // ---- 4b: where the pool will come from --------------------------------------
    print_free_vram(model, ndev, "free VRAM after planned load (weights only)");

    // ---- 5: the real proof — context at the planned pool -----------------------
    fprintf(stderr, "\n== context init at n_kv_cache = %d cells ==\n", n_cells);
    cparams.n_kv_cache = (uint32_t)n_cells;
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: context init at the planned pool failed — plan over-committed VRAM\n");
        return 1;
    }

    // The MTP draft context when the deployment has one (fllama + planner shape):
    // same model weights, own nextn KV pool at the full planned size.
    struct llama_context * ctx_mtp = NULL;
    if (llama_model_n_layer_nextn(model) > 0) {
        fprintf(stderr, "\n== MTP draft context init at n_kv_cache = %d cells ==\n", n_cells);
        struct llama_context_params cp_mtp = cparams;
        cp_mtp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        cp_mtp.n_rs_seq  = 0;
        cp_mtp.ctx_other = ctx;
        ctx_mtp = llama_init_from_model(model, cp_mtp);
        if (!ctx_mtp) {
            fprintf(stderr, "error: MTP context init at the planned pool failed — plan over-committed VRAM\n");
            return 1;
        }
    }

    // headroom AFTER the full deployment exists: pool + compute buffers in place
    print_free_vram(model, ndev, "free VRAM after context init at the planned pool");

    // ---- 6: run Shakespeare — force every lazy allocation a decode pays -------
    // exactly what planning measured: main prefill + generate steps, and (with
    // MTP) one single-token draft decode to upload every kernel an MTP run launches
    fprintf(stderr, "\n== warmup decode (same Hamlet excerpt as planning) ==\n");
    {
        int32_t * p_tokens = malloc((strlen(WARMUP_TEXT) + 8) * sizeof(int32_t));
        const int n_tok = llama_tokenize(
            llama_model_get_vocab(model), WARMUP_TEXT, strlen(WARMUP_TEXT),
            p_tokens, (int)(strlen(WARMUP_TEXT) + 8), true, false);
        if (n_tok <= 0) {
            fprintf(stderr, "error: warmup tokenization failed\n");
            return 1;
        }
        const int rc = warmup_decode(ctx, p_tokens, n_tok);
        if (rc < 0) {
            free(p_tokens);
            fprintf(stderr, "error: warmup decode at the planned pool failed — lazy allocations over-committed VRAM\n");
            return 1;
        }
        if (ctx_mtp) {
            // The nextn graph takes hidden-state embeddings as input: fllama
            // feeds it real h_nextn rows; a zero row of the right width
            // forces the same lazy allocations. pos 0 — the MTP KV here is
            // its own (empty), not the mirrored production layout.
            const int n_embd = llama_model_n_embd(model);
            float * state = calloc((size_t)n_embd, sizeof(float));

            struct llama_batch_ext * mb = llama_batch_ext_init(ctx_mtp);
            const int32_t idx = llama_batch_ext_add_token(mb, 0, p_tokens[n_tok - 1]);
            const llama_pos pos = 0;
            llama_batch_ext_set_pos(mb, idx, &pos);
            struct llama_embd embd_state = { state, 1, (size_t)n_embd };
            llama_batch_ext_set_embd_state(mb, idx, embd_state);
            llama_batch_ext_set_output_logits(mb, idx, true);

            const int rc_mtp = llama_process(ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, mb);
            llama_batch_ext_free(mb);
            free(state);
            if (rc_mtp != 0 && rc_mtp != 1) {
                free(p_tokens);
                fprintf(stderr, "error: MTP draft decode failed — lazy allocations over-committed VRAM\n");
                return 1;
            }
        }
        free(p_tokens);
        print_free_vram(model, ndev, "free VRAM after warmup decode");
    }

    llama_free(ctx_mtp);
    llama_free(ctx);
    llama_model_free(model);
    llama_kv_fit_params_free(params);
    llama_backend_free();

    fprintf(stderr, "\nkv-fit clone: PASS — plan fits, placement verified, decode ran, pool %d cells\n", n_cells);
    return 0;
}
