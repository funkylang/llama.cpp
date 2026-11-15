// [FORK] kv-fit-deviation — measure how REAL memory usage deviates from our
// COMPUTED values, as a function of cache size and layer distribution.
//
// Controlled experiment (research mode, not the production planner): 4 loads,
// each fully isolated (everything unloaded before each run) so every run starts
// from a clean VRAM baseline:
//
//   Run 1: split 0..37 / 38..64  pool 16384   "provisional" (the native auto
//   Run 2: split 0..37 / 38..64  pool 81920    split's result, pinned so both
//                                                provisional runs share EXACTLY
//                                                the same placement)
//   Run 3: split 0..42 / 43..64  pool 16384
//   Run 4: split 0..42 / 43..64  pool 81920
//
// Both "provisional" runs hardcode the SAME boundary on purpose — the native
// auto-split depends on free VRAM at load time and would drift between calls;
// pinning the boundary removes that variable so only the pool size changes.
// (The native auto-split path is untouched and remains for production use.)
//
// Each run reports per-device USED memory (baseline_free - free) at every
// allocation stage: weights / +KV pool / +MTP KV / +compute. Compare those to
// the solver's computed terms (tensors, cache, buffers) to see where and how
// much they deviate as cache size and distribution vary.

#include "llama.h"
#include "../../src/warmup-text.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ND_MAX 16   // max devices we plan for (bounds array size)

static const char * g_path = NULL;

// ---- one controlled scenario ------------------------------------------------
struct run_cfg {
    const char * label;                        // short human description
    int32_t      bounds[ND_MAX];               // explicit placement: first layer on dev g
    int          n_bounds;                     // number of valid boundaries (= device count)
    uint32_t     pool_cells;                   // n_kv_cache for this run
};

// Raw per-backend-device free/total snapshot at one stage. No model needed.
static void read_all_devices(size_t * free_out, size_t * total_out) {
    const int ndev = (int)ggml_backend_dev_count();
    for (int i = 0; i < ndev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) { free_out[i] = 0; total_out[i] = 0; continue; }
        ggml_backend_dev_memory(dev, &free_out[i], &total_out[i]);
    }
}

// Print one stage: per-device used = baseline_free - current_free (VRAM this
// stage consumed over the clean state), so each row is comparable. Takes the
// already-sampled free arrays (no extra backend query here).
static void print_used(const char * header, int ndev,
                       const size_t * base_free, const size_t * now_free) {
    fprintf(stderr, "\n  %s:\n", header);
    for (int i = 0; i < ndev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        fprintf(stderr, "    %2d  %-14s used=%14zu  free=%14zu\n",
                i, ggml_backend_dev_name(dev), base_free[i] - now_free[i], now_free[i]);
    }
}

// Warmup decode on a context: prefill the Hamlet excerpt in n_batch chunks,
// then a few generate steps — forces every lazy allocation a real decode pays
// (kernels materializing, graph re-reservation). Mirrors kv-fit-test.c.
static int warmup_decode(struct llama_context * ctx, const llama_token * tok, int n_tok) {
    if (n_tok <= 0) return 0;
    int seq_id = 0;
    const int32_t n_batch = llama_n_batch(ctx);
    for (int32_t s = 0; s < n_tok; s += n_batch) {
        const int32_t e = s + n_batch < n_tok ? s + n_batch : n_tok;
        const int32_t n = e - s;
        llama_batch b;
        b.n_tokens = n;
        b.token    = malloc(n * sizeof(llama_token));
        b.embd     = NULL;
        b.pos      = malloc(n * sizeof(llama_pos));
        b.n_seq_id = malloc(n * sizeof(int32_t));
        b.seq_id   = malloc(n * sizeof(llama_seq_id *));
        b.logits   = malloc(n);
        for (int32_t i = 0; i < n; ++i) {
            b.token[i]    = tok[s + i];
            b.pos[i]      = s + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i]   = &seq_id;
            b.logits[i]   = (e == n_tok && i == n - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx, b);
        free(b.token); free(b.pos); free(b.n_seq_id); free(b.seq_id); free(b.logits);
        if (rc != 0) { fprintf(stderr, "    warmup: prefill decode failed at %d\n", s); return -1; }
    }
    for (int g = 0; g < 4; ++g) {
        llama_batch b;
        b.n_tokens = 1;
        b.token    = malloc(sizeof(llama_token));
        b.embd     = NULL;
        b.pos      = malloc(sizeof(llama_pos));
        b.n_seq_id = malloc(sizeof(int32_t));
        b.seq_id   = malloc(sizeof(llama_seq_id *));
        b.logits   = malloc(1);
        b.token[0] = tok[n_tok - 1];
        b.pos[0]   = n_tok + g;
        b.n_seq_id[0] = 1;
        b.seq_id[0]   = &seq_id;
        b.logits[0]   = 1;
        const int rc = llama_decode(ctx, b);
        free(b.token); free(b.pos); free(b.n_seq_id); free(b.seq_id); free(b.logits);
        if (rc != 0) { fprintf(stderr, "    warmup: generate decode %d failed\n", g); return -1; }
    }
    return n_tok;
}

// One isolated scenario: clean baseline -> load (explicit split) -> main ctx at
// pool -> MTP ctx (when nextn) -> warmup decode -> per-stage used-memory table.
// Frees everything before returning. Returns 0 ok, nonzero on a hard failure
// (the log shows which stage).
static int run_single_test(const char * path,
                           const struct llama_model_params * mp_base,
                           const struct llama_context_params * cp_base,
                           const struct run_cfg * cfg) {
    fprintf(stderr, "\n\n================ RUN %s  (pool=%u cells) ================\n",
            cfg->label, cfg->pool_cells);

    // baseline: nothing loaded
    size_t bf[ND_MAX], tot[ND_MAX];
    read_all_devices(bf, tot);
    for (int i = 0; i < (int)ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        fprintf(stderr, "  %2d  %-14s total=%14zu  free(baseline)=%14zu\n",
                i, ggml_backend_dev_name(dev), tot[i], bf[i]);
    }

    // model load with the pinned split (bounds[g] = first layer on dev g)
    struct llama_model_params mp = *mp_base;
    if ((int)ggml_backend_dev_count() != cfg->n_bounds) {
        fprintf(stderr, "    WARN: %d backend devices but run specifies %d boundaries\n",
                (int)ggml_backend_dev_count(), cfg->n_bounds);
    }
    mp.tensor_layer_bounds = cfg->bounds;

    struct llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "    LOAD FAILED\n"); return 1; }

    size_t f[ND_MAX];
    read_all_devices(f, tot);
    print_used("after load (weights only)", (int)ggml_backend_dev_count(), bf, f);

    // main context at the run's pool size
    struct llama_context_params cp = *cp_base;
    cp.n_kv_cache = cfg->pool_cells;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "    MAIN CTX INIT FAILED (pool too big)\n"); llama_model_free(model); return 1; }

    read_all_devices(f, tot);
    print_used("after main ctx (+KV pool +SSM state)", (int)ggml_backend_dev_count(), bf, f);

    // MTP draft context when the model has a nextn block (fllama shape)
    struct llama_context * ctx_mtp = NULL;
    if (llama_model_n_layer_nextn(model) > 0) {
        struct llama_context_params cm = cp;
        cm.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        cm.n_rs_seq  = 0;
        cm.ctx_other = ctx;
        ctx_mtp = llama_init_from_model(model, cm);
        if (!ctx_mtp) fprintf(stderr, "    MTP CTX INIT FAILED\n");
        else {
            read_all_devices(f, tot);
            print_used("after MTP ctx (+nextn KV pool)", (int)ggml_backend_dev_count(), bf, f);
        }
    }

    // warmup decode: main + one MTP draft token (uploads every lazy kernel)
    int32_t * p_tok = malloc((strlen(WARMUP_TEXT) + 8) * sizeof(int32_t));
    const int n_tok = llama_tokenize(llama_model_get_vocab(model), WARMUP_TEXT, strlen(WARMUP_TEXT),
                                     p_tok, (int)(strlen(WARMUP_TEXT) + 8), true, false);
    if (n_tok > 0) {
        if (warmup_decode(ctx, p_tok, n_tok) < 0) {
            fprintf(stderr, "    MAIN WARMUP DECODE FAILED\n");
        } else if (ctx_mtp) {
            const int n_embd = llama_model_n_embd(model);
            float * st = calloc((size_t)n_embd, sizeof(float));
            struct llama_batch_ext * mb = llama_batch_ext_init(ctx_mtp);
            const int32_t idx = llama_batch_ext_add_token(mb, 0, p_tok[n_tok - 1]);
            const llama_pos pos = 0;
            llama_batch_ext_set_pos(mb, idx, &pos);
            struct llama_embd es = { st, 1, (size_t)n_embd };
            llama_batch_ext_set_embd_state(mb, idx, es);
            llama_batch_ext_set_output_logits(mb, idx, true);
            const int rc = llama_process(ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, mb);
            if (rc != 0 && rc != 1) fprintf(stderr, "    MTP WARMUP DECODE FAILED\n");
            llama_batch_ext_free(mb);
            free(st);
        }
        read_all_devices(f, tot);
        print_used("after warmup decode (+compute buffers) = DEPLOYMENT", (int)ggml_backend_dev_count(), bf, f);
    }
    free(p_tok);

    llama_free(ctx_mtp);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    // deployment shape — mirror the production params so the measurement is
    // realistic; these apply identically to all 4 runs.
    enum ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    bool flash_attn = true;
    uint32_t ubatch_size = 512;
    int parallel_arg = -1, checkpoints_arg = -1;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            fprintf(stderr, "usage: %s MODEL [options]\n\n"
                "Deviation analysis: 4 controlled loads across cache size x split.\n"
                "\n"
                "options (mirror tools/gpu-split):\n"
                "  --parallel P            parallel streams (plans 9 SSM rows each)\n"
                "  --checkpoints N         SSM state rows, single stream           [1]\n"
                "  --flash-attn / --no-flash-attn\n"
                "  --cache-type-k T        KV cache type for K    [f16]\n"
                "  --cache-type-v T        KV cache type for V    [f16]\n"
                "  --ubatch-size N         physical ubatch size     [512]\n", argv[0]);
            return 0;
        }
        if (!strcmp(a, "--parallel") || !strcmp(a, "--checkpoints") ||
            !strcmp(a, "--cache-type-k") || !strcmp(a, "--cache-type-v") || !strcmp(a, "--ubatch-size")) {
            if (++i >= argc) { fprintf(stderr, "error: '%s' requires a value\n", a); return 1; }
        }
        if      (!strcmp(a, "--parallel"))     parallel_arg    = atoi(argv[i]);
        else if (!strcmp(a, "--checkpoints"))  checkpoints_arg = atoi(argv[i]);
        else if (!strcmp(a, "--flash-attn"))   flash_attn      = true;
        else if (!strcmp(a, "--no-flash-attn"))flash_attn      = false;
        else if (!strcmp(a, "--cache-type-k")) { type_k = !strcmp(argv[i],"f16")?GGML_TYPE_F16: !strcmp(argv[i],"q8_0")?GGML_TYPE_Q8_0:GGML_TYPE_F16; }
        else if (!strcmp(a, "--cache-type-v")) { type_v = !strcmp(argv[i],"f16")?GGML_TYPE_F16: !strcmp(argv[i],"q8_0")?GGML_TYPE_Q8_0:GGML_TYPE_F16; }
        else if (!strcmp(a, "--ubatch-size"))  ubatch_size     = (uint32_t)atoi(argv[i]);
        else if (!g_path)                      g_path          = argv[i];
        else { fprintf(stderr, "error: unexpected arg '%s'\n", a); return 1; }
    }
    if (!g_path) { fprintf(stderr, "error: no model path\n"); return 1; }

    llama_log_set(log_cb, NULL);
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.load_mtp = true;   // load nextn weights so the MTP context can exist

    struct llama_context_params cp = llama_context_default_params();
    cp.kv_unified  = true;
    cp.n_ctx       = 0;    // natural context; pool extends the cache, never n_ctx
    cp.n_batch     = 2048;
    cp.n_ubatch    = ubatch_size;
    cp.type_k      = type_k;
    cp.type_v      = type_v;
    cp.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (parallel_arg >= 0) { cp.n_seq_max = (uint32_t)parallel_arg; cp.n_rs_seq = 8; }
    else                   { cp.n_seq_max = 1; cp.n_rs_seq = (checkpoints_arg >= 0) ? (uint32_t)checkpoints_arg - 1 : 0; }

    // The 4 scenarios. bounds[g] = first layer on device g (loader rule: layer
    // il -> last dev whose bound <= il). Both provisional runs pin 0..37/38..64.
    const struct run_cfg runs[4] = {
        { "1 provisional (native split 0..37/38..64) @ small pool",
          {0, 38}, 2, 16384 },
        { "2 provisional (SAME split) @ large pool",
          {0, 38}, 2, 81920 },
        { "3 explicit 0..42/43..64 @ small pool",
          {0, 43}, 2, 16384 },
        { "4 explicit 0..42/43..64 @ large pool",
          {0, 43}, 2, 81920 },
    };

    int failures = 0;
    for (int r = 0; r < 4; ++r) failures += run_single_test(g_path, &mp, &cp, &runs[r]);

    llama_backend_free();
    fprintf(stderr, "\n\nkv-fit-deviation: done (%d failed run(s))\n", failures);
    return failures ? 1 : 0;
}
