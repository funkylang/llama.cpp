// [FORK] gguf-info — model metadata and tensor analysis tool
//
// Reads GGUF file directly for tensor sizes + loads model (no weights) for
// architecture hparams to compute:
//   1. model parameters (name, arch, KV types)
//   2. tensor weight sizes per layer
//   3. KV cell bytes per layer from architecture metadata
//   4. totals

#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>

static const char * type_name(int t) {
    switch (t) {
	case GGML_TYPE_F16:  return "f16";
	case GGML_TYPE_Q8_0: return "q8_0";
	case GGML_TYPE_Q4_0: return "q4_0";
	case GGML_TYPE_F32:  return "f32";
	default:             return ggml_type_name((ggml_type)t);
    }
}

static bool set_kv_type(const char * s, ggml_type & out) {
    if      (!strcmp(s, "f16"))   { out = GGML_TYPE_F16;  return true; }
    else if (!strcmp(s, "q8_0"))  { out = GGML_TYPE_Q8_0; return true; }
    else if (!strcmp(s, "q4_0"))  { out = GGML_TYPE_Q4_0; return true; }
    else if (!strcmp(s, "f32"))   { out = GGML_TYPE_F32;  return true; }
    fprintf(stderr, "unknown kv type '%s', keeping current\n", s);
    return false;
}

static void log_callback_quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 2) {
	fprintf(stderr, "usage: %s <model.gguf> [-type-k f16|q8_0] [-type-v f16|q8_0]\n"
			"                       [-flash-attn|-no-flash-attn]\n", argv[0]);
	return 1;
    }

    const char * path = argv[1];
    ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    bool flash_attn = true;

    for (int a = 2; a < argc; a++) {
	std::string arg = argv[a];
	if      (arg == "-type-k")         set_kv_type(argv[++a], type_k);
	else if (arg == "-type-v")         set_kv_type(argv[++a], type_v);
	else if (arg == "-flash-attn")     flash_attn = true;
	else if (arg == "-no-flash-attn")  flash_attn = false;
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
    for (int64_t tid = 0; tid < n_tensors; tid++) {
	const char * tname = gguf_get_tensor_name(ctx, tid);
	size_t tsize = gguf_get_tensor_size(ctx, tid);

	int il = -1;
	if (strstr(tname, "blk.")) {
	    const char * p = strstr(tname, "blk.");
	    sscanf(p + 4, "%d", &il);
	}
	if (il >= 0) layer_weights[il] += tsize;
    }

    // ---- load model via llama API for hparams (no weights) -----------------
    llama_log_set(log_callback_quiet, nullptr);  // suppress verbose loading logs

    llama_backend_init();

    llama_model_params mp; memset(&mp, 0, sizeof(mp));
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
	fprintf(stderr, "failed to load model metadata\n");
	gguf_free(ctx);
	return 1;
    }

    const auto & hp = model->hparams;
    const uint32_t n_layer_all = hp.n_layer_all;

    // ---- parameters ---------------------------------------------------------
    fprintf(stderr, "\n");
    fprintf(stderr, "  file:       %s\n", path);
    fprintf(stderr, "  name:       %s\n", name ? name : "(unknown)");
    fprintf(stderr, "  arch:       %s\n", arch ? arch : llm_arch_name(model->arch));
    fprintf(stderr, "  layers:     %u (all)\n", (unsigned)n_layer_all);
    fprintf(stderr, "  kv_type_k:  %s\n", type_name(type_k));
    fprintf(stderr, "  kv_type_v:  %s\n", type_name(type_v));

    // ---- per-layer table with block characters ------------------------------
    fprintf(stderr, "\n");
    fprintf(stderr, "┌───────┬─────────────┬───────────┐\n");
    fprintf(stderr, "│ layer │ tensor size │ cell size │\n");
    fprintf(stderr, "├───────┼─────────────┼───────────┤\n");

    int64_t total_weights = 0;
    int64_t total_rate = 0;

    for (uint32_t il = 0; il < n_layer_all; il++) {
	int64_t weights = layer_weights[il];
	total_weights += weights;

	if (!hp.has_kv(il)) {
	    fprintf(stderr, "│ %5u  │ %11lld │     -     │\n",
		    (unsigned)il, (long long)weights);
	    continue;
	}

	const uint32_t n_embd_k = hp.n_embd_k_gqa(il);
	const uint32_t n_embd_v = v_trans ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);

	int64_t k_bytes = (int64_t)n_embd_k * ggml_type_size(type_k);
	int64_t v_bytes = hp.is_mla() ? 0 : (int64_t)n_embd_v * ggml_type_size(type_v);
	int64_t cell_bytes = k_bytes + v_bytes;
	total_rate += cell_bytes;

	fprintf(stderr, "│ %5u │ %11lld │ %9lld │\n",
		(unsigned)il, (long long)weights, (long long)cell_bytes);
    }

    // ---- totals -------------------------------------------------------------
    fprintf(stderr, "├───────┼─────────────┼───────────┤\n");
    fprintf(stderr, "│ TOTAL │ %11lld │ %9lld │\n",
	    (long long)total_weights, (long long)total_rate);
    fprintf(stderr, "└────────┴────────────┴───────────┘\n");

    llama_model_free(model);
    gguf_free(ctx);
    llama_backend_free();
    return 0;
}
