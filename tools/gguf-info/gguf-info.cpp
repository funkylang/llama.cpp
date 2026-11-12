// [FORK] gguf-info — model metadata and tensor analysis tool
//
// Reads GGUF file directly for tensor sizes + loads model (no weights) for
// architecture hparams to compute:
//   1. model parameters (name, arch, KV types)
//   2. tensor weight sizes per layer (+ embed/norm global tensors)
//   3. KV cell bytes per layer from architecture metadata
//      - hybrid models: only non-recurrent layers allocate unified pool KV
//      - quantized KV types: block-rounded via ggml_row_size (allocator-identical)
//   4. SWA window size per layer: one sliding window = n_swa x cell_bytes
//   5. MTP/nextn layers reported separately (they belong to the MTP context,
//      not the main unified pool)
//   6. totals — tensor TOTAL equals the exact sum of all GGUF tensors

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
#include <vector>

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
    // global tensors: token_embd.weight ("embed", before layer 0),
    // output_norm.weight ("norm", after the last layer); every other
    // non-blk tensor is listed above the table but still counted in TOTAL.
    int64_t embed_bytes = 0, norm_bytes = 0;
    std::vector<std::pair<std::string, int64_t>> other_tensors;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
	const char * tname = gguf_get_tensor_name(ctx, tid);
	size_t tsize = gguf_get_tensor_size(ctx, tid);

	if (!strcmp(tname, "token_embd.weight")) { embed_bytes += tsize; continue; }
	if (!strcmp(tname, "output_norm.weight")) { norm_bytes += tsize; continue; }

	int il = -1;
	if (strstr(tname, "blk.")) {
	    const char * p = strstr(tname, "blk.");
	    sscanf(p + 4, "%d", &il);
	}
	if (il >= 0) { layer_weights[il] += tsize; continue; }

	other_tensors.emplace_back(tname, (int64_t)tsize);
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
    const uint32_t n_layer_nextn = hp.n_layer_nextn;

    // ---- parameters ---------------------------------------------------------
    fprintf(stderr, "\n");
    fprintf(stderr, "  file:       %s\n", path);
    fprintf(stderr, "  name:       %s\n", name ? name : "(unknown)");
    fprintf(stderr, "  arch:       %s\n", arch ? arch : llm_arch_name(model->arch));
    fprintf(stderr, "  layers:     %u (all)\n", (unsigned)n_layer_all);
    fprintf(stderr, "  kv_type_k:  %s\n", type_name(type_k));
    fprintf(stderr, "  kv_type_v:  %s\n", type_name(type_v));

    // ---- per-layer table with block characters ------------------------------
    if (!other_tensors.empty()) {
	fprintf(stderr, "\n");
	for (const auto & [tname, tsize] : other_tensors) {
	    fprintf(stderr, "  %-45s %12lld\n", tname.c_str(), (long long)tsize);
	}
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "┌───────┬─────────────┬───────────┬───────────┐\n");
    fprintf(stderr, "│ layer │ tensor size │ cell size │ swa size  │\n");
    fprintf(stderr, "├───────┼─────────────┼───────────┼───────────┤\n");

    int64_t total_weights = 0;
    int64_t total_rate = 0, total_swa = 0;
    int64_t mtp_weights = 0, mtp_rate = 0, mtp_swa = 0;

    if (embed_bytes > 0) {
        fprintf(stderr, "│ embed │ %11lld │         0 │         0 │\n", (long long)embed_bytes);
        total_weights += embed_bytes;
    }

    for (uint32_t il = 0; il < n_layer_all; il++) {
	int64_t weights = layer_weights[il];
	total_weights += weights;

	const bool is_mtp_layer = il >= (n_layer_all - n_layer_nextn);

	// Skip layers without KV cache OR recurrent layers (hybrid models) —
	// they don't allocate unified pool KV cells
	if (!hp.has_kv(il) || hp.is_recr(il)) {
	    fprintf(stderr, "│ %5u │ %11lld │         0 │         0 │\n",
		    (unsigned)il, (long long)weights);
	    continue;
	}

	// block-rounded rows: identical to what the allocator computes,
	// correct for quantized KV types (q8_0 packs 32 elements + scale)
	const uint32_t n_embd_k = hp.n_embd_k_gqa(il);
	const uint32_t n_embd_v = v_trans ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);

	int64_t k_bytes = ggml_row_size(type_k, n_embd_k);
	int64_t v_bytes = hp.is_mla() ? 0 : (int64_t)ggml_row_size(type_v, n_embd_v);
	int64_t cell_bytes = k_bytes + v_bytes;

	// one sliding window per SWA layer: fixed cost that always exists
	// ("we will always need one to hold the current state")
	int64_t swa_bytes = hp.is_swa(il) ? (int64_t)hp.n_swa * cell_bytes : 0;

	if (is_mtp_layer) {
	    // nextn/MTP layers belong to the MTP context, not the main pool
	    mtp_weights += weights;
	    mtp_rate    += cell_bytes;
	    mtp_swa     += swa_bytes;
	} else {
	    total_rate += cell_bytes;
	    total_swa  += swa_bytes;
	}

	fprintf(stderr, "│ %5u │ %11lld │ %9lld │ %9lld │%s\n",
		(unsigned)il, (long long)weights, (long long)cell_bytes,
		(long long)swa_bytes, is_mtp_layer ? "  (mtp)" : "");
    }

    for (const auto & [tname, tsize] : other_tensors) {
        total_weights += tsize;
    }

    if (norm_bytes > 0) {
        fprintf(stderr, "│  norm │ %11lld │         0 │         0 │\n", (long long)norm_bytes);
        total_weights += norm_bytes;
    }

    if (n_layer_nextn > 0) {
	fprintf(stderr, "│  mtp  │ %11lld │ %9lld │ %9lld │\n",
		(long long)mtp_weights, (long long)mtp_rate, (long long)mtp_swa);
    }

    // ---- totals -------------------------------------------------------------
    fprintf(stderr, "├───────┼─────────────┼───────────┼───────────┤\n");
    fprintf(stderr, "│ TOTAL │ %11lld │ %9lld │ %9lld │\n",
        (long long)total_weights, (long long)total_rate, (long long)total_swa);
    fprintf(stderr, "└───────┴─────────────┴───────────┴───────────┘\n");

    llama_model_free(model);
    gguf_free(ctx);
    llama_backend_free();
    return 0;
}
