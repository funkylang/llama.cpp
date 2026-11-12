// [FORK] gguf-info — model metadata and tensor analysis tool
//
// Reads GGUF file directly for tensor sizes + loads model (no weights) for
// architecture hparams to compute:
//   1. model parameters (name, arch, KV types)
//   2. tensor weight sizes per layer (+ embed/norm global tensors)
//   3. KV cell bytes per layer from architecture metadata
//      - hybrid models: only non-recurrent layers allocate unified pool KV
//      - quantized KV types: block-rounded via ggml_row_size (allocator-identical)
//   4. fixed per-layer state in the swa/ssm column:
//      - SWA attention layers: one sliding window = n_swa x cell_bytes
//      - recurrent (SSM) layers: one snapshot row-set of R/S tensors
//   5. MTP/nextn usage in dedicated mtp columns — their weights are skipped
//      by the loader unless load_mtp, and their KV pool is a separate context;
//      base deployment cost (non-mtp) and the ADDITIONAL mtp cost are both
//      visible at a glance
//   6. totals — tensor TOTAL equals the exact sum of all GGUF tensors
//
// Table columns (fixed order, optional ones conditional on model capability):
//   layer | tensor size | [mtp tensor] | cell size | [mtp cell] | [swa/ssm]
//   - mtp columns only if the model has nextn/MTP layers (n_layer_nextn > 0)
//   - swa/ssm column (always last) only if the model has SWA or recurrent
//     layers: per-layer fixed state cost — "we will always need one to hold
//     the current state":
//       SWA attention layer: one sliding window = n_swa x cell_bytes
//       recurrent (SSM) layer: one snapshot row-set of its R/S tensors
//         (llama-memory-recurrent.cpp: r/s allocated F32 per create_memory,
//          one live row + n_rs_seq snapshot rows per sequence share the cost
//          of a single row-set each); PLE layers add their conv-history row

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

static const int NUM_COL_W = 13; // numeric column cell width (incl. side padding)

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

// ---- table printing -----------------------------------------------------------
// widths[] holds per-column cell width INCLUDING the 1-space padding on each
// side of the content. Column 0 is a left-aligned label; all others are
// right-aligned numbers.

static std::string box(const char * tl, const char * mid, const char * tr,
                       const std::vector<int> & widths) {
    std::string s(tl);
    for (size_t i = 0; i < widths.size(); ++i) {
        if (i > 0) s += mid;
        for (int j = 0; j < widths[i]; ++j) s += "─";
    }
    return s + tr;
}

static std::string data_row(const char * label, const std::vector<int64_t> & vals,
                            const std::vector<int> & widths) {
    std::string s("│");
    const int l = (int)strlen(label);
    // col 0: right-aligned (layer numbers and labels share the column)
    s += std::string(widths[0] - 1 - l, ' ') + label + " │";
    for (size_t i = 1; i < vals.size(); ++i) {
        char buf[24];
        snprintf(buf, sizeof(buf), "%lld", (long long)vals[i]);
        const int n = (int)strlen(buf);
        s += std::string(widths[i] - n - 1, ' ') + buf + " │";
    }
    return s;
}

static std::string header_row(const std::vector<const char *> & hdr,
                              const std::vector<int> & widths) {
    std::string s("│");
    for (size_t i = 0; i < hdr.size(); ++i) {
        const int l = (int)strlen(hdr[i]);
        // visually centered over the full width; " │" supplies one right
        // padding space, so subtract it from pad_r. On uneven padding the
        // text leans left:
        const int pad_l = (widths[i] - l) / 2;
        const int pad_r = widths[i] - l - pad_l - 1;
        s += std::string(pad_l, ' ') + hdr[i] + std::string(pad_r, ' ') + " │";
    }
    return s;
}

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
    const uint32_t n_layer_all   = hp.n_layer_all;
    const uint32_t n_layer_nextn = hp.n_layer_nextn;

    // ---- capabilities -------------------------------------------------------
    const bool has_moe = hp.n_expert > 0;
    const bool has_mtp = n_layer_nextn > 0;
    bool has_swa = false, has_ssm = false;
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        has_swa |= hp.is_swa(il);
        has_ssm |= hp.is_recr(il);
    }

    // ---- parameters ---------------------------------------------------------
    fprintf(stderr, "\n");
    fprintf(stderr, "  file:       %s\n", path);
    fprintf(stderr, "  name:       %s\n", name ? name : "(unknown)");
    fprintf(stderr, "  arch:       %s\n", arch ? arch : llm_arch_name(model->arch));
    fprintf(stderr, "  layers:     %u (all)\n", (unsigned)n_layer_all);
    fprintf(stderr, "  MoE:        %s\n", has_moe ? "yes" : "no");
    fprintf(stderr, "  SWA:        %s\n", has_swa ? "yes" : "no");
    fprintf(stderr, "  SSM:        %s\n", has_ssm ? "yes" : "no");
    fprintf(stderr, "  MTP:        %s\n", has_mtp ? "yes" : "no");
    fprintf(stderr, "  kv_type_k:  %s\n", type_name(type_k));
    fprintf(stderr, "  kv_type_v:  %s\n", type_name(type_v));

    if (!other_tensors.empty()) {
	fprintf(stderr, "\n");
	for (const auto & [tname, tsize] : other_tensors) {
	    fprintf(stderr, "  %-45s %12lld\n", tname.c_str(), (long long)tsize);
	}
    }

    // ---- table layout: mtp + swa columns are capability-driven --------------

    std::vector<const char *> hdr;
    std::vector<int> widths;
    hdr.push_back("layer");       widths.push_back(7);   // label/layer column
    hdr.push_back("tensor size"); widths.push_back(NUM_COL_W);
    if (has_mtp) { hdr.push_back("mtp tensor"); widths.push_back(NUM_COL_W); }
    hdr.push_back("cell size");   widths.push_back(NUM_COL_W);
    if (has_mtp) { hdr.push_back("mtp cell");   widths.push_back(NUM_COL_W); }
    if (has_swa || has_ssm) { hdr.push_back("swa/ssm"); widths.push_back(NUM_COL_W); }

    auto emit_row = [&](const char * label, int64_t tsize, int64_t mtens,
                        int64_t csize, int64_t mcell, int64_t state) {
        std::vector<int64_t> vals;
        vals.push_back(0);                     // slot 0 is the label (unused)
        vals.push_back(tsize);                 // tensor size        (always)
        if (has_mtp) vals.push_back(mtens);    // mtp tensor         (optional)
        vals.push_back(csize);                 // cell size          (always)
        if (has_mtp) vals.push_back(mcell);    // mtp cell           (optional)
        if (has_swa || has_ssm) vals.push_back(state); // swa/ssm     (optional, last)
        fprintf(stderr, "%s\n", data_row(label, vals, widths).c_str());
    };

    int64_t total_weights = 0;
    int64_t mtp_weights = 0;
    int64_t total_rate = 0, mtp_rate = 0;
    int64_t total_state = 0, mtp_state = 0;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s\n", box("┌", "┬", "┐", widths).c_str());
    fprintf(stderr, "%s\n", header_row(hdr, widths).c_str());
    fprintf(stderr, "%s\n", box("├", "┼", "┤", widths).c_str());

    if (embed_bytes > 0) {
        emit_row("embed", embed_bytes, 0, 0, 0, 0);
        total_weights += embed_bytes;
    }

    for (uint32_t il = 0; il < n_layer_all; il++) {
	int64_t weights = layer_weights[il];
	const bool is_mtp_layer = il >= (n_layer_all - n_layer_nextn);

	// base deployment cost: without MTP, nextn layers are neither loaded
	// (load_mtp=false -> TENSOR_SKIP) nor executed — their weights and KV
	// appear in the mtp columns as the ADDITIONAL cost when MTP is enabled.
	total_weights += weights;
	if (is_mtp_layer) { mtp_weights += weights; }

	int64_t tsize = 0, mtens = 0, csize = 0, mcell = 0, state = 0;

	// Skip layers without KV cache OR recurrent layers (hybrid models) —
	// they don't allocate unified pool KV cells
	if (hp.has_kv(il) && !hp.is_recr(il)) {
		// block-rounded rows: identical to what the allocator computes,
		// correct for quantized KV types (q8_0 packs 32 elements + scale)
		const uint32_t n_embd_k = hp.n_embd_k_gqa(il);
		const uint32_t n_embd_v = v_trans ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);

		int64_t k_bytes = ggml_row_size(type_k, n_embd_k);
		int64_t v_bytes = hp.is_mla() ? 0 : (int64_t)ggml_row_size(type_v, n_embd_v);
		int64_t cell_bytes = k_bytes + v_bytes;

		// one sliding window per SWA layer: fixed cost that always exists
		// ("we will always need one to hold the current state")
		state = hp.is_swa(il) ? (int64_t)hp.n_swa * cell_bytes : 0;

		if (is_mtp_layer) {
			mtens = weights; mcell = cell_bytes;
			mtp_rate += cell_bytes; mtp_state += state;
		} else {
			tsize = weights; csize = cell_bytes;
			total_rate += cell_bytes; total_state += state;
		}
	} else {
		// recurrent / no-KV layer: its weights deploy with the base model
		tsize = is_mtp_layer ? 0 : weights;
		mtens = is_mtp_layer ? weights : 0;

		if (hp.is_recr(il)) {
			// one snapshot row-set of this recurrent layer's R/S tensors:
			// the live row and every rollback snapshot cost the same row-set
			// each (llama-memory-recurrent.cpp allocates n_rows =
			// mem_size * (1 + n_rs_seq)); r/s are F32 per create_memory,
			// PLE layers add their conv-history row
			state  = (int64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_r());
			state += (int64_t)ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
			if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
				state += (int64_t)ggml_row_size(GGML_TYPE_F32, hp.ple_conv_state());
			}
			if (is_mtp_layer) { mtp_state += state; } else { total_state += state; }
		}
	}

	char label[16];
	snprintf(label, sizeof(label), "%u", il);
	emit_row(label, tsize, mtens, csize, mcell, state);
    }

    for (const auto & [tname, tsize] : other_tensors) {
        total_weights += tsize;
    }

    if (norm_bytes > 0) {
        emit_row("norm", norm_bytes, 0, 0, 0, 0);
        total_weights += norm_bytes;
    }

    // ---- totals -------------------------------------------------------------
    int64_t base_weights = total_weights - mtp_weights;
    fprintf(stderr, "%s\n", box("├", "┼", "┤", widths).c_str());
    emit_row("TOTAL", base_weights, has_mtp ? mtp_weights : 0,
             total_rate, has_mtp ? mtp_rate : 0,
             (has_swa || has_ssm) ? (total_state + mtp_state) : 0);
    fprintf(stderr, "%s\n", box("└", "┴", "┘", widths).c_str());

    llama_model_free(model);
    gguf_free(ctx);
    llama_backend_free();
    return 0;
}
