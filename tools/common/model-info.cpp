// shared model header + layer table for the fork's sizing tools.
// Ported from tools/gpu-split (2026-11-17) so every tool prints the same
// block; see model-info.h for the contract.

#include "model-info.h"

#include <llama-impl.h>
#include <llama-model.h>
#include <gguf.h>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int NUM_COL_W = 13; // numeric column cell width (incl. side padding)

std::string box(const char * tl, const char * mid, const char * tr,
                const std::vector<int> & widths) {
    std::string s(tl);
    for (size_t i = 0; i < widths.size(); ++i) {
        if (i > 0) s += mid;
        for (int j = 0; j < widths[i]; ++j) s += "─";
    }
    return s + tr;
}

std::string data_row(const char * label, const std::vector<int64_t> & vals,
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

std::string header_row(const std::vector<const char *> & hdr,
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



} // namespace

// complete definition of the opaque handle declared in model-info.h; it must
// live in the global namespace so the extern "C" accessors can see it.
struct mi_model_info {
    char path[1024];
    const llama_model * model = nullptr;
    const char * arch = nullptr;      // valid while the file is open (gguf)
    const char * name = nullptr;      // ditto
    uint32_t n_layer_all = 0;
    uint32_t n_layer_nextn = 0;

    // per-layer classification from the GGUF tensor names (no_alloc peek):
    std::map<int, int64_t> layer_weights;
    std::vector<bool> layer_has_exps;   // expert tensors -> MoE fallback flag
    int64_t embed_bytes = 0, norm_bytes = 0, output_bytes = 0;
    std::vector<std::pair<std::string, int64_t>> other_tensors;

    struct gguf_context * ctx = nullptr;   // kept until free() for the name/arch pointers
};

extern "C" mi_model_info_t mi_model_info_new(const char * path, const struct llama_model * model) {
    if (!path || !model) return nullptr;

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        return nullptr;
    }

    auto info = std::make_unique<mi_model_info>();
    strncpy(info->path, path, sizeof(info->path) - 1);
    info->path[sizeof(info->path) - 1] = '\0';
    info->model = model;
    info->ctx   = ctx;

    int64_t id_arch = gguf_find_key(ctx, "general.architecture");
    if (id_arch >= 0) info->arch = gguf_get_val_str(ctx, id_arch);

    int64_t id_name = gguf_find_key(ctx, "general.name");
    if (id_name >= 0) info->name = gguf_get_val_str(ctx, id_name);

    const auto & hp = model->hparams;
    info->n_layer_all   = hp.n_layer_all;
    info->n_layer_nextn = hp.n_layer_nextn;

    // nextn block indices (runtime layout): blk.nextn.<i> maps to hp layer
    // n_layer + i — the trunk occupies [0, n_layer), nextn sits right after.
    int64_t n_layer_meta = 0;
    if (info->arch) {
        const std::string nk = std::string(info->arch) + ".n_layer";
        int64_t id_nlayer = gguf_find_key(ctx, nk.c_str());
        if (id_nlayer >= 0) {
            n_layer_meta = (int64_t)gguf_get_val_u32(ctx, id_nlayer);
        }
    }

    // per-layer routed-expert flags derived from the GGUF tensor names —
    // needed because some architectures (e.g. granite-hybrid) never fill
    // n_ff_exp_arr and key their experts off a global n_expert only.
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(ctx, tid);
        const size_t tsize = gguf_get_tensor_size(ctx, tid);

        if (!strcmp(tname, "token_embd.weight")) { info->embed_bytes += tsize; continue; }
        if (!strcmp(tname, "output_norm.weight")) { info->norm_bytes += tsize; continue; }
        if (!strcmp(tname, "output.weight"))      { info->output_bytes += tsize; continue; }

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
            info->layer_weights[il] += tsize;
            // routed-expert tensors mark this layer as MoE (see note above)
            const bool exp_tensor = strstr(tname, "ffn_gate_exp") || strstr(tname, "ffn_up_exp") ||
                                    strstr(tname, "ffn_down_exp") || strstr(tname, "ffn_gate.") ||
                                    strstr(tname, "ffn_up.") || strstr(tname, "ffn_down.");
            if (exp_tensor) {
                if (il >= (int)info->layer_has_exps.size()) info->layer_has_exps.resize(il + 1, false);
                info->layer_has_exps[il] = true;
            }
            continue;
        }

        info->other_tensors.emplace_back(tname, (int64_t)tsize);
    }

    return info.release();
}

extern "C" int mi_print_header(FILE * out, mi_model_info_t info,
                               enum ggml_type type_k, enum ggml_type type_v,
                               const char * keep_free) {
    if (!info || !out) return 0;
    const auto & hp = info->model->hparams;
    const uint32_t n_layer_all   = info->n_layer_all;
    const uint32_t n_layer_nextn = info->n_layer_nextn;

    // per-layer MoE predicate: hparams experts OR expert tensors in the file
    const auto is_moe_layer = [&](uint32_t il) -> bool {
        return hp.n_ff_exp(il) > 0 ||
               (il < info->layer_has_exps.size() && info->layer_has_exps[il]);
    };

    // ---- capabilities -------------------------------------------------------
    const bool has_moe = hp.n_expert > 0;
    const bool has_mtp = n_layer_nextn > 0;
    bool has_swa = false, has_ssm = false;
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        has_swa |= hp.is_swa(il);
        has_ssm |= hp.is_recr(il);
    }

    // ---- header block -------------------------------------------------------
    fprintf(out, "\n");
    fprintf(out, "  file:       %s\n", info->path);
    fprintf(out, "  name:       %s\n", info->name ? info->name : "(unknown)");
    fprintf(out, "  arch:       %s\n", info->arch ? info->arch : llm_arch_name(info->model->arch));
    fprintf(out, "  layers:     %u (all)\n", (unsigned)n_layer_all);

    // per-layer MoE membership = has routed experts (n_ff_exp(il) > 0);
    // dense runs and MoE runs are reported as compact layer ranges
    {
        bool any_dense = false;
        std::string scheme;
        for (uint32_t il = 0; il < n_layer_all; ) {
            const bool moe_layer = is_moe_layer(il);
            uint32_t j = il;
            while (j < n_layer_all && is_moe_layer(j) == moe_layer) ++j;
            char range[48];
            snprintf(range, sizeof(range), "%s%u-%u", scheme.empty() ? "" : " ", il, j - 1);
            scheme += moe_layer ? range : std::string("d") + range; // 'd' = dense run
            if (!moe_layer) any_dense = true;
            il = j;
        }
        if (has_moe) {
            fprintf(out, "  MoE:        yes (%u experts, top-%u)%s\n",
                    (unsigned)hp.n_expert, (unsigned)hp.n_expert_used_max(),
                    any_dense ? (" scheme: " + scheme).c_str() : "");
        } else {
            fprintf(out, "  MoE:        no\n");
        }
    }
    fprintf(out, "  SWA:        %s\n", has_swa ? "yes" : "no");
    fprintf(out, "  SSM:        %s\n", has_ssm ? "yes" : "no");
    fprintf(out, "  MTP:        %s\n", has_mtp ? "yes" : "no");
    fprintf(out, "  kv_type_k:  %s\n", ggml_type_name(type_k));
    fprintf(out, "  kv_type_v:  %s\n", ggml_type_name(type_v));
    if (keep_free) {
        fprintf(out, "  keep_free:  %s\n", keep_free);
    }

    // the token embedding ALWAYS stays on the CPU (llama-model.cpp: "there is
    // very little benefit to offloading the input layer") — it is not part of
    // what gets distributed to GPUs, so it is shown here, outside the table.
    // EXCEPTION: tied models (no separate output.weight) get a second copy as
    // the lm_head on the output device — that copy appears as the LAST ROW of
    // the table (the bytes to distribute).
    if (info->embed_bytes > 0) {
        fprintf(out, "  embed(CPU): %lld\n", (long long)info->embed_bytes);
    }

    return 1;
}

extern "C" int mi_print_layer_table(FILE * out, mi_model_info_t info,
                                    enum ggml_type type_k, enum ggml_type type_v,
                                    bool flash_attn) {
    if (!info || !out) return 0;
    const auto & hp = info->model->hparams;
    const uint32_t n_layer_all   = info->n_layer_all;
    const uint32_t n_layer_nextn = info->n_layer_nextn;

    const auto is_moe_layer = [&](uint32_t il) -> bool {
        return hp.n_ff_exp(il) > 0 ||
               (il < info->layer_has_exps.size() && info->layer_has_exps[il]);
    };

    const bool has_moe = hp.n_expert > 0;
    const bool has_mtp = n_layer_nextn > 0;
    bool has_swa = false, has_ssm = false;
    for (uint32_t il = 0; il < n_layer_all; ++il) {
        has_swa |= hp.is_swa(il);
        has_ssm |= hp.is_recr(il);
    }

    const bool v_trans = !flash_attn;

    // ---- table layout: mtp + swa/ssm columns are capability-driven ----------
    int col0_w = 7; // at least "layer"
    for (const auto & [tname, tsize] : info->other_tensors) {
        const int l = (int)tname.size() + 2;
        if (l > col0_w) col0_w = l;
    }

    std::vector<const char *> hdr;
    std::vector<int> widths;
    hdr.push_back("layer");       widths.push_back(col0_w); // label/layer column
    hdr.push_back("tensor size"); widths.push_back(NUM_COL_W);
    if (has_moe) { hdr.push_back("moe"); widths.push_back(5); }
    if (has_mtp) { hdr.push_back("mtp tensor"); widths.push_back(NUM_COL_W); }
    hdr.push_back("cell size");   widths.push_back(NUM_COL_W);
    if (has_mtp) { hdr.push_back("mtp cell");   widths.push_back(NUM_COL_W); }
    if (has_swa || has_ssm) { hdr.push_back("swa/ssm"); widths.push_back(NUM_COL_W); }

    // accounting invariant (code-enforced): emit_row accumulates every weight
    // byte it prints; the TOTAL row is built from exactly these accumulators,
    // so TOTAL always equals the sum of the visible rows — no hidden tensors.
    int64_t printed_base = 0, printed_mtp = 0;

    auto emit_row = [&](const char * label, int64_t tsize, int moe_flag,
                        int64_t mtens, int64_t csize, int64_t mcell,
                        int64_t state) {
        std::vector<int64_t> vals;
        vals.push_back(0);                              // slot 0 is the label (unused)
        vals.push_back(tsize);                          // tensor size        (always)
        if (has_moe) vals.push_back(moe_flag);          // moe flag           (optional)
        if (has_mtp) vals.push_back(mtens);             // mtp tensor         (optional)
        vals.push_back(csize);                          // cell size          (always)
        if (has_mtp) vals.push_back(mcell);             // mtp cell           (optional)
        if (has_swa || has_ssm) vals.push_back(state);  // swa/ssm            (optional, last)
        fprintf(out, "%s\n", data_row(label, vals, widths).c_str());
        printed_base += tsize;                          // accumulate what was printed
        printed_mtp  += mtens;
    };

    int64_t total_rate = 0, mtp_rate = 0;
    int64_t total_state = 0, mtp_state = 0;

    fprintf(out, "\n");
    fprintf(out, "%s\n", box("┌", "┬", "┐", widths).c_str());
    fprintf(out, "%s\n", header_row(hdr, widths).c_str());
    fprintf(out, "%s\n", box("├", "┼", "┤", widths).c_str());

    for (uint32_t il = 0; il < n_layer_all; il++) {
        const int64_t weights = info->layer_weights[il];
        const bool is_mtp_layer = il >= (n_layer_all - n_layer_nextn);

        // base deployment cost: without MTP, nextn layers are neither loaded
        // (load_mtp=false -> TENSOR_SKIP) nor executed — their weights and KV
        // appear in the mtp columns as the ADDITIONAL cost when MTP is enabled.

        int64_t tsize = 0, mtens = 0, csize = 0, mcell = 0, state = 0;

        // Skip layers without KV cache OR recurrent layers (hybrid models) —
        // they don't allocate unified pool KV cells
        if (hp.has_kv(il) && !hp.is_recr(il)) {
            const uint32_t n_embd_k = hp.n_embd_k_gqa(il);
            const uint32_t n_embd_v = v_trans ? hp.n_embd_v_gqa_max() : hp.n_embd_v_gqa(il);

            int64_t k_bytes = ggml_row_size(type_k, n_embd_k);
            int64_t v_bytes = hp.is_mla() ? 0 : (int64_t)ggml_row_size(type_v, n_embd_v);
            int64_t cell_bytes = k_bytes + v_bytes;

            // one sliding window per SWA layer: fixed cost that always exists
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
        emit_row(label, tsize, is_moe_layer(il) ? 1 : 0, mtens, csize, mcell, state);
    }

    // globals after the last layer, in data flow order:
    // norm -> output (lm_head); any remaining non-blk tensors each get a row
    if (info->norm_bytes > 0) {
        emit_row("norm", info->norm_bytes, 0, 0, 0, 0, 0);
    }
    if (info->output_bytes > 0) {
        emit_row("output", info->output_bytes, 0, 0, 0, 0, 0);
    }
    for (const auto & [tname, tsize] : info->other_tensors) {
        emit_row(tname.c_str(), tsize, 0, 0, 0, 0, 0);
    }
    // tied lm_head: without a separate output tensor the loader duplicates the
    // token embedding as an OUTPUT tensor (llama-model-loader.cpp) — that GPU
    // copy is part of what must be distributed, hence this last row.
    if (info->output_bytes == 0 && info->embed_bytes > 0) {
        emit_row("embed", info->embed_bytes, 0, 0, 0, 0, 0);
    }

    // ---- totals: exactly the sum of the rows printed above ------------------
    fprintf(out, "%s\n", box("├", "┼", "┤", widths).c_str());
    {
        int64_t total_moe = 0;
        for (uint32_t il = 0; il < n_layer_all - n_layer_nextn; ++il) {
            if (is_moe_layer(il)) ++total_moe;
        }
        std::vector<int64_t> vals;
        vals.push_back(0);                          // label slot (unused)
        vals.push_back(printed_base);               // sum of visible rows
        if (has_moe) vals.push_back(total_moe);     // moe layer count
        if (has_mtp) vals.push_back(printed_mtp);
        vals.push_back(total_rate);
        if (has_mtp) vals.push_back(mtp_rate);
        if (has_swa || has_ssm) vals.push_back(total_state + mtp_state);
        fprintf(out, "%s\n", data_row("TOTAL", vals, widths).c_str());
    }
    fprintf(out, "%s\n", box("└", "┴", "┘", widths).c_str());

    return 1;
}

extern "C" void mi_model_info_free(mi_model_info_t info) {
    if (!info) return;
    if (info->ctx) gguf_free(info->ctx);
    delete info;
}
