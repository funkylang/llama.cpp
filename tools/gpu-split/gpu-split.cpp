// [FORK] gpu-split — GPU device scan + device placement & KV pool planner
//
// Current stage: DEVICE SCAN (STEP 0 of concept-gpu-split-tool.md)
//   Enumerates every device the ggml backend registry sees and prints its
//   VRAM in two phases: free (raw, nothing loaded) and after-init (after each
//   GPU/IGPU backend uploaded its shaders/fixed driver parts) — the figure to
//   plan from. CPU has no shaders: both phases measure the same thing.
//   The split solver (phases 2-5 of the concept doc) builds on this; the
//   GGUF/hparams table code below is carried over from llama-gguf-info so
//   both tools share the same accounting logic.
//
// The full model table (per-layer tensor sizes, KV cell bytes, mtp/swa/ssm
// columns, TOTAL) is computed exactly as in llama-gguf-info:
//   - what must be DISTRIBUTED TO THE GPUS — layers, norm/output globals,
//     and any other deployed non-blk tensors; the input embedding always
//     stays on CPU (shown separately as embed(CPU)) EXCEPT tied models,
//     where its lm_head copy joins the table as the last row
//   - KV cell bytes per layer from architecture metadata, selectable via
//     -ctk/-ctv like llama-server
//   - fixed per-layer state in the swa/ssm column (window / snapshot rows)
//   - MTP/nextn usage in dedicated mtp columns

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

// same allowed KV cache types as llama-server (common/arg.cpp)
static const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const char * s) {
    for (const auto & type : kv_cache_types) {
        if (!strcmp(ggml_type_name(type), s)) {
            return type;
        }
    }
    fprintf(stderr, "Unsupported cache type: %s\n", s);
    exit(1);
}

static std::string all_kv_cache_types() {
    std::string msg;
    for (const auto & type : kv_cache_types) {
        if (!msg.empty()) msg += ", ";
        msg += ggml_type_name(type);
    }
    return msg;
}

static void log_callback_quiet(enum ggml_log_level, const char *, void *) {}

// ---- device scan -----------------------------------------------------------
// STEP 0 (concept-gpu-split-tool.md §15.4/§15.6): two-phase measurement of
// every device the backend registry sees:
//   free       — raw, right after llama_backend_init(), nothing loaded yet
//   after-init — after each GPU/IGPU backend is initialized, i.e. shaders and
//                other fixed driver parts are uploaded; this is the real
//                "available before deployment" figure (Michael's "loaded" col)
// No filtering yet: the output shows exactly what the driver advertises on
// the rig. The GPU/IGPU filter for the split solver comes with phases 2+.

static const char * dev_type_str(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_META: return "META";
        default:                            return "other";
    }
}

// GiB with two decimals; VRAM figures are read in GiB for planning
static std::string gib_str(size_t bytes) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

struct dev_snapshot { size_t free_bytes = 0, total_bytes = 0; bool ok = false; };

static void read_all_devices(std::vector<dev_snapshot> & snaps) {
    const size_t ndevs = ggml_backend_dev_count();
    snaps.resize(ndevs);
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        snaps[i].free_bytes = 0;
        snaps[i].total_bytes = 0;
        ggml_backend_dev_memory(dev, &snaps[i].free_bytes, &snaps[i].total_bytes);
        snaps[i].ok = true;
    }
}

// On Linux the ggml CPU backend reports free == total unconditionally
// (ggml-cpu.cpp:377) — a deliberate assumption, not a measurement. Read the
// real figure from /proc/meminfo so the table is truthful there; leave the
// raw value where /proc/meminfo doesn't provide it (Windows, old kernels).
static bool meminfo_available(size_t * available) {
#if defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) {
        return false;
    }
    char line[256];
    size_t kb = 0;
    bool ok = false;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
            ok = true;
            break;
        }
    }
    fclose(f);
    *available = ok ? kb * 1024 : 0;
    return ok;
#else
    (void)available;
    return false;
#endif
}

// Replace the CPU rows' free with the real MemAvailable value (see
// meminfo_available). No shaders exist for CPUs, so apply to every phase —
// both measurements are the same quantity.
static void fix_cpu_free(std::vector<dev_snapshot> & snaps) {
    const size_t ndevs = ggml_backend_dev_count();
    for (size_t i = 0; i < ndevs && i < snaps.size(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU &&
            meminfo_available(&snaps[i].free_bytes)) {
            // total stays as reported; only free is corrected
        }
    }
}

static void print_device_scan(const std::vector<dev_snapshot> & raw,
                              const std::vector<dev_snapshot> & init) {
    fprintf(stderr, "\n");
    fprintf(stderr, "  device scan:\n\n");

    const size_t ndevs = ggml_backend_dev_count();
    if (ndevs == 0) {
        fprintf(stderr, "    no devices registered\n");
        return;
    }

    char hdr[256];
    snprintf(hdr, sizeof(hdr), "    %-4s %-6s %-14s %-14s %-14s  %s",
             "idx", "type", "total", "free", "after-init", "name");
    fprintf(stderr, "%s\n", hdr);

    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            fprintf(stderr, "    %-4zu (null)\n", i);
            continue;
        }

        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        const size_t free_init = (i < init.size() && init[i].ok) ? init[i].free_bytes : 0;

        char line[512];
        snprintf(line, sizeof(line),
                 "    %-4zu %-6s %-14s %-14s %-14s  %s",
                 i,
                 dev_type_str(type),
                 gib_str(raw[i].total_bytes).c_str(),
                 gib_str(raw[i].free_bytes).c_str(),
                 gib_str(free_init).c_str(),
                 ggml_backend_dev_name(dev));
        fprintf(stderr, "%s\n", line);
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "    (after-init = free after backend/shader upload)\n");
}

// Initialize each GPU/IGPU backend so shaders/fixed driver parts load; the
// initialized backends are returned for later cleanup (we free them once the
// scan is done — they are not needed for the metadata-only model load).
static void init_gpu_backends(std::vector<ggml_backend_t> & backends) {
    const size_t ndevs = ggml_backend_dev_count();
    for (size_t i = 0; i < ndevs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) {
            fprintf(stderr, "  device %zu (%s): backend init failed — after-init not measured\n",
                    i, ggml_backend_dev_name(dev));
        }
        backends.push_back(be);
    }
}

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
        fprintf(stderr, "usage: %s <model.gguf> [-ctk|--cache-type-k TYPE] [-ctv|--cache-type-v TYPE]\n"
                        "                       [-flash-attn|-no-flash-attn]\n"
                        "  allowed cache types: %s\n", argv[0], all_kv_cache_types().c_str());
        return 1;
    }

    const char * path = argv[1];
    ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    bool flash_attn = true;

    for (int a = 2; a < argc; a++) {
        std::string arg = argv[a];
        if      (arg == "-ctk" || arg == "--cache-type-k") type_k = kv_cache_type_from_str(argv[++a]);
        else if (arg == "-ctv" || arg == "--cache-type-v") type_v = kv_cache_type_from_str(argv[++a]);
        else if (arg == "-flash-attn")                     flash_attn = true;
        else if (arg == "-no-flash-attn")                  flash_attn = false;
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
    // per-layer routed-expert flags derived from the GGUF tensor names —
    // needed because some architectures (e.g. granite-hybrid) never fill
    // n_ff_exp_arr and key their experts off a global n_expert only.
    // Expert tensors: ffn_{gate,up,down}_exps (packed 3D) or the per-expert
    // naming ffn_{gate,up,down}.<exp_id>
    std::vector<bool> layer_has_exps;
    // global tensors (outside blk.*), each one a visible table row:
    //   token_embd.weight  -> "embed"  row (before layer 0)
    //   output_norm.weight -> "norm"   row (after the last layer)
    //   output.weight      -> "output" row (after norm — lm_head, data flow
    //                          order: embed -> layers -> norm -> output)
    // everything else non-blk gets its own named row as well — TOTAL must
    // equal the sum of the visible rows, never include hidden tensors.
    int64_t embed_bytes = 0, norm_bytes = 0, output_bytes = 0;
    std::vector<std::pair<std::string, int64_t>> other_tensors;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(ctx, tid);
        size_t tsize = gguf_get_tensor_size(ctx, tid);

        if (!strcmp(tname, "token_embd.weight")) { embed_bytes += tsize; continue; }
        if (!strcmp(tname, "output_norm.weight")) { norm_bytes += tsize; continue; }
        if (!strcmp(tname, "output.weight"))      { output_bytes += tsize; continue; }

        int il = -1;
        if (strstr(tname, "blk.")) {
            const char * p = strstr(tname, "blk.");
            sscanf(p + 4, "%d", &il);
        }
        if (il >= 0) {
            layer_weights[il] += tsize;
            // routed-expert tensors mark this layer as MoE (see note above)
            const bool exp_tensor = strstr(tname, "ffn_gate_exp") || strstr(tname, "ffn_up_exp") ||
                                    strstr(tname, "ffn_down_exp") || strstr(tname, "ffn_gate.") ||
                                    strstr(tname, "ffn_up.") || strstr(tname, "ffn_down.");
            if (exp_tensor) {
                if (il >= (int)layer_has_exps.size()) layer_has_exps.resize(il + 1, false);
                layer_has_exps[il] = true;
            }
            continue;
        }

        other_tensors.emplace_back(tname, (int64_t)tsize);
    }

    // ---- load model via llama API for hparams (no weights) -----------------
    llama_log_set(log_callback_quiet, nullptr);  // suppress verbose loading logs

    llama_backend_init();

    // STEP 0: two-phase device scan — raw VRAM, then after GPU backend init
    // (shaders/fixed driver parts). The initialized backends are freed right
    // after the scan: not needed for the metadata-only model load.
    std::vector<dev_snapshot> scan_raw, scan_init;
    read_all_devices(scan_raw);
    fix_cpu_free(scan_raw);
    std::vector<ggml_backend_t> scan_backends;
    init_gpu_backends(scan_backends);
    read_all_devices(scan_init);
    fix_cpu_free(scan_init);
    print_device_scan(scan_raw, scan_init);
    for (auto * be : scan_backends) {
        if (be) ggml_backend_free(be);
    }

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

    // per-layer MoE predicate: hparams experts OR expert tensors in the file
    const auto is_moe_layer = [&](uint32_t il) -> bool {
        return hp.n_ff_exp(il) > 0 ||
               (il < layer_has_exps.size() && layer_has_exps[il]);
    };

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

    // ---- MoE scheme -----------------------------------------------------------
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
            fprintf(stderr, "  MoE:        yes (%u experts, top-%u)%s\n",
                    (unsigned)hp.n_expert, (unsigned)hp.n_expert_used_max(),
                    any_dense ? (" scheme: " + scheme).c_str() : "");
        } else {
            fprintf(stderr, "  MoE:        no\n");
        }
    }
    fprintf(stderr, "  SWA:        %s\n", has_swa ? "yes" : "no");
    fprintf(stderr, "  SSM:        %s\n", has_ssm ? "yes" : "no");
    fprintf(stderr, "  MTP:        %s\n", has_mtp ? "yes" : "no");
    fprintf(stderr, "  kv_type_k:  %s\n", ggml_type_name(type_k));
    fprintf(stderr, "  kv_type_v:  %s\n", ggml_type_name(type_v));

    // the token embedding ALWAYS stays on the CPU (llama-model.cpp: "there is
    // very little benefit to offloading the input layer") — it is not part of
    // what gets distributed to GPUs, so it is shown here, outside the table.
    // EXCEPTION: tied models (no separate output.weight) get a second copy as
    // the lm_head on the output device — that copy appears as the LAST ROW of
    // the table (the bytes to distribute).
    if (embed_bytes > 0) {
        fprintf(stderr, "  embed(CPU): %lld\n", (long long)embed_bytes);
    }

    // ---- table layout: mtp + swa/ssm columns are capability-driven ----------
    // col 0 is sized to fit every label that will be printed (layer numbers,
    // embed/norm/output/TOTAL, and any exotic global tensor names)

    int col0_w = 7; // at least "layer"
    for (const auto & [tname, tsize] : other_tensors) {
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
        fprintf(stderr, "%s\n", data_row(label, vals, widths).c_str());
        printed_base += tsize;                          // accumulate what was printed
        printed_mtp  += mtens;
    };

    int64_t total_rate = 0, mtp_rate = 0;
    int64_t total_state = 0, mtp_state = 0;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s\n", box("┌", "┬", "┐", widths).c_str());
    fprintf(stderr, "%s\n", header_row(hdr, widths).c_str());
    fprintf(stderr, "%s\n", box("├", "┼", "┤", widths).c_str());

    for (uint32_t il = 0; il < n_layer_all; il++) {
        int64_t weights = layer_weights[il];
        const bool is_mtp_layer = il >= (n_layer_all - n_layer_nextn);

        // base deployment cost: without MTP, nextn layers are neither loaded
        // (load_mtp=false -> TENSOR_SKIP) nor executed — their weights and KV
        // appear in the mtp columns as the ADDITIONAL cost when MTP is enabled.

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
        emit_row(label, tsize, is_moe_layer(il) ? 1 : 0, mtens, csize, mcell, state);
    }

    // globals after the last layer, in data flow order:
    // norm -> output (lm_head); any remaining non-blk tensors each get a row
    if (norm_bytes > 0) {
        emit_row("norm", norm_bytes, 0, 0, 0, 0, 0);
    }
    if (output_bytes > 0) {
        emit_row("output", output_bytes, 0, 0, 0, 0, 0);
    }
    for (const auto & [tname, tsize] : other_tensors) {
        emit_row(tname.c_str(), tsize, 0, 0, 0, 0, 0);
    }
    // tied lm_head: without a separate output tensor the loader duplicates the
    // token embedding as an OUTPUT tensor (llama-model-loader.cpp:1174) — that
    // GPU copy is part of what must be distributed, hence this last row.
    if (output_bytes == 0 && embed_bytes > 0) {
        emit_row("embed", embed_bytes, 0, 0, 0, 0, 0);
    }

    // ---- totals: exactly the sum of the rows printed above ------------------
    // the moe column total = count of base-model MoE layers (mtp excluded,
    // matching how the mtp columns are an ADDITIONAL cost)
    int64_t total_moe = 0;
    for (uint32_t il = 0; il < n_layer_all - n_layer_nextn; ++il) {
        if (is_moe_layer(il)) ++total_moe;
    }
    fprintf(stderr, "%s\n", box("├", "┼", "┤", widths).c_str());
    {
        std::vector<int64_t> vals;
        vals.push_back(0);                          // label slot (unused)
        vals.push_back(printed_base);               // sum of visible rows
        if (has_moe) vals.push_back(total_moe);     // moe layer count
        if (has_mtp) vals.push_back(printed_mtp);
        vals.push_back(total_rate);
        if (has_mtp) vals.push_back(mtp_rate);
        if (has_swa || has_ssm) vals.push_back(total_state + mtp_state);
        fprintf(stderr, "%s\n", data_row("TOTAL", vals, widths).c_str());
    }
    fprintf(stderr, "%s\n", box("└", "┴", "┘", widths).c_str());

    llama_model_free(model);
    gguf_free(ctx);
    llama_backend_free();
    return 0;
}
