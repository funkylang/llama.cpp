#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

LLAMA_API std::vector<ggml_tensor *> llama_memory_get_tensors(const llama_memory_t mem);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Questions packed after a shared state in one sequence, contexts without memory only
// 0 (default) is the state, k > 0 the k-th question; full attention layers hide other questions from a question
LLAMA_API bool llama_batch_ext_set_decision_group(struct llama_batch_ext * batch, int32_t idx, int32_t group);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// model/context data extraction
//

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);

//
// ===================== FUNKYLANG FORK EXTENSIONS =====================
// Append-only fork additions. Kept in one place so upstream rewrites of this
// header cannot scatter or break them across merges.
//

// NextN hidden-state extraction (fllama MTP catch-up, speculative drivers)
#ifdef __cplusplus
extern "C" {
#endif
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);
#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// other context of a draft MTP context (0 = none)
LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Unified KV pool planning (fork) — the gpu-split algorithm.
//
// Solves the optimal tensor split and the maximum unified KV cache pool size
// (llama_context_params.n_kv_cache) against the live devices' VRAM. The
// application supplies its deployment parameters; the library solves.
// Measurement-based: one real model load with a warmup decode materializes
// kernels/shaders; per-device readback plus the solver derive the plan —
// what is measured is exactly what the deployment will pay.
//
// ABI discipline: the options struct is allocated and initialized by the
// LIBRARY — a caller's header may be stale, so never trust your own sizeof().
// Existing field positions are IMMUTABLE: future versions append fields only;
// llama_kv_fit_params_new() refuses to run when this library is older than
// the version the caller requires.
//

#define LLAMA_KV_FIT_PARAMS_VERSION 6

// llama_kv_split() behavior options (bitmask): set by the caller from its own
// command-line options; all off by default.
#define LLAMA_KV_FIT_DEBUG          (1u << 0)   // planning internals (breakdown readbacks, rates, formulas)
#define LLAMA_KV_FIT_VERIFY         (1u << 1)   // post-solve verification load (expensive; kv-split tool)
#define LLAMA_KV_FIT_PRINT_TABLES   (1u << 2)   // print the deployment report: model header, per-layer table,
                                                // cost breakdown and the plan placement block

struct llama_kv_fit_params {
    double vram_fraction;                 // hard cap on device usage (default 0.97)
    const size_t * margins;               // per-device bytes to keep free; NULL = none
    const struct llama_kv_fit_extra * extra;   // e.g. MTP draft context; NULL = none
    uint32_t min_pool_cells;              // viability floor, reported below it (default 32768)
    // v2: recurrent-state rows per recurrent layer the deployment allocates —
    // the SSM "checkpoints" of gpu-split. The application alone knows its own
    // checkpoint policy (parallelism x rollback rows), so it passes the final
    // count; the planner uses it verbatim, never re-derives it. 0 = legacy:
    // derive n_seq_max * (1 + n_rs_seq) from the context params.
    uint32_t n_checkpoints;
    // v5: behavior option bitmask (see LLAMA_KV_FIT_* flags above). Replaces
    // the v3 verify / v4 debug booleans; same struct size as before.
    uint32_t options;
    // FUTURE FIELDS: append here; initialize in llama_kv_fit_params_new()

    // v6: deployment-report stream. The model header, per-layer table and cost-
    // breakdown tables are written here INSTEAD of stderr so the caller owns the
    // whole report as one unit (the kv-fit tool collects it in memory and emits
    // it as a single atomic write). Planning LOG messages still flow through
    // llama_log_set() — set this to funnel tables into the same stream. NULL =
    // tables print straight to stderr (default, back-compat). Caller-owned; the
    // library only writes to it and never closes or frees it.
    FILE * report_stream;
};

// Allocates + initializes the options struct with library defaults. Terminates
// the program when this library implements a version older than min_version —
// running on a mismatched libllama is a deployment error, never a fallback.
LLAMA_API struct llama_kv_fit_params * llama_kv_fit_params_new(uint32_t min_version);
LLAMA_API void llama_kv_fit_params_free(struct llama_kv_fit_params * p);

// an extra model + context that competes with the main one for the same devices
//   - its context follows the main one, so its pool size mirrors the main pool
//   - shares_model tells the fit the weights are already counted in the main
//     model, as for an MTP draft context running on the target model's weights
struct llama_kv_fit_extra {
    const char * path_model;
    struct llama_model_params * mparams;
    struct llama_context_params * cparams;
    bool shares_model;
};

// Unified KV split planning: ONE call computes the layer split AND decides the
// KV pool size from a single measurement run (one model load, probe contexts at
// 16K and 32K). The split is computed independently of any supplied cache size;
// target_cells only caps what is allocated — the resource is never claimed twice.
//   - target_cells = 0  : allocate everything that fits (auto-fit)
//   - target_cells > 0  : allocate exactly that many; when the request does not
//                         fit, result->ok == false and error_msg states what would
//                         fit — there is no fallback to a smaller or CPU deployment.
// The planned placement is returned in result->layer_bounds (per-device first-
// layer boundaries, library-owned); it is NOT applied to mparams automatically —
// the caller applies it when the plan's split is to be deployed:
//     mparams->n_gpu_layers       = -1;
//     mparams->tensor_layer_bounds = result->layer_bounds;
//     mparams->tensor_split       = nullptr;   // the boundaries supersede ratio logic
// On CPU-only deployments there is no placement (layer_bounds == NULL).
// Returns a pointer to the library-owned result; there are never two concurrent
// calls. result->ok == false => nothing may be deployed, see error_msg.
struct llama_kv_fit_result {
    bool ok;                        // true: plan computed, placement applied to mparams
    int32_t n_cells;                // the decided pool size in cells (256-aligned)
    uint32_t n_devices_planned;     // devices in the placement (0 on CPU-only deployments)
    const int32_t * layer_bounds;   // per-device first-layer boundaries (library-owned)
    char error_msg[512];            // why the deployment is not viable (when !ok)
};

LLAMA_API const struct llama_kv_fit_result * llama_kv_split(
        const char * path_model,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        const struct llama_kv_fit_params * params,
        int32_t target_cells);

LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

// [FORK] total transformer layers including the nextn (MTP) block.
LLAMA_API uint32_t llama_model_n_layer_all(const struct llama_model * model);

// [FORK] physical device index (0..llama_model_n_devices()-1) holding layer il;
// pass il == n_layer_all for the output pseudo-layer. Returns -1 when the layer
// sits on CPU or il is out of range. Verifies a planned load's placement.
LLAMA_API int32_t llama_model_layer_device_index(const struct llama_model * model, int il);

// [FORK] generic boxed table for the sizing tools (model-info layer table,
// kv-fit cost breakdown). Thin box-drawing characters only; key column
// (row_labels) and data columns right-aligned, header centered.
//   labels      : n_col column headers (may be NULL -> empty header cells)
//   row_labels  : n_row row keys, right-aligned in the first column (may be NULL)
//   key_header  : header cell of the key column; NULL/"" = empty
//   data        : n_row x n_col numbers (may be NULL -> empty cells)
//   cell_widths : one width per column (key column first); NULL = auto-measure
LLAMA_API void print_table(FILE * out,
        int n_row, int n_col,
        const char ** labels,
        const char ** row_labels,
        const char * key_header,
        const long ** data,
        const int * cell_widths,
        bool totals_separated);


#ifdef __cplusplus
} // extern "C"
#endif
// ================== END FUNKYLANG FORK EXTENSIONS =====================
