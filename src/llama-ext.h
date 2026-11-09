#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#ifdef __cplusplus
#  include <cstdint>
#  include <map>
#  include <vector>
#else
#  include <stdint.h>
#endif

#ifdef __cplusplus
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
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

#endif // __cplusplus
// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
#ifdef __cplusplus
extern "C" {
#endif
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);
#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

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

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);
#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

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
#endif // __cplusplus

//
// KV pool fit (fork): compute the maximum unified KV cache pool that fits the devices' VRAM
// given the model's current layer placement. Pure computation: no allocations, no GPU memory
// touched. Call after model load and before context creation, with the exact KV types the
// context will use; result is only valid for that combination.
//

#ifdef __cplusplus
extern "C" {
#endif

// per-device contribution of one KV cell (exact, from the real allocation logic)
struct llama_kv_fit_dev {
    int32_t  dev_index;      // ggml device index (-1 = host)
    int32_t  has_kv_layers;  // 1 if this device hosts KV cache layers
    uint64_t total_mem;      // device memory, bytes
    uint64_t used_bytes;     // measured at fit time (weights, other processes, ...)
    uint64_t fixed_bytes;    // state not scaling with pool size (e.g. recurrent)
    uint64_t bytes_per_cell; // KV row bytes per cell for this device's layers
    uint32_t max_cells;      // pool size allowed by this device (padded to 256)
};

struct llama_kv_fit_data {
    uint32_t n_cells;             // maximum pool: min over devices, padded to 256
    uint32_t n_dev;               // number of entries in dev[]
    size_t   total_bytes;         // fixed_bytes + bytes_per_cell * n_cells (all devices)
    size_t   fixed_bytes;         // sum of device-independent fixed terms
    size_t   bytes_per_cell;      // linear coefficient, all devices
    struct llama_kv_fit_dev * dev;// array of n_dev entries (caller-allocated, see contract)
};

// out is a caller-allocated struct with dev pre-allocated to at most llama_max_devices()
// entries; on return out->n_dev is the number of filled entries.
// flash_attn and type_k/type_v must match what the context will actually use - they change
// the KV tensor layout. swa_full is taken from the context default (false).
// n_seq_max and n_rs_seq must match the context's real values: hybrid/SSM models allocate
// a recurrent-state cache sized by mem_size*(1+n_rs_seq) where mem_size=n_seq_max, which is
// folded into the fixed term - probing with wrong values over-estimates the pool.
// Returns 1 on success, 0 if no fit is possible (e.g. nothing offloaded).
LLAMA_API int32_t llama_model_fit_kv_cache(
        const struct llama_model * model,
        enum ggml_type type_k, enum ggml_type type_v,
        int32_t kv_unified,       // must be 1 for pool math (unified mode)
        int32_t flash_attn,       // context flash-attn setting
        double  vram_fraction,    // VRAM utilization cap, e.g. 0.97
        uint32_t n_seq_max,       // context sequence count
        uint32_t n_rs_seq,        // recurrent-state rollback snapshots
        struct llama_kv_fit_data * out);

// ggml device index of layer il's tensors (-1 = host). Requires a loaded model.
LLAMA_API int32_t llama_model_layer_device(const struct llama_model * model, int32_t il);

#ifdef __cplusplus
} // extern "C"
#endif
