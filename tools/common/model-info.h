// shared model header + layer table for the fork's sizing tools
// (llama-gpu-split, llama-kv-fit-test, llama-kv-fit-deviation).
// ONE complete header per tool run (gpu-split label style), followed by the
// boxed per-layer table — printed ONCE, never per load.
#pragma once

#include <stdio.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

typedef struct mi_model_info * mi_model_info_t;

// deployment params printed into the unified header (NULL = omit the line).
// values are pre-formatted strings.
struct mi_header_opts {
    const char * keep_free;   // "0.03"        (gpu-split, kv-fit-test)
    const char * n_batch;     // "2048"        (kv-fit-test, deviation)
    const char * n_ubatch;    // "512"
    const char * seq_max;     // "3"
    const char * rs_seq;      // "8"
};

// Opens the GGUF metadata (no_alloc peek) for tensor names/sizes and keeps a
// pointer to the caller's model (hyperparameters). The model must outlive the
// handle. Returns NULL if the file cannot be opened.
mi_model_info_t mi_model_info_new(const char * path, const struct llama_model * model);

// The unified header: model facts (file/name/arch/layers/MoE/SWA/SSM/MTP),
// then kv_type_k/v + whatever deployment params opts provides, then embed(CPU).
// Returns 1 on success.
int mi_print_header(FILE * out, mi_model_info_t info,
                    enum ggml_type type_k, enum ggml_type type_v,
                    const struct mi_header_opts * opts);

// Boxed per-layer table (tensor size / [moe] / [mtp tensor] / cell size /
// [mtp cell] / [swa/ssm]; TOTAL row = sum of visible rows). flash_attn selects
// the V layout used for cell sizes. Returns 1 on success.
int mi_print_layer_table(FILE * out, mi_model_info_t info,
                         enum ggml_type type_k, enum ggml_type type_v,
                         bool flash_attn);

void mi_model_info_free(mi_model_info_t info);

#ifdef __cplusplus
}
#endif
