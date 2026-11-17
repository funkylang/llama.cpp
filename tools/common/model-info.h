// shared model header + layer table for the fork's sizing tools
// (llama-gpu-split, llama-kv-fit-test, llama-kv-fit-deviation).
// Style: aligned label block ("  file:       ...") followed by the boxed
// per-layer table. Printed ONCE per tool run — never per load.
#pragma once

#include <stdio.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

typedef struct mi_model_info * mi_model_info_t;

// Opens the GGUF metadata (no_alloc peek) for tensor names/sizes and keeps a
// pointer to the caller's model (hyperparameters). The model must outlive the
// handle. Returns NULL if the file cannot be opened.
mi_model_info_t mi_model_info_new(const char * path, const struct llama_model * model);

// Aligned label block: file/name/arch/layers/MoE/SWA/SSM/MTP/kv_type_k/
// kv_type_v/[keep_free]/embed(CPU). keep_free: pre-formatted string
// (e.g. "0.03") or NULL to omit the line. Returns 1 on success.
int mi_print_header(FILE * out, mi_model_info_t info,
                    enum ggml_type type_k, enum ggml_type type_v,
                    const char * keep_free);

// Boxed per-layer table (tensor size / [moe] / [mtp tensor] / cell size /
// [mtp cell] / [swa/ssm], TOTAL row = sum of visible rows). flash_attn selects
// the V layout used for cell sizes. Returns 1 on success.
int mi_print_layer_table(FILE * out, mi_model_info_t info,
                         enum ggml_type type_k, enum ggml_type type_v,
                         bool flash_attn);

void mi_model_info_free(mi_model_info_t info);

#ifdef __cplusplus
}
#endif
