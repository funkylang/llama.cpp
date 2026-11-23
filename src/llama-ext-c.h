/* fork: C-compatible view of the llama-ext.h fork API.
 *
 * Upstream's src/llama-ext.h is now pure C++ (staging header), so our C
 * consumers (fllama, tools/kv-fit) include this instead. Declarations are
 * duplicated from the FUNKYLANG FORK EXTENSIONS section of src/llama-ext.h —
 * keep them in sync when that section changes. */
#pragma once

#include <llama.h>

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

#define LLAMA_KV_FIT_PARAMS_VERSION 5

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
