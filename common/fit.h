#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml.h"
#include "llama.h"

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    struct llama_model_params * mparams;
    struct llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
enum common_params_fit_status common_fit_params(
                         const char * path_model,
                 struct llama_model_params * mparams,
               struct llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   struct llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const struct common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     enum ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 struct llama_model_params * mparams,
               struct llama_context_params * cparams);

// [FORK] maximize the physical unified KV pool size for the given (already fitted) params.
//   - probes the model with no_alloc at two pool sizes to derive exact per-device KV rates
//   - the budget covers model weights, compute buffers and the extra (e.g. MTP) model memory
//   - hard cap: vram_fraction of each device's total memory (headroom for the driver),
//     further limited by measured free memory minus margins (co-located processes)
//   - returns the maximum pool size in cells (256-aligned); 0 if no pool fits at all
uint32_t common_fit_max_kv_cache(
                         const char * path_model,
               struct llama_model_params * mparams,
             struct llama_context_params * cparams,
                            size_t * margins,         // per-device margins to keep free (bytes)
      const struct common_fit_extra_model * extra,           // e.g. MTP context sharing the devices
                             double vram_fraction,    // hard cap as fraction of total memory
                     enum ggml_log_level   log_level);

// [FORK] compute optimal tensor_split proportional to VRAM and find best n_gpu_layers via binary search
//   - tensor_split is set proportionally to device VRAM (bigger GPUs get more layers)
//   - n_gpu_layers is found via 1D monotone search considering KV pool capacity
//   - returns: true if a valid split was computed, false otherwise
bool common_fit_auto_split(
                         const char * path_model,
               struct llama_model_params * mparams,
             struct llama_context_params * cparams,
                             float * tensor_split,   // writable buffer for proportional split
      const struct common_fit_extra_model * extra,           // e.g. MTP context
                     enum ggml_log_level   log_level);

#ifdef __cplusplus
} // extern "C"

#include <vector>

// [FORK] C++-only API below (uses std::vector) - not available to C consumers
void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const struct llama_model_params * mparams,
         const struct llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     enum ggml_log_level   log_level);
#endif
