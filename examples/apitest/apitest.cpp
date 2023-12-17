#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <signal.h>
#include <unistd.h>

enum e_model {
    MODEL_UNKNOWN,
    MODEL_1B,
    MODEL_3B,
    MODEL_7B,
    MODEL_8B,
    MODEL_13B,
    MODEL_15B,
    MODEL_30B,
    MODEL_34B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
};

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_PERSIMMON,
    LLM_ARCH_REFACT,
    LLM_ARCH_BLOOM,
    LLM_ARCH_UNKNOWN,
};

struct hparams {
    bool vocab_only;
    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    float rope_freq_base_train;
    float rope_freq_scale_train;

    float f_clamp_kqv;
    float f_max_alibi_bias;
};

struct llama_model_header {
    e_model     type;
    llm_arch    arch;
    llama_ftype ftype;

    std::string name;
    hparams params;
};

static llama_context **g_ctx;
static llama_model **g_model;
static std::vector<llama_token> *g_input_tokens;
static std::ostringstream *g_output_ss;
static std::vector<llama_token> *g_output_tokens;

static void llama_null_log_callback(
  enum ggml_log_level level, const char * text, void * user_data
) {
  (void)level;
  (void)text;
  (void)user_data;
}

static std::vector<llama_token> llama_tokenize(
  const struct llama_model *model,
  const std::string &text,
  bool add_bos,
  bool special
) {
  // upper limit for the number of tokens
  int n_tokens = text.length() + add_bos;
  std::vector<llama_token> result(n_tokens);
  n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
  if (n_tokens < 0) {
    result.resize(-n_tokens);
    int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
    GGML_ASSERT(check == -n_tokens);
  } else {
    result.resize(n_tokens);
  }
  return result;
}

int main() {
  llama_log_set(llama_null_log_callback, NULL);

  const char *model_filename =
    "/mnt/models/losslessmegacoder-llama2-7b-mini.Q5_K_M.gguf";

  // load the contents of "prompt.txt"
  char *prompt;
  FILE *fh = fopen("prompt.txt", "r");
  if (fh == NULL) {
    fprintf(stderr, "Could not open prompt.txt\n");
    return 1;
  }
  fseek(fh, 0, SEEK_END);
  long size = ftell(fh);
  rewind(fh);
  prompt = (char *)malloc(size + 1);
  fread(prompt, 1, size, fh);
  prompt[size] = '\0';
  fclose(fh);

  llama_backend_init(false);

  llama_model *model;
  llama_context *ctx;
  g_model = &model;
  g_ctx = &ctx;

  auto mparams = llama_model_default_params();
  mparams.vocab_only = true;
  model = llama_load_model_from_file(model_filename, mparams);
  int layer_count = ((struct llama_model_header *)model)->params.n_layer;
  int context_size = ((struct llama_model_header *)model)->params.n_ctx_train;
  llama_free_model(model);

  mparams.vocab_only = false;
  mparams.n_gpu_layers = layer_count+3;
  model = llama_load_model_from_file(model_filename, mparams);
  auto cparams = llama_context_default_params();
  cparams.n_ctx = context_size;
  cparams.seed = 1;
  fprintf(stderr,
    "layers: %d\n"
    "context size: %d\n"
    "batch size: %d\n"
    "threads: %d\n"
    "seed: %d\n"
    ,
    mparams.n_gpu_layers, cparams.n_ctx, cparams.n_batch, cparams.n_threads,
    cparams.seed);
  ctx = llama_new_context_with_model(model, cparams);
  if (!model) {
    fprintf(stderr, "error: unable to load model %s\n", model_filename);
    return EXIT_FAILURE;
  }

  const int n_ctx = llama_n_ctx(ctx);

  const bool add_bos = llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM;

  std::vector<llama_token> embd_inp;

  embd_inp = ::llama_tokenize(model, prompt, add_bos, true);
  fprintf(stderr, "token count: %d\n", embd_inp.size());

  int n_past = 0;
  int n_remain = 128;
  int batch_size = 512;
  int n_consumed = 0;
  int keep_n = 2000; //256;

  std::vector<int> input_tokens;
  std::vector<int> output_tokens;
  std::ostringstream output_ss;

  g_input_tokens = &input_tokens;
  g_output_tokens = &output_tokens;
  g_output_ss = &output_ss;

  std::vector<llama_token> embd;

  int vocab_size = llama_n_vocab(model);

  while (n_remain > 0) {
    // predict
    if (!embd.empty()) {
      // infinite text generation via context swapping
      // if we run out of context then
      //
      // * take the *n_keep* first tokens from the original prompt
      // * take half of the last tokens and recompute the logits in batches

      if (n_past + (int) embd.size() > n_ctx) {
	const int n_left = n_past-keep_n-1;
	int n_discard = n_left/8;
	if (n_past+(int)embd.size()-n_discard > n_ctx) {
	  n_discard = n_past+(int)embd.size()-n_ctx;
	}
	n_discard += 133;
	llama_kv_cache_seq_rm(
	  ctx, 0, keep_n+1, keep_n+n_discard+1);
	llama_kv_cache_seq_shift(
	  ctx, 0, keep_n+1+n_discard, n_past, -n_discard);
	fprintf(stderr, "shift %d ... %d << %d\n",
	  keep_n+1+n_discard, n_past, n_discard);
	n_past -= n_discard;
      }
      for (int i = 0; i < (int) embd.size(); i += batch_size) {
	int n_eval = (int) embd.size() - i;
	if (n_eval > batch_size) {
	  n_eval = batch_size;
	}
	if (
	  llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))
	) {
	  fprintf(stderr, "evaluation failed\n");
	  return EXIT_FAILURE;
	}
	n_past += n_eval;
      }
    }
    embd.clear();
    if ((int) embd_inp.size() <= n_consumed) {
	float *logits = llama_get_logits(ctx);
	int i;
	int best = 0;
	float best_value = logits[best];
	for (i = 1; i < vocab_size; ++i) {
	  if (logits[i] > best_value) {
	    best = i;
	    best_value = logits[i];
	  }
	}
	embd.push_back(best);
	--n_remain; // decrement remaining sampling budget
    } else {
      // some user input remains from prompt or interaction,
      // forward it to processing
      while ((int) embd_inp.size() > n_consumed) {
	embd.push_back(embd_inp[n_consumed]);

	++n_consumed;
	if ((int) embd.size() >= 2*batch_size) {
	    break;
	}
      }
    }

    // display text
    for (auto id : embd) {
      std::vector<char> result(8, 0);
      const int n_tokens =
	llama_token_to_piece(model, id, result.data(), result.size());
      if (n_tokens < 0) {
	result.resize(-n_tokens);
	llama_token_to_piece(model, id, result.data(), result.size());
      } else {
	result.resize(n_tokens);
      }

      const std::string token_str = std::string(result.data(), result.size());
      printf("%s", token_str.c_str());
      if (embd.size() > 1) {
	input_tokens.push_back(id);
      } else {
	output_tokens.push_back(id);
	output_ss << token_str;
      }
    }
    fflush(stdout);

    if (!embd.empty() && embd.back() == llama_token_eos(model)) {
	fprintf(stderr, " [end of text]\n");
	break;
    }
  }

  llama_free(ctx);
  llama_free_model(model);

  llama_backend_free();

  return EXIT_SUCCESS;
}
