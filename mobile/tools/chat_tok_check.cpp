// chat_tok_check: render a user turn through the GGUF's chat template exactly like GenieX's llama_cpp
// plugin (common_chat_templates_init + common_chat_session, use_jinja, enable_thinking) and tokenize it
// the way LlamaLlm::generate() does (common_tokenize(add_special=true, parse_special=true)). Prints the
// ids so they can be diffed against the HF tokenizer (apply_chat_template, add_special_tokens=False).
//   usage: chat_tok_check model.gguf "user text" [think=1|0|raw]   (raw: tokenize the text itself, add_special=true)
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf text [think]\n", argv[0]); return 1; }
    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const bool think = argc < 4 || std::string(argv[3]) != "0";
    if (argc >= 4 && std::string(argv[3]) == "raw") {
        const auto ids = common_tokenize(llama_model_get_vocab(model), argv[2], true, true);
        printf("IDS");
        for (auto id : ids) printf(" %d", id);
        printf("\n");
        llama_model_free(model);
        return 0;
    }

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "", "", "");
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    common_chat_msg m; m.role = "user"; m.content = argv[2];
    inputs.messages = { m };
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = think;
    common_chat_session session(tmpls.get(), llama_model_get_vocab(model), inputs);
    common_params_sampling sampling;
    session.apply_sampling(sampling);
    const std::string prompt = session.prompt().str();

    const auto ids = common_tokenize(llama_model_get_vocab(model), prompt, true, true);
    printf("PROMPT_BYTES %zu\nIDS", prompt.size());
    for (auto id : ids) printf(" %d", id);
    printf("\n");
    llama_model_free(model);
    return 0;
}
