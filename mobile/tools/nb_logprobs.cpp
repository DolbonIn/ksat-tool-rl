// nb_logprobs: teacher-forced per-token log-probs and greedy generation for a GGUF model,
// driven by explicit token ids so llama.cpp and the HF reference see identical inputs.
//
// Input file: one sequence per line: "<name> <n_prompt> <id0> <id1> ...". Positions >= n_prompt-1
// are scored (the log-prob of token t+1 given tokens 0..t); n_prompt=1 scores everything.
// Output (per sequence, in -o DIR):
//   <name>.lp.tsv        pos  target  logp(target)  argmax  logp(argmax)
//   <name>.full.f32      (with --full) raw float32 log-softmax rows [n_scored, n_vocab]
//   <name>.gen.txt       (with --gen N) greedy continuation ids after the full sequence, stopping at EOG
//   <name>.kl.tsv        (with --ref DIR) pos  KL(ref||this)  logp_ref(target)  logp(target)  ref_argmax==argmax
//                        using DIR/<name>.full.f32 written by an earlier --full run (e.g. the F16 model)
//
// Build: see mobile/scripts/build_x86_tools.sh
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static ggml_type parse_type(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    if (s == "iq4_nl") return GGML_TYPE_IQ4_NL;
    fprintf(stderr, "unknown cache type %s\n", s.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_dir = ".";
    std::string ctk = "f16", ctv = "f16";
    int  n_threads = 32, n_gen = 0, n_ubatch = 512, n_ctx = 0;
    bool full = false;
    std::string ref_dir;
    int  fa = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")      model_path = next();
        else if (a == "-i")      in_path    = next();
        else if (a == "-o")      out_dir    = next();
        else if (a == "-t")      n_threads  = std::stoi(next());
        else if (a == "-ctk")    ctk        = next();
        else if (a == "-ctv")    ctv        = next();
        else if (a == "-ub")     n_ubatch   = std::stoi(next());
        else if (a == "-c")      n_ctx      = std::stoi(next());
        else if (a == "-fa")     { std::string v = next(); fa = v == "on" ? 1 : v == "off" ? 0 : -1; }
        else if (a == "--full")  full       = true;
        else if (a == "--gen")   n_gen      = std::stoi(next());
        else if (a == "--ref")   ref_dir    = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || in_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i seqs.txt [-o dir] [-t N] [-ctk T] [-ctv T] [-fa on|off|auto] [--full] [--gen N]\n", argv[0]);
        return 1;
    }

    struct seq { std::string name; int n_prompt; std::vector<llama_token> ids; };
    std::vector<seq> seqs;
    size_t max_len = 0;
    {
        std::ifstream f(in_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            seq s;
            ss >> s.name >> s.n_prompt;
            long v;
            while (ss >> v) s.ids.push_back((llama_token) v);
            max_len = std::max(max_len, s.ids.size() + (size_t) n_gen);
            seqs.push_back(std::move(s));
        }
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx > 0 ? n_ctx : (uint32_t) ((max_len + 255) / 256 * 256);
    cparams.n_batch         = cparams.n_ctx;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k          = parse_type(ctk);
    cparams.type_v          = parse_type(ctv);
    cparams.flash_attn_type = (llama_flash_attn_type) fa;
    cparams.no_perf         = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    std::vector<double> row(n_vocab);
    for (const auto & s : seqs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int n = (int) s.ids.size();
        llama_batch batch = llama_batch_init(n, 0, 1);
        const int first_scored = std::max(0, s.n_prompt - 1);
        for (int i = 0; i < n; i++) {
            batch.token[i]     = s.ids[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i >= first_scored) || (i == n - 1);
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed for %s\n", s.name.c_str()); return 1; }

        FILE * flp   = fopen((out_dir + "/" + s.name + ".lp.tsv").c_str(), "w");
        FILE * ffull = full ? fopen((out_dir + "/" + s.name + ".full.f32").c_str(), "wb") : nullptr;
        FILE * fref  = ref_dir.empty() ? nullptr : fopen((ref_dir + "/" + s.name + ".full.f32").c_str(), "rb");
        FILE * fkl   = fref ? fopen((out_dir + "/" + s.name + ".kl.tsv").c_str(), "w") : nullptr;
        if (!ref_dir.empty() && !fref) { fprintf(stderr, "missing reference for %s\n", s.name.c_str()); return 1; }
        std::vector<float> frow(n_vocab), rrow(n_vocab);
        for (int i = first_scored; i < n - 1; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            double mx = -INFINITY;
            int am = 0;
            for (int v = 0; v < n_vocab; v++) if (lg[v] > mx) { mx = lg[v]; am = v; }
            double se = 0.0;
            for (int v = 0; v < n_vocab; v++) se += std::exp((double) lg[v] - mx);
            const double lse = mx + std::log(se);
            const int tgt = s.ids[i + 1];
            fprintf(flp, "%d\t%d\t%.6f\t%d\t%.6f\n", i, tgt, (double) lg[tgt] - lse, am, mx - lse);
            if (ffull || fkl) {
                for (int v = 0; v < n_vocab; v++) frow[v] = (float) ((double) lg[v] - lse);
            }
            if (ffull) fwrite(frow.data(), sizeof(float), n_vocab, ffull);
            if (fkl) {
                if (fread(rrow.data(), sizeof(float), n_vocab, fref) != (size_t) n_vocab) { fprintf(stderr, "short reference\n"); return 1; }
                double kl = 0.0; int ram = 0;
                for (int v = 0; v < n_vocab; v++) {
                    if (rrow[v] > rrow[ram]) ram = v;
                    kl += std::exp((double) rrow[v]) * ((double) rrow[v] - (double) frow[v]);
                }
                fprintf(fkl, "%d\t%.8f\t%.6f\t%.6f\t%d\n", i, kl, (double) rrow[tgt], (double) frow[tgt], ram == am ? 1 : 0);
            }
        }
        fclose(flp);
        if (ffull) fclose(ffull);
        if (fref) fclose(fref);
        if (fkl) fclose(fkl);
        llama_batch_free(batch);

        if (n_gen > 0) {
            FILE * fg = fopen((out_dir + "/" + s.name + ".gen.txt").c_str(), "w");
            llama_token tok;
            {
                const float * lg = llama_get_logits_ith(ctx, n - 1);
                int am = 0; for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[am]) am = v;
                tok = am;
            }
            int pos = n;
            for (int g = 0; g < n_gen; g++) {
                fprintf(fg, "%d\n", tok);
                if (llama_vocab_is_eog(vocab, tok)) break;
                llama_batch b1 = llama_batch_init(1, 0, 1);
                b1.token[0] = tok; b1.pos[0] = pos++; b1.n_seq_id[0] = 1; b1.seq_id[0][0] = 0; b1.logits[0] = 1; b1.n_tokens = 1;
                if (llama_decode(ctx, b1) != 0) { fprintf(stderr, "gen decode failed\n"); return 1; }
                llama_batch_free(b1);
                const float * lg = llama_get_logits_ith(ctx, 0);
                int am = 0; for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[am]) am = v;
                tok = am;
            }
            fclose(fg);
        }
        fprintf(stderr, "done %s (%d tokens)\n", s.name.c_str(), n);
    }
    llama_perf_context_print(ctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
