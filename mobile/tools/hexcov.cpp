// hexcov: host-side (x86) static op-coverage / fusion simulator for the ggml Hexagon (HTP) backend.
//
// The Hexagon backend decides everything on the host: supports_op() is
//   alloc_size(op and srcs) <= max_vmem  &&  htp_kparams_check(sess, op)
// and graph_compute() runs graph_optimize (matmul stacking), htp_fusion_tag_graph() and
// htp_fusion_try_fuse() on the host before anything is sent to the DSP. All of that lives in
// htp-kparams.cpp / htp-fusion.cpp / htp-opnode.cpp / htp-repack.cpp, which only see the session
// through five accessors (name, VTCM size, #HVX, #HMX, mdev count). This tool compiles those
// files unchanged on x86 with a stub session and feeds them the *real* Nanbeige graph that
// llama.cpp builds (captured with the scheduler eval callback on a CPU build), so op coverage,
// kernel selection (HMX/HVX), VTCM budgets and fusion decisions are exactly the ones the device
// would make for the same session parameters. Nothing here talks to a DSP.
//
// What is approximated (stated in the report): the scheduler's backend assignment (an op runs on
// HTP iff supports_op; the token_embd get_rows stays on CPU like llama.cpp's input layer), and the
// opbatch tensor-fit limit (assumed to fit).
//
// usage: hexcov -m model.gguf [-ub 128] [-c 4096] [-ctk f16] [-ctv f16] [-fa on|off]
//               [--nhvx 4,6] [--vtcm 8,4] [--nhmx 1] [--sweep-kv 1024,8192,32768] [-v]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "htp-common.h"
#include "htp-kparams.h"
#include "htp-fusion.h"
#include "htp-opnode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------
// stub session + backend option globals (defaults copied from ggml-hexagon.cpp at the pinned tip)

struct ggml_hexagon_session {
    std::string name = "HTP0(sim)";
    size_t      vtcm = 8u << 20;
    uint32_t    n_hvx = 4;
    uint32_t    n_hmx = 1;
    uint32_t    mdev  = 1;
};
const char * htp_session_name(const struct ggml_hexagon_session * s)      { return s->name.c_str(); }
size_t       htp_session_vtcm_size(const struct ggml_hexagon_session * s) { return s->vtcm; }
uint32_t     htp_session_n_hvx(const struct ggml_hexagon_session * s)     { return s->n_hvx; }
uint32_t     htp_session_n_hmx(const struct ggml_hexagon_session * s)     { return s->n_hmx; }
uint32_t     htp_session_mdev_count(const struct ggml_hexagon_session * s){ return s->mdev; }

int    opt_arch = 79;
size_t opt_ndev = 1;
size_t opt_nhvx = 0;
int    opt_nhmx = 1;
size_t opt_vmem = HTP_OP_MAX_VMEM_DEFAULT;
int    opt_etm = 0;
int    opt_verbose = 0;
int    opt_profile = 0;
bool   opt_hostbuf = true;
bool   opt_dma64 = false;
int    opt_power_mode = 0;
size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;
size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024;
size_t opt_mbuf_total  = 0;
int    opt_mm_select  = 2;
int    opt_fa_select  = 2;
int    opt_fa_split   = 1;
int    opt_li_select  = 2;
int    opt_gdn_select = 2;
int    opt_ar_select  = 2;
int    opt_ar_scatter = 1;
int    opt_opbatch   = 2048;
int    opt_opqueue   = 8;
int    opt_optrace   = 0;
int    opt_oppoll    = 0;
int    opt_opfusion  = 1;
int    opt_rt_repack = 1;

// ---------------------------------------------------------------------------------------------
// graph capture via the scheduler eval callback

struct capture {
    bool                       on = false;
    std::vector<ggml_tensor *> nodes;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (ask && c->on) {
        c->nodes.push_back(t);
    }
    return false; // never request data -> no syncs, graph runs normally
}

// ---------------------------------------------------------------------------------------------
// helpers

static bool is_weight(const ggml_tensor * t) {
    const ggml_tensor * b = t->view_src ? t->view_src : t;
    return b->buffer && ggml_backend_buffer_get_usage(b->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static std::string shape(const ggml_tensor * t) {
    char buf[96];
    snprintf(buf, sizeof(buf), "%s[%lld,%lld,%lld,%lld]", ggml_type_name(t->type),
             (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return buf;
}

// strip the layer index so "attn_norm-17" and "attn_norm-39" group together
static std::string base_name(const char * n) {
    std::string s(n);
    auto p = s.rfind('-');
    if (p != std::string::npos && p + 1 < s.size() && isdigit((unsigned char) s[p + 1])) s = s.substr(0, p);
    if (s.rfind("blk.", 0) == 0) { auto q = s.find('.', 4); if (q != std::string::npos) s = "blk.*" + s.substr(q); }
    return s;
}

static int layer_of(const char * n) {
    std::string s(n);
    auto p = s.rfind('-');
    if (p != std::string::npos && p + 1 < s.size() && isdigit((unsigned char) s[p + 1])) return atoi(s.c_str() + p + 1);
    return -1;
}

static const char * mm_kernel_name(int k) {
    switch (k) {
        case HTP_MM_KERNEL_HMX_2D:             return "HMX_2D";
        case HTP_MM_KERNEL_HMX_BATCHED:        return "HMX_BATCHED";
        case HTP_MM_KERNEL_HVX_QUANT_ROW:      return "HVX_QROW";
        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:    return "HVX_QBLOCK";
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:   return "HVX_F16";
        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:   return "HVX_F32";
        case HTP_MM_KERNEL_HVX_BF16_F32_VTCM:  return "HVX_BF16";
        default:                               return "?";
    }
}

static std::string kernel_info(const ggml_hexagon_session * sess, const ggml_tensor * op) {
    char buf[128] = "";
    if (op->op == GGML_OP_MUL_MAT) {
        htp_mm_kernel_params kp;
        memset(&kp, 0, sizeof(kp));
        if (htp_kparams_solve(sess, op, &kp)) {
            snprintf(buf, sizeof(buf), "%s vtcm=%dK", mm_kernel_name(kp.kernel_type), kp.vtcm_size / 1024);
        }
    } else if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        htp_fa_kernel_params kp;
        memset(&kp, 0, sizeof(kp));
        if (htp_kparams_solve(sess, op, &kp)) {
            snprintf(buf, sizeof(buf), "%s Br=%d Bc=%d vtcm=%dK", kp.kernel_type == HTP_FA_KERNEL_HMX ? "HMX" : "HVX",
                     (int) kp.Br, (int) kp.Bc, kp.vtcm_size / 1024);
        }
    }
    return buf;
}

// copy of ggml_backend_hexagon_buffer_type_get_alloc_size() for a non-host HTP buffer
static size_t htp_alloc_size(const ggml_tensor * t) {
    if (opt_rt_repack != 0 && ggml_op_is_empty(t->op) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        size_t tiled = htp_tiled_row_size(t->type, ne0) * ne1 * t->ne[2] * t->ne[3];
        return std::max(tiled, ggml_nbytes(t));
    }
    return ggml_nbytes(t);
}

static bool sim_supports_op(const ggml_hexagon_session * sess, const ggml_tensor * op, std::string * why) {
    const size_t max_vmem = opt_vmem; // minus the opqueue shm (a few MB) on device
    if (ggml_op_is_empty(op->op) || ggml_is_empty(op)) return true;
    if (htp_alloc_size(op) > max_vmem) { if (why) *why = "dst > max_vmem"; return false; }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && htp_alloc_size(op->src[i]) > max_vmem) { if (why) *why = "src > max_vmem"; return false; }
    }
    if (htp_op_remap(op) == HTP_OP_INVALID) { if (why) *why = "no HTP opcode"; return false; }
    if (!htp_kparams_check(sess, op)) { if (why) *why = "kparams solver rejected"; return false; }
    return true;
}

// ggml_graph_view() is internal to ggml-base (not exported); same definition as ggml.c
static ggml_cgraph graph_view(ggml_cgraph * g0, int i0, int i1) {
    ggml_cgraph g = *g0;
    g.size = 0; g.n_nodes = i1 - i0; g.n_leafs = 0; g.nodes = g0->nodes + i0;
    g.grads = nullptr; g.grad_accs = nullptr; g.leafs = nullptr; g.uid = 0;
    return g;
}

// ---------------------------------------------------------------------------------------------
// replica of ggml_hexagon_graph_optimize_reorder() (static in ggml-hexagon.cpp)

struct node_info {
    ggml_tensor *                                node;
    std::vector<ggml_tensor *>                   fused;
    const ggml_tensor * dst() const { return fused.empty() ? node : fused.back(); }
};

struct mem_range { uint64_t pb, p0, p1; bool dst; };

static mem_range mr_of(const ggml_tensor * t, bool dst) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    if (t->buffer) {
        return { (uint64_t) t->buffer, (uint64_t) t->data, (uint64_t) t->data + ggml_nbytes(t), dst };
    }
    return { (uint64_t) base, 0, 1024, dst };
}

static bool mr_check(const std::vector<mem_range> & rs, const mem_range & mr) {
    for (const auto & c : rs) {
        if (mr.pb != c.pb) continue;
        if (!mr.dst && !c.dst) continue;
        if (mr.p0 < c.p1 && mr.p1 > c.p0) return false;
    }
    return true;
}

static void mr_add_node(std::vector<mem_range> & rs, const node_info & n) {
    if (ggml_op_is_empty(n.node->op)) return;
    for (int i = 0; i < GGML_MAX_SRC; i++) if (n.node->src[i]) rs.push_back(mr_of(n.node->src[i], false));
    for (auto * f : n.fused) for (int i = 0; i < GGML_MAX_SRC; i++) if (f->src[i]) rs.push_back(mr_of(f->src[i], false));
    rs.push_back(mr_of(n.dst(), true));
}

static bool mr_check_node(const std::vector<mem_range> & rs, const node_info & n) {
    if (ggml_op_is_empty(n.node->op)) return true;
    for (int i = 0; i < GGML_MAX_SRC; i++) if (n.node->src[i] && !mr_check(rs, mr_of(n.node->src[i], false))) return false;
    for (auto * f : n.fused) for (int i = 0; i < GGML_MAX_SRC; i++) if (f->src[i] && !mr_check(rs, mr_of(f->src[i], false))) return false;
    return mr_check(rs, mr_of(n.dst(), true));
}

static void graph_optimize_split(ggml_cgraph * gview) {
    const int n = gview->n_nodes;
    std::vector<node_info> nodes;
    for (int i = 0; i < n; i++) {
        node_info ni{ gview->nodes[i], {} };
        const int f = htp_fusion_max(gview, i);
        for (int k = 1; k < f; k++) ni.fused.push_back(gview->nodes[++i]);
        nodes.push_back(ni);
    }
    std::vector<int> order;
    std::vector<bool> used(nodes.size(), false);
    std::vector<mem_range> mrs;
    for (int i0 = 0; i0 < (int) nodes.size(); i0++) {
        if (used[i0]) continue;
        if (!htp_fusion_can_stack(nodes[i0].node)) { order.push_back(i0); used[i0] = true; continue; }
        const int N_FORWARD = 64;
        std::vector<int> stack{ i0 };
        mrs.clear();
        for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < (int) nodes.size(); i1++) {
            if (used[i1]) continue;
            if (htp_fusion_can_stack(nodes[i0].node, nodes[i1].node) && mr_check_node(mrs, nodes[i1])) stack.push_back(i1);
            else mr_add_node(mrs, nodes[i1]);
        }
        for (int idx : stack) { order.push_back(idx); used[idx] = true; }
    }
    int j = 0;
    for (int i : order) {
        gview->nodes[j++] = nodes[i].node;
        for (auto * f : nodes[i].fused) gview->nodes[j++] = f;
    }
}

// ---------------------------------------------------------------------------------------------

struct sim_result {
    int n_compute = 0, n_htp = 0, n_cpu = 0, n_splits = 0, n_htp_ops_after_fusion = 0, n_solve_fail = 0;
    std::map<std::string, int> cpu_ops;          // "OP name shape" -> count
    std::map<std::string, int> fused_kinds;      // fused op name -> count
    std::vector<std::string>   boundary_trace;   // fused ops around the loop boundary
    std::map<std::string, std::string> kernels;  // base op name -> kernel info
};

static sim_result simulate(const ggml_hexagon_session * sess, const std::vector<ggml_tensor *> & captured,
                           int n_layer_phys, bool verbose) {
    sim_result R;

    // reconstruct a cgraph with the same node order and use counts as the scheduler's graph
    ggml_init_params ip = { ggml_graph_overhead_custom(captured.size() * 2 + 64, false) + 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, captured.size() * 2 + 64, false);
    // ggml_build_forward_expand() sets GGML_TENSOR_FLAG_COMPUTE on every node it visits; keep the flags the
    // scheduler saw (branches de-selected by ggml_build_forward_select() have it cleared)
    std::vector<int32_t> saved_flags;
    for (auto * t : captured) saved_flags.push_back(t->flags);
    for (auto * t : captured) ggml_build_forward_expand(gf, t);
    for (size_t i = 0; i < captured.size(); i++) captured[i]->flags = saved_flags[i];

    // HTP-resident weights: every weight except the input embedding (llama.cpp keeps the input
    // layer on the CPU). Their buffers are hidden so ggml_hexagon_tensor_can_tile() sees a
    // device weight, and they are flagged TILED as after load-time / rt repack.
    std::map<ggml_tensor *, ggml_backend_buffer_t> hidden;
    std::set<ggml_tensor *> cpu_weights;
    for (int i = 0; i < gf->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_tensor * w = gf->nodes[i]->src[s];
            if (!w || !is_weight(w)) continue;
            if (strncmp(w->name, "token_embd", 10) == 0) { cpu_weights.insert(w); continue; }
            if (!hidden.count(w)) {
                hidden[w] = w->buffer;
                w->buffer = nullptr;
                w->extra  = nullptr;
                if (htp_is_tiled_weight_tensor(w)) ggml_hexagon_tensor_set_flags(w, HTP_TENSOR_TILED);
            }
        }
    }

    // backend assignment
    std::vector<int> on_htp(gf->n_nodes, 1);
    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * n = gf->nodes[i];
        // same filter as ggml_hexagon_op_is_compute(): views and branches de-selected by
        // ggml_build_forward_select() (e.g. llama.cpp's unused mixed-embedding path) are not run
        if (ggml_op_is_empty(n->op) || ggml_is_empty(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) { on_htp[i] = -1; continue; }
        R.n_compute++;
        bool cpu_src = false;
        for (int s = 0; s < GGML_MAX_SRC; s++) if (n->src[s] && cpu_weights.count(n->src[s])) cpu_src = true;
        std::string why;
        const bool supp = !cpu_src && sim_supports_op(sess, n, &why);
        on_htp[i] = supp ? 1 : 0;
        if (supp) R.n_htp++; else {
            R.n_cpu++;
            std::string key = std::string(ggml_op_desc(n)) + " " + base_name(n->name) + " " + shape(n->src[0] ? n->src[0] : n) +
                              (n->src[1] ? " x " + shape(n->src[1]) : "") + " -> " + shape(n) + "  (" + (cpu_src ? "CPU-resident weight" : why) + ")";
            R.cpu_ops[key]++;
        }
        if (supp && (n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_FLASH_ATTN_EXT)) {
            const std::string nm = n->op == GGML_OP_FLASH_ATTN_EXT ? std::string("flash_attn") : base_name(n->name);
            R.kernels[nm + " " + shape(n->src[0]) + " x " + shape(n->src[1])] = kernel_info(sess, n);
        }
    }

    // splits: maximal runs of same-backend compute nodes (views inherit, they don't break runs)
    std::vector<std::pair<int, int>> htp_runs;
    {
        int cur = -1, start = 0;
        for (int i = 0; i <= gf->n_nodes; i++) {
            const int b = i < gf->n_nodes ? on_htp[i] : -2;
            if (b == -1) continue;
            if (b != cur) {
                if (cur == 1) htp_runs.push_back({ start, i });
                if (cur >= 0) R.n_splits++;
                cur = b;
                start = i;
            }
        }
    }

    // per HTP split: graph_optimize reorder, fusion tagging, kparams solve + fusion like enqueue_op()
    htp_fusion_tag_graph(gf); // tags use the global use counts, same as ggml_graph_view splits
    std::vector<htp_opnode> all_ops;
    for (auto [i0, i1] : htp_runs) {
        ggml_cgraph view = graph_view(gf, i0, i1);
        graph_optimize_split(&view);
        htp_fusion_tag_graph(&view);
        std::vector<htp_opnode> batch;
        for (int i = 0; i < view.n_nodes; i++) {
            ggml_tensor * n = view.nodes[i];
            if (ggml_op_is_empty(n->op) || ggml_is_empty(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
            htp_opnode node(htp_op_remap(n), n);
            if (!htp_kparams_solve(sess, node)) {
                R.n_solve_fail++;
                fprintf(stderr, "  !! kparams_solve failed in graph_compute for %s (%s) -- would GGML_ASSERT on device\n",
                        n->name, ggml_op_desc(n));
                continue;
            }
            if (!batch.empty() && htp_fusion_try_fuse(sess, batch.back(), node)) continue;
            batch.push_back(std::move(node));
        }
        for (auto & b : batch) all_ops.push_back(std::move(b));
    }
    R.n_htp_ops_after_fusion = (int) all_ops.size();
    for (const auto & op : all_ops) {
        std::string nm = op.name();
        R.fused_kinds[nm + (op.fused.empty() ? "" : " (fused " + std::to_string(op.fused.size() + 1) + ")")]++;
        // trace the last layer of loop 1, the loop boundary, and the first layer of loop 2
        const int l = layer_of(op.node->name);
        if (l == n_layer_phys - 1 || l == n_layer_phys || (l == 2 * n_layer_phys - 1) || l == -1) {
            std::string s = "L" + std::to_string(l) + "  " + nm + "  [" + op.node->name;
            for (auto * f : op.fused) s += " + " + std::string(f->name);
            s += "]";
            R.boundary_trace.push_back(s);
        }
    }

    for (auto & [w, b] : hidden) { w->buffer = b; w->extra = nullptr; }
    for (int i = 0; i < gf->n_nodes; i++) gf->nodes[i]->extra = nullptr;
    ggml_free(gctx);
    (void) verbose;
    return R;
}

// FA support sweep over KV lengths, using a captured FA node as the template
static void fa_sweep(const ggml_hexagon_session * sess, const ggml_tensor * fa, const std::vector<int> & kvs) {
    for (int nkv : kvs) {
        ggml_tensor q = *fa->src[0], k = *fa->src[1], v = *fa->src[2], dst = *fa;
        ggml_tensor m;
        k.ne[1] = nkv; v.ne[1] = nkv;
        ggml_tensor op = dst;
        op.src[0] = &q; op.src[1] = &k; op.src[2] = &v;
        if (fa->src[3]) {
            m = *fa->src[3];
            m.ne[0] = nkv;
            m.nb[1] = m.nb[0] * m.ne[0]; m.nb[2] = m.nb[1] * m.ne[1]; m.nb[3] = m.nb[2] * m.ne[2];
            op.src[3] = &m;
        }
        std::string why;
        const bool s = sim_supports_op(sess, &op, &why);
        printf("    n_kv=%-6d q=%s k=%s : %s %s\n", nkv, shape(&q).c_str(), shape(&k).c_str(),
               s ? "HTP" : "CPU", s ? kernel_info(sess, &op).c_str() : why.c_str());
    }
}

static std::vector<int> parse_list(const std::string & s) {
    std::vector<int> v; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(atoi(x.c_str()));
    return v;
}

static ggml_type parse_type(const std::string & s) {
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "iq4_nl") return GGML_TYPE_IQ4_NL;
    fprintf(stderr, "bad type %s\n", s.c_str()); exit(1);
}

int main(int argc, char ** argv) {
    std::string model, ctk = "f16", ctv = "f16";
    int ub = 128, n_ctx = 4096, fa = 1, nhmx = 1;
    bool verbose = false;
    std::vector<int> nhvx_list{ 4 }, vtcm_list{ 8 }, sweep{ 512, 1024, 4096, 16384, 32768 };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model = nx();
        else if (a == "-ub") ub = atoi(nx().c_str());
        else if (a == "-c") n_ctx = atoi(nx().c_str());
        else if (a == "-ctk") ctk = nx();
        else if (a == "-ctv") ctv = nx();
        else if (a == "-fa") fa = nx() == "on" ? 1 : 0;
        else if (a == "--nhvx") nhvx_list = parse_list(nx());
        else if (a == "--vtcm") vtcm_list = parse_list(nx());
        else if (a == "--nhmx") nhmx = atoi(nx().c_str());
        else if (a == "--sweep-kv") sweep = parse_list(nx());
        else if (a == "-v") verbose = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false; // no CPU repack buffers: keep tensor->extra free for the HTP flags
    llama_model * mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) return 1;
    const int n_layer_all = llama_model_n_layer(mdl);

    capture cap;
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = std::max(ub, 512); cp.n_ubatch = ub;
    cp.type_k = parse_type(ctk); cp.type_v = parse_type(ctv);
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &cap;
    cp.n_threads = 16; cp.n_threads_batch = 16;
    llama_context * ctx = llama_init_from_model(mdl, cp);
    if (!ctx) return 1;

    struct session_cfg { int nhvx, vt; };
    std::vector<session_cfg> cfgs;
    for (int nhvx : nhvx_list) for (int vt : vtcm_list) cfgs.push_back({ nhvx, vt });

    // The captured tensors live in the graph context, which the next llama_decode() rebuilds, so every
    // graph is analysed right after it is captured.
    auto analyse = [&](const std::vector<ggml_tensor *> & g, const char * phase, int n_tok) {
        if (strcmp(phase, "decode") == 0) {
            std::map<std::string, std::set<const ggml_tensor *>> by_name;
            std::map<const ggml_tensor *, int> uses;
            for (auto * n : g) if (n->op == GGML_OP_MUL_MAT) { by_name[n->src[0]->name].insert(n->src[0]); uses[n->src[0]]++; }
            size_t uniq_bytes = 0, per_use_bytes = 0, uniq_tiled = 0;
            int max_uses = 0, min_uses = 1 << 30;
            for (auto & [t, u] : uses) {
                uniq_bytes += ggml_nbytes(t); uniq_tiled += htp_alloc_size(t); per_use_bytes += ggml_nbytes(t) * u;
                max_uses = std::max(max_uses, u); min_uses = std::min(min_uses, u);
            }
            int multi = 0; for (auto & [nm, st] : by_name) if (st.size() != 1) multi++;
            printf("== weight sharing (decode graph) ==\n");
            printf("  logical layers %d, matmul weight tensors %zu (unique pointers), names with >1 tensor: %d\n",
                   n_layer_all, uses.size(), multi);
            printf("  uses per weight: min %d max %d (2 = read by both loops, 1 = lm_head)\n", min_uses, max_uses);
            printf("  bytes: unique %.1f MiB (HTP tiled alloc %.1f MiB) vs. %.1f MiB if each logical layer had its own copy\n",
                   uniq_bytes / 1048576.0, uniq_tiled / 1048576.0, per_use_bytes / 1048576.0);
        }
        for (auto c : cfgs) {
            ggml_hexagon_session sess;
            sess.n_hvx = c.nhvx; sess.vtcm = (size_t) c.vt << 20; sess.n_hmx = nhmx;
            printf("\n== %s (n_tokens=%d) | session n_hvx=%d n_hmx=%d vtcm=%dMB | KV %s/%s FA %s ==\n", phase, n_tok,
                   c.nhvx, nhmx, c.vt, ctk.c_str(), ctv.c_str(), fa ? "on" : "off");
            sim_result R = simulate(&sess, g, n_layer_all / 2, verbose);
            printf("compute nodes %d, on HTP %d, on CPU %d, graph splits %d, HTP ops after fusion %d, solve failures %d\n",
                   R.n_compute, R.n_htp, R.n_cpu, R.n_splits, R.n_htp_ops_after_fusion, R.n_solve_fail);
            for (auto & [k, cnt] : R.cpu_ops) printf("   CPU x%-3d %s\n", cnt, k.c_str());
            printf("   kernels (op  weight x activation):\n");
            for (auto & [k, v] : R.kernels) printf("     %-72s %s\n", k.c_str(), v.c_str());
            printf("   HTP op mix after fusion:\n");
            for (auto & [k, cnt] : R.fused_kinds) printf("     x%-4d %s\n", cnt, k.c_str());
            if (strcmp(phase, "decode") == 0 || verbose) {
                printf("   loop-boundary trace (layer 21 = last of loop 1, loop_norm, layer 22 = first of loop 2, layer 43, head):\n");
                for (auto & s : R.boundary_trace) printf("     %s\n", s.c_str());
            }
            const ggml_tensor * fa_node = nullptr;
            for (auto * n : g) if (n->op == GGML_OP_FLASH_ATTN_EXT) { fa_node = n; break; }
            if (fa_node) { printf("   FA sweep over KV length (%s, %d query tokens):\n", phase, n_tok); fa_sweep(&sess, fa_node, sweep); }
        }
    };

    // prefill ubatch (ub tokens, output only for the last one like a normal prompt) then one decode step
    llama_batch b = llama_batch_init(ub, 0, 1);
    for (int i = 0; i < ub; i++) { b.token[i] = 1000 + (i * 7919) % 50000; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == ub - 1; }
    b.n_tokens = ub;
    cap.on = true; cap.nodes.clear();
    if (llama_decode(ctx, b)) return 1;
    cap.on = false;
    analyse(cap.nodes, "prefill", ub);

    llama_batch b1 = llama_batch_init(1, 0, 1);
    b1.token[0] = 4242; b1.pos[0] = ub; b1.n_seq_id[0] = 1; b1.seq_id[0][0] = 0; b1.logits[0] = 1; b1.n_tokens = 1;
    cap.on = true; cap.nodes.clear();
    if (llama_decode(ctx, b1)) return 1;
    cap.on = false;
    analyse(cap.nodes, "decode", 1);

    llama_free(ctx);
    llama_model_free(mdl);
    return 0;
}
