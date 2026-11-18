// Host check for llama-hexagon-fa-gqa-row-split.patch: replays the HVX flash-attn decode row partition
// (op_flash_attn_ext -> qrows_per_hvx) and the worker's G_local grouping loop (flash_attn_ext_f16_thread)
// for the old and the new split. Verifies every query row is processed exactly once and counts how many
// times a KV head's K/V blocks are streamed from DDR (one stream per G_local group).
#include <stdio.h>
#include <string.h>

#define FA_HVX_G_MAX 8

static int simulate(int n_head, int G, int n_hvx, int aligned, int * max_streams_per_thread, int * ok) {
    const int qrows = n_head, qrow_start = 0;            // decode: neq1 = 1, neq3 = 1, single device
    int per = (qrows + n_hvx - 1) / n_hvx;
    if (aligned && G > 1 && G <= FA_HVX_G_MAX && qrow_start % G == 0 && qrows % G == 0) {
        const int n_groups = qrows / G;
        per = ((n_groups + n_hvx - 1) / n_hvx) * G;
    }
    int seen[1024]; memset(seen, 0, sizeof(seen));
    int streams = 0; *max_streams_per_thread = 0;
    for (int ith = 0; ith < n_hvx; ith++) {
        const int ir0 = qrow_start + per * ith;
        int ir1 = ir0 + per; if (ir1 > qrow_start + qrows) ir1 = qrow_start + qrows;
        int t_streams = 0;
        for (int ir = ir0; ir < ir1;) {
            const int ik2 = ir / G;                      // broadcast_rk2 = n_head / n_kv_heads = G
            int G_local = 1;
            while (ir + G_local < ir1 && G_local < FA_HVX_G_MAX && (ir + G_local) / G == ik2) G_local++;
            for (int g = 0; g < G_local; g++) seen[ir + g]++;
            streams++; t_streams++;
            ir += G_local;
        }
        if (t_streams > *max_streams_per_thread) *max_streams_per_thread = t_streams;
    }
    *ok = 1;
    for (int r = 0; r < qrows; r++) if (seen[r] != 1) *ok = 0;
    return streams;
}

int main(void) {
    const int cfg[][2] = { {48, 6}, {32, 4}, {32, 8}, {64, 8}, {28, 7}, {40, 5}, {16, 2}, {32, 1} };
    int all_ok = 1;
    printf("n_head G  n_hvx | KV-head streams old -> new (ideal) | max streams/thread old -> new\n");
    for (unsigned c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        for (int n_hvx = 1; n_hvx <= 8; n_hvx++) {
            int m0, m1, ok0, ok1;
            const int s0 = simulate(cfg[c][0], cfg[c][1], n_hvx, 0, &m0, &ok0);
            const int s1 = simulate(cfg[c][0], cfg[c][1], n_hvx, 1, &m1, &ok1);
            all_ok &= ok0 & ok1;
            if (s1 > s0 || m1 > m0) all_ok = 0;         // the new split must never be worse
            if (cfg[c][0] == 48 || s0 != s1)
                printf("%6d %2d %5d | %3d -> %3d (%2d) | %d -> %d %s\n", cfg[c][0], cfg[c][1], n_hvx, s0, s1,
                       cfg[c][0] / cfg[c][1] * ((cfg[c][1] + FA_HVX_G_MAX - 1) / FA_HVX_G_MAX), m0, m1, (ok0 && ok1) ? "" : "COVERAGE ERROR");
        }
    }
    printf(all_ok ? "ALL OK: every row exactly once, new split never streams more\n" : "FAILED\n");
    return all_ok ? 0 : 1;
}
