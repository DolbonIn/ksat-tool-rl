"""Stream compact per-step summaries from a veRL training log (stdin): one line per step / validation / error.
veRL step lines look like: 'step:12 - key:np.float64(0.93) - key:123 - ...'."""
import re, sys
KEYS = [("critic/score/mean", "reward"), ("response_length/mean", "len"), ("response_length/clip_ratio", "clip"),
        ("actor/entropy", "ent"), ("actor/grad_norm", "gnorm"), ("actor/pg_clipfrac", "pgclip"),
        ("rollout_corr/log_ppl_abs_diff", "mismatch"), ("timing_s/gen", "t_gen"), ("timing_s/update_actor", "t_upd"),
        ("timing_s/old_log_prob", "t_olp"), ("timing_s/update_weights", "t_sync"), ("timing_s/step", "t_step"),
        ("val-core/ksat_math/acc/mean@4", "VAL_acc@4")]
NUM = re.compile(r"^(?:np\.\w+\()?(-?[0-9.eE+-]+|nan|inf)\)?$")
def parse(line):
    kv = {}
    for item in line.split(" - "):
        if ":" not in item: continue
        k, v = item.rsplit(":", 1) if item.count(":") > 1 and "np." in item.rsplit(":", 1)[-1] else item.split(":", 1)
        m = NUM.match(v.strip())
        if m:
            try: kv[k.strip().split()[-1]] = float(m.group(1))
            except ValueError: pass
    return kv
for line in sys.stdin:
    m = re.search(r"step:(\d+) - ", line)
    if m:
        kv = parse(line[m.start():])
        parts = [f"step {m.group(1)}"] + [f"{n}={kv[k]:.3f}" if abs(kv[k]) < 100 else f"{n}={kv[k]:.0f}" for k, n in KEYS if k in kv]
        if len(parts) > 1: print(" ".join(parts), flush=True)
    elif re.search(r"Traceback \(most|out of memory|OutOfMemory|CUDA error|ActorDiedError|Killed", line):
        print("ERROR " + line.strip()[:300], flush=True)
