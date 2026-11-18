#!/usr/bin/env python3
"""2026 수능 수학 accuracy on a Snapdragon device through GenieX (geniex-bench --accuracy), graded exactly
like rl/scripts/eval_csat.py (same instruction text, thinking on, last \\boxed{} after </think>).

Problem text is read from the KSAT JSONL at run time and only written to a scratch directory outside the
repo (and to the device's /data/local/tmp); never commit the prompt file or the logs.

Sub-commands
  prepare  --problems JSONL --out DIR                 write DIR/prompts.txt (segments separated by '---') + DIR/meta.json
  run      --out DIR --model /data/local/tmp/... [--compute npu|gpu|cpu] [--seeds 0,1,2,3] [--ctx 32768] [--max-new 30000]
           pushes prompts.txt, runs geniex-bench once per seed over adb, pulls DIR/gen_<compute>_seed<S>.log
  grade    --out DIR [--logs 'gen_*.log']               parse the [gen ] blocks, grade, print eval_csat-style report
  cmd      --out DIR ...                                 only print the device command line (for QDC / manual runs)

Sampling: --temperature 0.6 --top-p 0.95 like eval_csat.py. geniex-bench has no top-k / min-p flags, so the
GenieX llama_cpp defaults apply (top_k=40, min_p=0.05 instead of eval_csat's top_k=20, min_p=0); keep that in
mind when comparing with the SGLang numbers. max_tokens is capped by the phone's context (default 30000 new
tokens in a 32768 context instead of 65536); truncations are reported separately.
"""
import argparse, collections, glob, json, os, re, shlex, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "rl"))
sys.path.insert(0, os.path.join(REPO, "rl", "scripts"))

DEV_ROOT = "/data/local/tmp/geniex"


def _stub_missing(*mods):
    """make_parquet / eval_csat import pandas / openai at module level; instruction() and report() need
    neither, so allow running this on a plain host python."""
    import importlib, types
    for m in mods:
        try:
            importlib.import_module(m)
        except ImportError:
            sys.modules[m] = types.SimpleNamespace(AsyncOpenAI=None)


def instruction(p):
    _stub_missing("pandas")
    from ksat_rl.make_parquet import instruction as _instr  # identical instruction text to eval_csat.py
    return _instr(p)


def cmd_prepare(a):
    os.makedirs(a.out, exist_ok=True)
    probs = [json.loads(l) for l in open(a.problems) if l.strip()]
    segs = []
    for p in probs:
        user = f"{instruction(p)}\n\n{p['text']}"
        if any(line.strip() == "---" for line in user.splitlines()):
            sys.exit(f"{p['id']}: problem text contains a '---' line, which geniex-bench uses as prompt separator")
        segs.append(user)
    with open(os.path.join(a.out, "prompts.txt"), "w") as f:
        f.write("\n---\n".join(segs) + "\n")
    meta = [{k: p[k] for k in ("id", "kind", "answer", "choices", "points", "section") if k in p} for p in probs]
    json.dump(dict(problems=a.problems, items=meta), open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    print(f"wrote {len(segs)} prompts to {a.out}/prompts.txt (do not commit)")


def device_cmd(a, seed, prompt_path):
    env = (f"cd {DEV_ROOT} && LD_LIBRARY_PATH=./lib:./lib/llama_cpp ADSP_LIBRARY_PATH=./lib/llama_cpp "
           f"GENIEX_PLUGIN_PATH=./lib GENIEX_LLAMA_CACHE_TYPE_K={a.kv} GENIEX_LLAMA_CACHE_TYPE_V={a.kv} ")
    if a.hex_devices:
        env += f"GGML_HEXAGON_DEVICES={a.hex_devices} "
    args = ["./bin/geniex-bench", "--plugin", "llama_cpp", "--device", a.compute, "-m", a.model,
            "--accuracy", "--think", "--prompt-file", prompt_path, "--no-warmup", "-r", "1",
            "-c", str(a.ctx), "-n", str(a.max_new), "--temperature", str(a.temperature), "--top-p", str(a.top_p),
            "--seed", str(seed), "--power-mode", a.power_mode, "--log", "info"]
    if a.compute == "npu" and a.device_id:
        args += ["--device-id", a.device_id]
    if a.threads:
        args += ["-t", str(a.threads)]
    return env + " ".join(shlex.quote(x) for x in args)


def adb(a, *args, **kw):
    base = ["adb"] + (["-s", a.serial] if a.serial else [])
    return subprocess.run(base + list(args), check=True, **kw)


def cmd_run(a):
    remote_prompts = f"{DEV_ROOT}/csat_prompts.txt"
    adb(a, "push", os.path.join(a.out, "prompts.txt"), remote_prompts)
    try:
        for seed in [int(s) for s in a.seeds.split(",")]:
            log = os.path.join(a.out, f"gen_{a.compute}_seed{seed}.log")
            print(f"[seed {seed}] -> {log}", flush=True)
            with open(log, "w") as f:
                adb(a, "shell", device_cmd(a, seed, remote_prompts), stdout=f, stderr=subprocess.STDOUT)
    finally:
        # the prompt file holds problem text: do not leave it on the device
        adb(a, "shell", "rm", "-f", remote_prompts)


def cmd_print(a):
    for seed in [int(s) for s in a.seeds.split(",")]:
        print(f"adb shell {shlex.quote(device_cmd(a, seed, DEV_ROOT + '/csat_prompts.txt'))} > gen_{a.compute}_seed{seed}.log")


def parse_log(path, n_items):
    """-> list of (stop_reason, text) per prompt, in prompt order."""
    out, cur, stop, idx = {}, None, None, 0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"^\[sep \] prompt (\d+)/(\d+)", line)
        if m:
            if cur is not None:
                out[idx] = (stop, "\n".join(cur))
            idx, cur, stop = int(m.group(1)) - 1, None, None
            continue
        if line.startswith("[info] stop_reason="):
            stop = line.split("=", 1)[1]
            cur = []
            continue
        if line.startswith("[gen ] ") or line == "[gen ]":
            cur = [] if cur is None else cur
            cur.append(line[7:])
    if cur is not None:
        out[idx] = (stop, "\n".join(cur))
    return [out.get(i, ("missing", "")) for i in range(n_items)]


def cmd_grade(a):
    _stub_missing("pandas", "openai")
    from ksat_rl.reward import grade, split_response
    from eval_csat import report  # same report as the SGLang eval
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    items = meta["items"]
    probs = {p["id"]: p for p in map(json.loads, open(meta["problems"]))}
    logs = sorted(glob.glob(os.path.join(a.out, a.logs)))
    if not logs:
        sys.exit("no logs")
    res = []
    for k, path in enumerate(logs):
        for it, (stop, text) in zip(items, parse_log(path, len(items))):
            closed, content = split_response(text + "")
            pred, ok = grade(it["kind"], int(it["answer"]), it.get("choices"), content) if closed else (None, False)
            finish = {"stop": "stop", "eos": "stop", "length": "length"}.get(str(stop), str(stop))
            res.append(dict(id=it["id"], sample=k, pred=pred, correct=ok, tokens=0, finish=finish, log=os.path.basename(path)))
    problems = [probs[it["id"]] for it in items]
    summary = report(problems, res)
    by_stop = collections.Counter(r["finish"] for r in res)
    print("stop reasons:", dict(by_stop), "| logs:", [os.path.basename(p) for p in logs])
    json.dump(dict(summary=summary, results=[{k: v for k, v in r.items()} for r in res]),
              open(os.path.join(a.out, "grade.json"), "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--problems", default="/raid/cache/b300-07/ksat-rl/data/ksat_math/test_2026.jsonl"); p.add_argument("--out", required=True)
    for name in ("run", "cmd"):
        r = sub.add_parser(name)
        r.add_argument("--out", required=True)
        r.add_argument("--model", required=True, help="GGUF path on the device")
        r.add_argument("--compute", default="npu", choices=["npu", "gpu", "cpu", "hybrid"])
        r.add_argument("--device-id", default=None, help="e.g. HTP0,HTP1,HTP2 for a 3-session layer split (with --hex-devices 3)")
        r.add_argument("--hex-devices", default=None, help="GGML_HEXAGON_DEVICES, e.g. 3 (virtual sessions HTP0..HTP2 on one NPU)")
        r.add_argument("--kv", default="q8_0", help="KV cache type (GENIEX_LLAMA_CACHE_TYPE_K/V, needs the GenieX patch)")
        r.add_argument("--seeds", default="0,1,2,3")
        r.add_argument("--ctx", type=int, default=32768)
        r.add_argument("--max-new", type=int, default=30000)
        r.add_argument("--temperature", type=float, default=0.6)
        r.add_argument("--top-p", type=float, default=0.95)
        r.add_argument("--threads", type=int, default=0)
        r.add_argument("--power-mode", default="sustained_high_performance")
        r.add_argument("--serial", default=None)
    g = sub.add_parser("grade"); g.add_argument("--out", required=True); g.add_argument("--logs", default="gen_*.log")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "cmd": cmd_print, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    main()
