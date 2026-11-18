"""Turn gen_tir.py raw attempts into the deliverables.

  data/tool/tir_traces.jsonl   verified trajectories: OpenAI-style messages (system, user, assistant with
                               reasoning_content + tool_calls, tool with the REAL sandbox output, ..., final
                               assistant with \\boxed{}), tools schema, per-call execution records, metadata
  data/tool/tir_traces_meta_extra.jsonl  verified, but the teacher's reasoning calls the tool unnecessary /
                               prompt-driven ("not necessary but instructions encourage"); kept apart
  data/tool/tir_rejects.jsonl  rejected attempts: id, reasons, metadata (+ messages for inspection)
  data/tool/tir_gen_summary.json  counts per source/section, calls-per-trajectory, reject reasons, token usage
Pure JSON processing (no model code is executed here).
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl"))
from ksat_rl.tools.schema import TOOL_SYSTEM_PROMPT, TOOLS  # noqa: E402
from ksat_rl.tools.trace_utils import meta_tool_commentary  # noqa: E402

META = ("id", "source", "section", "exam", "school_year", "number", "kind", "points", "family_id", "passrate",
        "affinity", "label_source", "n_calls", "answer_check", "tool_output_used", "model", "gen_prompt_version",
        "temperature", "usage", "secs")


def strong_use(r) -> bool:
    """At least one used call matched a non-trivial token: a multi-digit integer, a fraction/decimal, or a
    symbolic value (single digits like '3' can match by coincidence)."""
    for t in r["tool_runs"]:
        for tok in t.get("used_tokens") or []:
            if "/" in tok or "." in tok or not re.fullmatch(r"-?\d+", tok) or len(tok.lstrip("-")) >= 2:
                return True
    return False


def classify(r) -> tuple:
    """-> ("clean" | "meta" | "rejected", reasons).  clean = verified, strongly used output, no meta-commentary."""
    reasons = [x for x in r["reasons"] if x != "meta_tool_commentary"]
    if r["status"] not in ("verified", "rejected") or reasons:
        return "rejected", r["reasons"]
    if not strong_use(r):
        return "rejected", ["tool_output_weakly_used"]
    if meta_tool_commentary(r["messages"]):
        return "meta", ["meta_tool_commentary"]
    return "clean", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["/raid/cache/b300-07/ksat-rl/data/tool/gen/tir_raw.jsonl",
                                                  "/raid/cache/b300-07/ksat-rl/data/tool/gen/tir_raw_pass2.jsonl",
                                                  "/raid/cache/b300-07/ksat-rl/data/tool/gen/pilot_raw.jsonl"])
    ap.add_argument("--outdir", default="/raid/cache/b300-07/ksat-rl/data/tool")
    ap.add_argument("--exclude-ids", default=None, help="file with ids to drop (e.g. failed format checks)")
    a = ap.parse_args()
    attempts, by_id = [], collections.defaultdict(list)
    for f in a.raw:
        if not os.path.exists(f):
            continue
        for l in open(f):
            r = json.loads(l)
            r["_pass"] = os.path.basename(f)
            attempts.append(r)
            if r.get("gen_prompt_version") == "gen_v2":   # gen_v1 pilot: teacher-only prompt, discarded
                by_id[r["id"]].append(r)
    drop = set()
    if a.exclude_ids and os.path.exists(a.exclude_ids):
        drop = {l.strip() for l in open(a.exclude_ids) if l.strip()}
    rank = {"clean": 0, "meta": 1, "rejected": 2}
    ver, meta, rej, chosen = [], [], [], {}
    for pid, rs in by_id.items():
        best = None
        for r in rs:
            assert r["messages"][0]["content"] == TOOL_SYSTEM_PROMPT
            c, why = classify(r)
            if pid in drop and c != "rejected":
                c, why = "rejected", ["failed_format_check"]
            if best is None or rank[c] < rank[best[0]]:
                best = (c, why, r)
        c, why, r = best
        chosen[pid] = (c, r)
        rec = {k: r.get(k) for k in META}
        rec["generation_pass"] = r["_pass"]
        rec["attempts"] = len(rs)
        if c == "clean":
            ver.append({**rec, "messages": r["messages"], "tools": TOOLS, "tool_runs": r["tool_runs"]})
        elif c == "meta":
            meta.append({**rec, "meta_tool_commentary": meta_tool_commentary(r["messages"])[:5],
                         "messages": r["messages"], "tools": TOOLS, "tool_runs": r["tool_runs"]})
        else:
            rej.append({**rec, "status": r["status"], "reasons": why, "messages": r["messages"],
                        "tool_runs": r["tool_runs"]})
    # hard guard: no held-out item (2026 수능, 2026 9월 모평) or synthetic item of their families in any output
    D = "/raid/cache/b300-07/ksat-rl/data"
    held = {json.loads(l)["id"] for l in open(f"{D}/ksat_math/test_2026.jsonl")}
    for f in (f"{D}/ksat_math/train.jsonl", f"{D}/ksat_math_old/train_2017_2021_hard.jsonl"):
        held |= {p["id"] for p in map(json.loads, open(f)) if p["exam"] == "kice_09" and int(p["school_year"]) == 2026}
    for r in ver + meta + rej:
        assert r["id"] not in held and r.get("family_id") not in held, f"held-out item leaked: {r['id']}"
    for rows, name in ((ver, "tir_traces.jsonl"), (meta, "tir_traces_meta_extra.jsonl"), (rej, "tir_rejects.jsonl")):
        rows.sort(key=lambda r: (r["source"], r["section"], r["id"]))
        with open(f"{a.outdir}/{name}", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def tok(rows, k):
        return sum((r.get("usage") or {}).get(k, 0) for r in rows)

    keys = ("prompt", "completion", "reasoning", "cached", "requests")
    per_pass = {}
    for r in attempts:
        per_pass.setdefault(r["_pass"], []).append(r)
    reason_c = collections.Counter(x.split(":")[0] for r in rej for x in r["reasons"])
    attempt_reasons = collections.Counter()
    for r in attempts:
        if r.get("gen_prompt_version") == "gen_v2":
            c, why = classify(r)
            attempt_reasons[c if c != "rejected" else "rejected:" + ",".join(sorted({x.split(":")[0] for x in why}))] += 1
    summ = {
        "attempted_problems": len(by_id), "attempts_total": sum(len(v) for v in by_id.values()),
        "verified_clean": len(ver), "verified_meta_commentary_extra": len(meta), "rejected_problems": len(rej),
        "verified_by_source": dict(collections.Counter(r["source"] for r in ver)),
        "verified_by_section": dict(sorted(collections.Counter(f"{r['source']}/{r['section']}" for r in ver).items())),
        "attempted_by_section": dict(sorted(collections.Counter(f"{c[1]['source']}/{c[1]['section']}" for c in chosen.values()).items())),
        "calls_per_verified_trajectory": dict(sorted(collections.Counter(r["n_calls"] for r in ver).items())),
        "mean_calls_verified": round(statistics.mean(r["n_calls"] for r in ver), 2) if ver else 0,
        "tool_run_status_verified": dict(collections.Counter(t["status"] for r in ver for t in r["tool_runs"])),
        "verified_with_tool_error_recovery": sum(any(t["status"] != "ok" for t in r["tool_runs"]) for r in ver),
        "label_source_verified": dict(collections.Counter(r["label_source"] for r in ver)),
        "per_attempt_outcomes": dict(attempt_reasons),
        "rejected_problem_reasons": dict(reason_c),
        "tokens_by_pass": {p: {k: tok(v, k) for k in keys} for p, v in per_pass.items()},
        "tokens_total": {k: tok(attempts, k) for k in keys},
        "completion_tokens_per_verified_median": statistics.median(r["usage"].get("completion", 0) for r in ver) if ver else 0,
        "model": "deepseek/deepseek-v4.1-flash-eco",
        "notes": {"pilot_raw.jsonl": "gen_v1 (teacher-only guidance in the system prompt), excluded from outputs",
                  "meta_extra": "verified but the reasoning says the tool is unnecessary / prompt-driven"},
    }
    json.dump(summ, open(f"{a.outdir}/tir_gen_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
