"""Second-source check of 교육청 answer keys against the EBSi 정답 및 해설 PDF text layer.

Looks for "n ans n+1 ans ..." runs in the 해설 text (answer tables) for 16-22 and 23-30, and checks whether the
VLM-read PNG answers appear as an exact run. Adds `hsj_check` notes to answer_keys_edu.json items.
Usage: python -I check_edu_keys_hsj.py --jobs jobs_edu.json --keys answer_keys_edu.json
"""
import argparse, json, re
import fitz


def runs(tokens, start, end):
    out = []
    L = end - start + 1
    for i in range(len(tokens) - 2 * L + 1):
        if all(tokens[i + 2 * k] == str(start + k) for k in range(L)):
            out.append([tokens[i + 2 * k + 1] for k in range(L)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--keys", required=True)
    a = ap.parse_args()
    keys = json.load(open(a.keys))
    C = "①②③④⑤"
    tot = {"checked": 0, "match": 0, "nosrc": 0, "mismatch": []}
    for j in json.load(open(a.jobs)):
        txt = "\n".join(pg.get_text() for p in j["hsj"] for pg in fitz.open(p))
        toks = re.findall(r"[①-⑤]|\d+", txt)
        items = keys[j["key"]]["items"]
        def tokv(e):
            v = items[e]; return C[v["answer"] - 1] if v["kind"] == "mc" else str(v["answer"])
        groups = [("common", 16, 22)] + [(s, 23, 30) for s in ("prob_stat", "calculus", "geometry")] + [(s, 23, 28) for s in ("prob_stat", "calculus", "geometry")]
        done = set()
        for sec, s0, s1 in groups:
            names = [f"{sec}-{n}" for n in range(s0, s1 + 1)]
            if all(n in done for n in names):
                continue
            mine = [tokv(n) for n in names]
            cands = runs(toks, s0, s1)
            if not cands:
                continue
            ok = mine in cands
            for n in names:
                if n in done:
                    continue
                done.add(n)
                items[n]["hsj_check"] = "match" if ok else f"no exact run among {len(cands)} candidate(s)"
                tot["checked"] += 1
                if ok:
                    tot["match"] += 1
            if not ok:
                tot["mismatch"].append((j["key"], sec, mine, cands[:3]))
        for e, v in items.items():
            if "hsj_check" not in v and not (v["section"] == "common" and v["number"] <= 15):
                v["hsj_check"] = "no parseable run in 해설 text layer"
                tot["nosrc"] += 1
            elif v["section"] == "common" and v["number"] <= 15:
                v["hsj_check"] = v.get("hsj_check", "match (circled sequence 1-15)")
    json.dump(keys, open(a.keys, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in tot.items() if k != "mismatch"}), "mismatches:")
    for m in tot["mismatch"]:
        print(" ", m)


if __name__ == "__main__":
    main()
