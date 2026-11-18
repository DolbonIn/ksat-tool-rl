"""Turn visual-review outputs (review2026/out/batch_*.json) into build overrides.

Each issue {field, old, new[, choice_index]} is applied as an exact string replacement on the CURRENT built text
(question / choices[i] / passage). Issues whose `old` is not found are reported and skipped. Skipped ids can be listed
with --reject id1,id2 after manual inspection. Output: overrides JSON {item_id: {stem, choices_list}, "<key>/<sid>": {passage}}
and a review log {item_id: {"manual_review": {...}}} for build_multi --verify-extra.

Usage: python3 apply_review.py <build_dir> <review_out_dir> <overrides_out.json> <review_log_out.json> [--reject a,b] [--extra fixes.json]
"""
import glob, json, os, re, sys

B, RO, OV, LOG = sys.argv[1:5]
reject = set()
extra = []
args = sys.argv[5:]
for i, x in enumerate(args):
    if x == "--reject":
        reject = set(args[i + 1].split(","))
    if x == "--extra":
        extra = json.load(open(args[i + 1]))
recs = {}
for f in glob.glob(os.path.join(B, "*_test_2026.jsonl")):
    for l in open(f):
        r = json.loads(l); recs[r["id"]] = r
passages = {}
for r in recs.values():
    if r.get("passage_id"):
        passages.setdefault(r["passage_id"], r)
results = []
for f in sorted(glob.glob(os.path.join(RO, "batch_*.json"))):
    try:
        results += json.load(open(f))
    except Exception as e:
        print("bad review file", f, e)
results += extra
ov, log, miss = {}, {}, []
stat = {"ok": 0, "minor": 0, "error": 0}


def seg_key(r):
    area = r["area"]
    key = f"{r['school_year']}_{r['exam']}_{area}" + (f"_{r['subject']}" if area in ("sci", "soc") else "")
    m = re.search(r"-p(\d+)_(\d+)$", r["passage_id"])
    sec = r["subject"] if area != "eng" else "eng"
    return f"{key}/{sec}-p{int(m.group(1)):02d}_{int(m.group(2)):02d}"


for e in results:
    st = e.get("status", "ok"); stat[st] = stat.get(st, 0) + 1
    iid = e["id"]
    is_p = iid in passages and iid not in recs
    tgt = passages[iid] if is_p else recs.get(iid)
    if tgt is None:
        print("unknown id", iid); continue
    applied = []
    if iid in reject:
        log.setdefault(iid, {})["manual_review"] = dict(status=st, applied=[], rejected=True, issues=[i.get("problem") for i in e.get("issues", [])])
        continue
    if is_p:
        cur = dict(passage=(ov.get(seg_key(tgt), {}).get("passage") or tgt["passage"]))
    else:
        cur = dict(stem=ov.get(iid, {}).get("stem") or tgt["question"], choices=list(ov.get(iid, {}).get("choices_list") or tgt["choices"]))
    for iss in e.get("issues", []):
        old, new, fld = iss.get("old", ""), iss.get("new", ""), iss.get("field")
        if not new and not old:
            continue
        if fld == "passage" and is_p:
            if old and old in cur["passage"]:
                cur["passage"] = cur["passage"].replace(old, new, 1); applied.append(iss.get("problem"))
            else:
                miss.append((iid, fld, old[:60]))
        elif fld == "choices" and not is_p:
            ci = iss.get("choice_index")
            if ci and 1 <= int(ci) <= 5:
                c = cur["choices"][int(ci) - 1]
                if old and old in c:
                    cur["choices"][int(ci) - 1] = c.replace(old, new, 1); applied.append(iss.get("problem"))
                elif not old:
                    cur["choices"][int(ci) - 1] = new; applied.append(iss.get("problem"))
                else:
                    miss.append((iid, fld, old[:60]))
            else:
                miss.append((iid, fld + " (no index)", old[:60]))
        elif fld == "question" and not is_p:
            if old and old in cur["stem"]:
                cur["stem"] = cur["stem"].replace(old, new, 1); applied.append(iss.get("problem"))
            elif not old and new:
                miss.append((iid, "question insert without anchor", new[:60]))
            else:
                miss.append((iid, fld, old[:60]))
        else:
            miss.append((iid, f"field {fld}", old[:60]))
    if applied:
        if is_p:
            ov[seg_key(tgt)] = dict(passage=cur["passage"])
        else:
            ov[iid] = dict(stem=cur["stem"], choices_list=cur["choices"])
    rec = dict(status=st, applied=applied, issues=[i.get("problem") for i in e.get("issues", [])])
    if is_p:
        for r in recs.values():
            if r.get("passage_id") == iid:
                log.setdefault(r["id"], {})["manual_review_passage"] = rec
    else:
        log.setdefault(iid, {})["manual_review"] = rec
json.dump(ov, open(OV, "w"), ensure_ascii=False, indent=1)
json.dump(log, open(LOG, "w"), ensure_ascii=False, indent=1)
print("review status:", stat, "| overrides:", len(ov), "| unapplied issues:", len(miss))
for m in miss:
    print("  UNAPPLIED", m)
