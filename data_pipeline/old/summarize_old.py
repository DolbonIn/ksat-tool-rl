"""Summary of a built 2017-2021 JSONL: counts per exam/year/track, figures, verification + solve-check stats."""
import collections, json, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
c = collections.Counter((r["exam"], r["school_year"], r["section"]) for r in rows)
exams = ["csat", "kice_06", "kice_09", "edu_03", "edu_04", "edu_07", "edu_10"]
print(f"rows {len(rows)}  figures {sum(r['has_figure'] for r in rows)}  mc {sum(r['kind'] == 'mc' for r in rows)}  sa {sum(r['kind'] == 'sa' for r in rows)}")
print("exam     " + " ".join(f"{y}ga {y}na" for y in range(2017, 2022)))
for e in exams:
    print(f"{e:8s} " + " ".join(f"{c[(e, y, 'ga')]:6d} {c[(e, y, 'na')]:6d}" for y in range(2017, 2022)))
v = collections.Counter()
acc = collections.Counter()
for r in rows:
    x = r["verify"]
    v["number_match"] += x.get("ocr_number_match", False); v["points_match"] += x.get("ocr_points_match", False)
    v["kind_consistent"] += x.get("kind_consistent", False); v["latex_errors"] += bool(x.get("latex_errors"))
    v["truncated"] += bool(x.get("truncated")); v["hangul_autofix"] += bool(x.get("hangul_autofix"))
    tl = x.get("text_layer")
    if isinstance(tl, dict):
        v["tl_checked"] += 1; v["tl_hangul_exact"] += tl["hangul_diff"] == 0
        v["tl_digits_exact"] += not tl["digits_extra"] and not tl["digits_missing_unexplained"]
    s = x.get("solve_check_deepseek")
    if s:
        acc[s["n_correct"]] += 1
    v["repaired_after_solve"] += bool(x.get("suspect_repair", {}).get("adopted"))
    v["suspect_unresolved"] += bool(x.get("suspect_unresolved"))
print(dict(v))
print("DeepSeek n_correct distribution (of n=4):", dict(sorted(acc.items())))
