"""Compare two segmentations (old render dir vs new) and list segments whose regions/draft changed.
Usage: python3 compare_segs.py <old_render> <new_render> <out_changed.txt>
"""
import glob, json, os, sys

old_d, new_d, out = sys.argv[1:4]
changed = []
for f in sorted(glob.glob(os.path.join(new_d, "*", "segments.json"))):
    key = os.path.basename(os.path.dirname(f))
    of = os.path.join(old_d, key, "segments.json")
    if not os.path.exists(of):
        continue
    new = {s["sid"]: s for s in json.load(open(f))["segments"]}
    old = {s["sid"]: s for s in json.load(open(of))["segments"]}
    if set(new) != set(old):
        print("SID SET DIFFERS", key, sorted(set(new) ^ set(old)))
    for sid, s in new.items():
        o = old.get(sid)
        if o is None:
            changed.append((key, sid, "new")); continue
        why = []
        if len(s["regions"]) != len(o["regions"]):
            why.append(f"regions {len(o['regions'])}->{len(s['regions'])}")
        else:
            for a, b in zip(s["regions"], o["regions"]):
                if a["page"] != b["page"] or a["col"] != b["col"] or abs(a["rect"][1] - b["rect"][1]) > 3 or abs(a["rect"][3] - b["rect"][3]) > 3:
                    why.append("rect"); break
        if s["draft"] != o["draft"]:
            why.append("draft")
        if why:
            changed.append((key, sid, ",".join(why)))
open(out, "w").write(",".join(f"{k}/{s}" for k, s, _ in changed))
print(len(changed), "changed segments")
from collections import Counter
print(Counter(w for _, _, w in changed).most_common(10))
for c in changed[:40]:
    print(" ", *c)
