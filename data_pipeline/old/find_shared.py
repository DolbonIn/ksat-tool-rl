"""Find shared-stem group headers ('[13～14] …', '13번과 14번', '물음에 답하시오') in the papers' text layers; print the
header text and its position (page, column, y) so the preamble can be attached to the items it governs."""
import json, re, sys
import fitz

PAT = re.compile(r"\[\s*(\d{1,2})\s*[～~∼\-]\s*(\d{1,2})\s*\]|(\d{1,2})\s*번\s*과\s*(\d{1,2})\s*번|물음에\s*답하시오")
for j in json.load(open(sys.argv[1])):
    try:
        doc = fitz.open(j["pdf"])
    except Exception as e:
        print(j["key"], "ERR", e); continue
    for i, p in enumerate(doc):
        W = p.rect.width
        for b in p.get_text("blocks"):
            t = re.sub(r"\s+", " ", b[4])
            if PAT.search(t):
                print(j["key"], f"p{i + 1}", "col", 0 if b[0] < W / 2 else 1, "y", round(b[1]), "|", t[:160])
