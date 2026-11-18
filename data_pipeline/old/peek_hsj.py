import sys, re, fitz
for p in sys.argv[1:]:
    t = fitz.open(p)[0].get_text()
    i = t.find("22")
    seg = t[max(0, i - 10): i + 120]
    print(p.split("/")[-1], [hex(ord(c)) if ord(c) > 0x7f and not ("가" <= c <= "힣") and c not in "①②③④⑤" else c for c in seg][:90])
