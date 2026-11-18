import sys, fitz
d = fitz.open(sys.argv[1]); out = sys.argv[2]
for i in [int(x) for x in sys.argv[3].split(",")]:
    d[i].get_pixmap(dpi=int(sys.argv[4]) if len(sys.argv) > 4 else 80).save(f"{out}/page{i+1}.png")
print(d.page_count, d[0].rect, len(d[0].get_images()), repr(d[0].get_text()[:100]))
