"""Stack several crop PNGs vertically (scaled to a common width) for quick human review."""
import sys
from PIL import Image, ImageDraw
out, width, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
ims = []
for p in paths:
    im = Image.open(p).convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)))
    ims.append(im)
H = sum(i.height for i in ims) + 12 * len(ims)
canvas = Image.new("RGB", (width, H), "white"); y = 0
d = ImageDraw.Draw(canvas)
for im in ims:
    canvas.paste(im, (0, y)); y += im.height
    d.rectangle([0, y + 4, width, y + 7], fill="red"); y += 12
canvas.save(out)
