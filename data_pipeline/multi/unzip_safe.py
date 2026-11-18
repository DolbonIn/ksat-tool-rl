"""Safely extract downloaded zips (untrusted): only .pdf members, cp949-aware names, no path traversal.
Each zip X.zip -> sibling dir X_unzipped/. Usage: python -I unzip_safe.py /data/dl2/kice"""
import os, sys, zipfile


def fix_name(info):
    n = info.filename
    if not (info.flag_bits & 0x800):
        try:
            n = n.encode("cp437").decode("cp949")
        except Exception:
            pass
    return n


for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        if not f.endswith(".zip"):
            continue
        zp = os.path.join(root, f)
        od = zp[:-4] + "_unzipped"
        os.makedirs(od, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = fix_name(info)
                base = os.path.basename(name.replace("\\", "/"))
                if not base.lower().endswith(".pdf") or info.file_size > 200 * 2 ** 20:
                    print("skip", zp, name); continue
                sub = os.path.basename(os.path.dirname(name.replace("\\", "/")))
                out = os.path.join(od, (sub + "__" if sub else "") + base)
                with z.open(info) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                print(out)
