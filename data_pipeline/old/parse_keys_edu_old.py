"""Official answer keys for 교육청 학평 2017-2021학년도 수학 가형/나형 (EBSi files only).

Sources: (1) 정답 및 해설 PDF text layer: the answer table before the first '해설'/'[출제의도]' (circled answers are text;
short answers are HWP equation-font private-use digits U+E034..E03D = 1..9,0; one PDF encodes ①..⑤ as U+6ABE..U+6AC2);
(2) the EBSi 정답 image (JPG/PNG, no text layer) read independently by BOTH DeepSeek models (temperature 0, no thinking).
Acceptance per item: hsj answer present and equal to both image reads -> ok; hsj missing -> both image reads must agree;
any disagreement -> error (item left without key until resolved). Points come from the '[n점]' markers in the paper's text
layer (render index points_text; for papers without a text layer from --points-json) and must sum to 100.
Usage: python -I parse_keys_edu_old.py --jobs /data/work3/jobs.json --render /data/render3 --out /data/work3/keys_edu.json
       --raw /data/work3/key_image_reads.json
"""
import argparse, asyncio, base64, json, os, re, sys
import fitz
sys.path.insert(0, "/pipe/multi")
import gw  # noqa: E402
# model policy (user instruction 2026-11-18): eco only from now on; override with GW_MODELS=a,b
gw.MODELS = os.environ.get("GW_MODELS", "deepseek/deepseek-v4.1-flash-eco").split(",")

CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
PUA = {0xE034 + i: str((i + 1) % 10) for i in range(10)}
ALT_CIRC = {0x6ABE: "①", 0x6ABF: "②", 0x6AC0: "③", 0x6AC1: "④", 0x6AC2: "⑤"}
PROMPT = ("이 이미지는 수학 영역 정답표입니다. 표의 각 행에는 '문항 번호 칸(회색 바탕, 기울임 숫자)'과 바로 오른쪽 '정답 칸'이 한 쌍으로 "
          "여러 쌍 나란히 있습니다. 문항 번호 칸의 숫자는 정답이 아닙니다. 각 문항 번호 칸 바로 오른쪽 칸의 내용이 그 문항의 정답입니다. "
          "1번부터 30번까지 모든 문항의 정답을 읽어 JSON 객체 하나로만 출력하세요. 키는 문항 번호(문자열), 값은 정답 칸에 적힌 그대로"
          "(객관식은 ①②③④⑤ 원문자, 단답형은 숫자, 비었거나 기호뿐이면 그 기호)입니다. "
          '예: {"1": "②", "2": "①", "22": "10"}. 다른 설명은 쓰지 마세요.')


def hsj_table(path):
    doc = fitz.open(path)
    txt = "\n".join(p.get_text() for p in doc)
    txt = re.sub(r"[\ue034-\ue03d]+", lambda m: " " + m.group(0).translate(PUA) + " ", txt)   # keep PUA-digit runs separate tokens
    txt = txt.translate(ALT_CIRC)
    cut = txt.find("[출제의도]")
    head = txt[:cut if cut > 0 else len(txt)]
    toks = re.findall(r"[①-⑤]|\d+|-", head)
    # longest run "1 a1 2 a2 ... 30 a30"
    best = {}
    for i in range(len(toks)):
        if toks[i] != "1":
            continue
        got, k, n = {}, i, 1
        while k + 1 < len(toks) and toks[k] == str(n):
            got[n] = toks[k + 1]; k += 2; n += 1
        if len(got) > len(best):
            best = got
    note = re.findall(r"※[^\n]*(?:\n[^\n※]*){0,3}", txt[:cut + 400])
    return best, note


def to_ans(v):
    v = str(v).strip() if v is not None else ""
    if v in CIRC:
        return CIRC[v], "mc"
    if re.fullmatch(r"\d{1,3}", v):
        return int(v), "sa"
    return None, None


async def read_img(path, model, key):
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)          # 2x upscale: small EBSi key images
    buf = io.BytesIO(); im.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = await gw.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}],
                      max_tokens=16000, temperature=0.0, thinking=True, tag=f"key:{key}", prefer=model, strict=True, retries=20)
    m = re.search(r"\{.*\}", r["content"], re.S)
    try:
        return json.loads(m.group(0)) if m else {}, r["model"]
    except Exception:
        return {}, r["model"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--raw", required=True); ap.add_argument("--points-json", default=None)
    ap.add_argument("--usage-log", default="/data/work3/usage.jsonl")
    a = ap.parse_args()
    gw.init(8, a.usage_log)
    jobs = [j for j in json.load(open(a.jobs)) if j["exam"].startswith("edu")]
    raw = json.load(open(a.raw)) if os.path.exists(a.raw) else {}
    todo = [(j, m) for j in jobs for m in gw.MODELS if f"{j['key']}|{m}" not in raw]
    outs = await asyncio.gather(*[read_img(j["answer_img"], m, j["key"]) for j, m in todo])
    for (j, m), (js, used) in zip(todo, outs):
        if js:
            raw[f"{j['key']}|{m}"] = dict(read=js, model=used)
    json.dump(raw, open(a.raw, "w"), ensure_ascii=False, indent=1)
    extp = json.load(open(a.points_json)) if a.points_json else {}
    res = {}
    for j in jobs:
        hs, note = hsj_table(j["hsj"]) if j.get("hsj") else ({}, [])
        reads = [raw.get(f"{j['key']}|{m}", {}).get("read", {}) for m in gw.MODELS]
        idx = os.path.join(a.render, "crops", j["key"], "index.json")
        pts = {}
        if os.path.exists(idx):
            pts = {c["number"]: (c["points_text"][0] if len(c["points_text"]) == 1 else None) for c in json.load(open(idx))["crops"]}
        for n, p in extp.get(j["key"], {}).items():
            pts.setdefault(int(n), p)
            if pts[int(n)] is None:
                pts[int(n)] = p
        items, errs = {}, []
        for n in range(1, 31):
            h = hs.get(n); r1, r2 = (str(r.get(str(n), "")).strip() for r in reads)
            vals = [x for x in (h, r1, r2) if x]
            ans, kind = to_ans(h if h else r1)
            status = "ok"
            if h and (h != r1 or h != r2):
                status = f"disagree hsj={h} img={r1}/{r2}"
            elif not h and (r1 != r2 or not r1):
                status = f"no hsj; img reads {r1}/{r2}"
            if ans is None:
                status = f"no valid answer (hsj={h!r} img={r1}/{r2})"
            if status != "ok":
                errs.append(f"{n}: {status}")
            if kind and (n <= 21) != (kind == "mc"):
                errs.append(f"{n}: kind {kind}")
            items[str(n)] = dict(number=n, answer=ans, kind=kind, points=pts.get(n), status=status,
                                 source=f"hsj:{os.path.basename(j['hsj']) if j.get('hsj') else None} + img:{os.path.basename(j['answer_img'])} x2 (DeepSeek eco/ultrafast)")
        tot = sum(v["points"] or 0 for v in items.values())
        if tot != 100 or any(v["points"] is None for v in items.values()):
            errs.append(f"points sum {tot}, missing {[k for k, v in items.items() if v['points'] is None]}")
        res[j["key"]] = dict(source_file=j.get("hsj"), answer_img=j["answer_img"], items=items, errors=errs, hsj_notes=note)
        print(j["key"], "hsj", len(hs), "img", [len(r) for r in reads], "OK" if not errs else errs, note[:1])
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
