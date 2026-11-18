"""Build per-segment correction hints for a re-transcription attempt from a verify report (verify_seg.py --report).
Usage: python3 make_hints.py verify.json out_hints.json out_sids.txt [--include-ok-raster]
"""
import json, sys

v = json.load(open(sys.argv[1]))
hints = {}
for k, x in v.items():
    if x.get("ok"):
        continue
    parts = []
    if x.get("h_missing"):
        parts.append(f"[텍스트 레이어]의 다음 한글 글자들이 전사에서 빠졌거나 다른 글자로 바뀌었습니다: '{x['h_missing']}'. 해당 부분을 [텍스트 레이어]대로 빠짐없이 옮기세요.")
    if x.get("h_extra") and not x.get("raster"):
        parts.append(f"다음 한글 글자들은 [텍스트 레이어]에 없는데 전사에 들어갔습니다(그림 설명 밖): '{x['h_extra'][:80]}'. 원문에 없는 말을 넣지 말고, 그림 속 글자라면 [그림] 블록 안에만 적으세요.")
    if x.get("d_missing"):
        parts.append(f"[텍스트 레이어]의 다음 숫자들이 전사에서 빠졌습니다: '{x['d_missing']}'.")
    if x.get("d_extra"):
        parts.append(f"다음 숫자들은 [텍스트 레이어]에 없는데 전사(그림 설명 밖)에 들어갔습니다: '{x['d_extra'][:60]}'. 숫자를 다시 확인하세요.")
    if x.get("l_missing"):
        parts.append(f"[텍스트 레이어]의 다음 영문자들이 전사에서 빠졌습니다: '{x['l_missing'][:80]}'.")
    for s in x.get("structure") or []:
        parts.append({"figure block unbalanced": "[그림]과 [/그림]의 짝을 맞추세요.", "table block unbalanced": "[표]와 [/표]의 짝을 맞추세요."}.get(s, f"구조 오류: {s}. 출력 형식을 정확히 지키세요."))
    if parts:
        hints[k] = " ".join(parts)
json.dump(hints, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
open(sys.argv[3], "w").write(",".join(hints))
print(len(hints), "hints")
