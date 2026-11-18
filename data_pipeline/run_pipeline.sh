#!/usr/bin/env bash
# KSAT math dataset pipeline (order of steps as actually run). Data never goes into the repo:
# everything is written under $DATA (= /raid/cache/b300-07/ksat-rl/data). GPUs 0-3 only.
set -euo pipefail
DATA=${DATA:-/raid/cache/b300-07/ksat-rl/data}
CODE=$(cd "$(dirname "$0")" && pwd)
TOOLS="docker run --rm -u $(id -u):$(id -g) -v $CODE:/code:ro -v $DATA:/data ksat-data-tools:local"
NET="docker run --rm --network host -u $(id -u):$(id -g) -v $CODE:/code:ro -v $DATA:/data ksat-data-tools:local"

step=${1:-all}
run() { [[ $step == all || $step == "$1" ]]; }

# 0. CPU tools image
run image && docker build -q -t ksat-data-tools:local -f "$CODE/docker/Dockerfile.tools" "$CODE/docker"

# 1. Official sources. KICE (수능/6·9월 모평) via suneung.re.kr boards; 교육청 학평 via EBSi listing (see
#    $DATA/dl/edu/manifest.json for the exact wdown.ebsi.co.kr URLs). cfpark00/KoreanSAT parquet -> $DATA/hf/KoreanSAT.
run fetch && $TOOLS python -I /code/fetch_kice.py --out /data/dl/kice --years 2022-2026
# (2022 수능 문제지 is a zip: extracted with python -I zipfile into dl/kice/2022_csat/unzipped/, cp949 names)
# jobs_kice.json / jobs_edu.json in $DATA/work list {key, exam, school_year, pdf|parts, answer_key|answer_pngs, hsj}.

# 2. Render pages + per-problem crops (labels from the PDF text layer)
run render && $TOOLS python -I /code/render_crops.py --jobs /data/work/jobs_kice.json --out /data/render \
           && $TOOLS python -I /code/render_crops.py --jobs /data/work/jobs_edu.json --out /data/render

# 3. Official answer keys (KICE: 정답표 text layer; 2025 수능 정답표 has no text layer -> read from the page image
#    and cross-checked with KoreanSAT; 교육청: EBSi 정답 PNG read by the VLM x3 + 해설 PDF text-layer cross-check)
run keys && $TOOLS python -I /code/parse_answer_keys.py --jobs /data/work/jobs_kice.json --out /data/work/answer_keys_kice.json

# 4. VLM server (Qwen3.5-397B-A17B-FP8, SGLang, TP4 on GPUs 0-3)
run serve && bash "$CODE/serve_vlm.sh"
run keys_edu && $NET python -I /code/read_edu_keys.py --jobs /data/work/jobs_edu.json --render /data/render --out /data/work/answer_keys_edu.json \
             && $TOOLS python -I /code/check_edu_keys_hsj.py --jobs /data/work/jobs_edu.json --keys /data/work/answer_keys_edu.json

# 5. OCR: pass A (transcribe) + pass B (proofread vs image); --reparse re-derives "final" from raw outputs
KEYS_ALL=$(python3 -c "import json;print(','.join(j['key'] for f in ['$DATA/work/jobs_kice.json','$DATA/work/jobs_edu.json'] for j in json.load(open(f))))")
run ocr && $NET python -I /code/ocr_vlm.py --render /data/render --out /data/ocr --keys "$KEYS_ALL" --passes a,b
run reparse && $TOOLS python -I /code/ocr_vlm.py --reparse --render /data/render --out /data/ocr --keys "$KEYS_ALL"

# 5b. Pick A vs proofread B by agreement with the PDF text layer; re-sample (k=6) items that still disagree
run select && $NET python -I /code/repair_ocr.py --jobs /data/work/jobs_kice.json,/data/work/jobs_edu.json --render /data/render --ocr /data/ocr --repair --k 6

# 6. Deterministic checks vs the PDF text layer (Hangul + digits), Hangul substitution auto-fix, points markers
run verify && $TOOLS python -I /code/add_points_text.py /data/render/crops/*/index.json \
  && $TOOLS python -I /code/fix_hangul.py --jobs /data/work/jobs_kice.json,/data/work/jobs_edu.json --ocr /data/ocr \
       --out-overrides /data/work/ov_hangul.json --out-log /data/work/log_hangul.json \
  && $TOOLS python -I /code/verify_text.py --jobs /data/work/jobs_kice.json,/data/work/jobs_edu.json --render /data/render \
       --ocr /data/ocr --out /data/work/verify_text.json --overrides /data/work/ov_hangul.json,/data/work/overrides_test_manual.json

# 7. Build JSONL (+ problems_2026.py). overrides_test_manual.json = hand-checked fixes for the 2026 test set.
#    verify extras: log_hangul.json (autofix log), verify_textlayer_extra.json (from verify_text.json),
#    verify_koreansat_extra.json, verify_solve.json (merge_solve.py), verify_manual.json (manual image review).
W=$DATA/work
run build && python3 "$CODE/build_dataset.py" --jobs "$W/jobs_kice.json,$W/jobs_edu.json" \
  --keys "$W/answer_keys_kice.json,$W/answer_keys_edu.json" --ocr "$DATA/ocr" --out "$DATA/ksat_math" \
  --overrides "$W/ov_hangul.json,$W/overrides_manual.json,$W/overrides_test_manual.json" \
  --verify-extra "$W/log_hangul.json,$W/verify_textlayer_extra.json,$W/verify_koreansat_extra.json,$W/verify_solve.json,$W/verify_manual.json"

# 8. Solve checks: local (Qwen3.5, non-DeepSeek sanity check) and DeepSeek via opengateway (only if the key has budget)
run solve_local && $NET python -I /code/solve_check.py --provider local --in /data/ksat_math/test_2026.jsonl --out /data/work/solve_local_test.jsonl --n 4
run solve_deepseek && docker run --rm --env-file "$CODE/../.env" -v "$CODE":/code:ro -v "$DATA":/data ksat-data-tools:local \
  python -I /code/solve_check.py --provider deepseek --in /data/ksat_math/test_2026.jsonl --out /data/work/solve_deepseek_test.jsonl --n 4
run merge && (cd "$W" && python3 "$CODE/merge_solve.py" --out verify_solve.json \
  deepseek=solve_deepseek_test.jsonl,solve_deepseek_train.jsonl deepseek_retry=solve_deepseek_retry.jsonl \
  local=solve_local_test.jsonl,solve_local_train.jsonl)
# 9. KoreanSAT diff for 수능 2022-2025
run koreansat && $TOOLS python -I /code/diff_koreansat.py --hf /data/hf/KoreanSAT --train /data/ksat_math/train.jsonl --out /data/work/verify_koreansat.json
