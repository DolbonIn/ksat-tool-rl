#!/usr/bin/env bash
# KSAT 국어/영어/과탐/사탐 dataset pipeline (order of steps as actually run). CPU-only containers (no GPUs).
# Data never goes into the repo: everything lives under $DATA (= /raid/cache/b300-07/ksat-rl/data):
#   dl2/ (downloads, untrusted)  render2/ (segments + crops)  ocr2/ (transcriptions)  work2/ (keys, logs, reviews)
#   ksat_multi/ (final JSONL + units_2026.json)
# DeepSeek V4.1 Flash via opengateway (.env), <= 48 concurrent requests in total. Ultrafast+eco were alternated until a
# mid-run instruction; from then on gw.py is eco-only. The model of every OCR attempt / solve sample is recorded in verify.
set -euo pipefail
M=$(cd "$(dirname "$0")" && pwd)
D=$M/dk.sh                      # docker run --rm --name ksat-data2-<name> ... ksat-data-tools:local
W=/data/work2
step=${1:-all}
run() { [[ $step == all || $step == "$1" ]]; }

# 1. Official sources. KICE boards (수능 + 6/9월 모평, 2022-2026학년도); zips extracted (pdf only, cp949 names).
run fetch && $D fetch --net python -I /code/fetch_kice_multi.py --out /data/dl2/kice --years 2022-2026 \
          && $D unzip python -I /code/unzip_safe.py /data/dl2/kice \
          && $D jobs python -I /code/make_jobs_kice.py --dl /data/dl2/kice --out $W/jobs_kice.json
# 교육청 학평 via EBSi listing API (17 sittings; 2024-2026 4월 not listed on EBSi). Keys: 해설 PDF text layer + 정답 PNG reads
# (parse_keys_edu.py --manual keys_manual_edu.json for 3 disputed cells). Same OCR/verify/solve steps with jobs_edu.json.
run fetch_edu && $D fetchedu --net python -I /code/fetch_edu_multi.py --out /data/dl2/edu --years 2022-2026 \
              && $D jobsedu python -I /code/make_jobs_edu.py --dl /data/dl2/edu --out $W/jobs_edu.json \
              && $D seg python -I /code/segment.py --jobs $W/jobs_edu.json --out /data/render2 \
              && $D edukeys --net python -I /code/parse_keys_edu.py --jobs $W/jobs_edu.json --render /data/render2 \
                   --out $W/answer_keys_edu.json --manual $W/keys_manual_edu.json

# 2. Segmentation: passage ([n～m]) / question (n.) segments from the text layer, padded crops, text-layer drafts
#    (PUA equation digits mapped, underlines from vector strokes -> <u>, margin brackets -> [A]…[/A]).
run segment && $D seg python -I /code/segment.py --jobs $W/jobs_kice.json --out /data/render2

# 3. Official answer keys: 정답표 text layer; 2025 수능 keys have no text layer -> page image read by both DeepSeek
#    models + an independent manual read (keys_manual_2025.json); accepted only if all reads agree and points sum.
run keys && $D keys python -I /code/parse_keys_multi.py --jobs $W/jobs_kice.json --out $W/answer_keys_kice.json --png-dir $W/keypng \
         && $D keyread --net python -I /code/read_key_images.py --jobs $W/jobs_kice.json --keys $W/answer_keys_kice.json \
              --manual $W/keys_manual_2025.json --raw $W/key_image_reads.json

# 4. Transcription (image + text-layer draft, no thinking, T=0) -> verify vs text layer -> attempt b (other model,
#    T=0.2) for flagged -> attempt c with explicit hints (missing chars etc.) -> verify (best attempt + Hangul autofix;
#    raster-only text accepted when two attempts agree >= 0.95 and nothing from the text layer is missing).
run ocr && $D ocr --net python -I /code/ocr_assist.py --render /data/render2 --out /data/ocr2 --jobs $W/jobs_kice.json --concurrency 26
run verify && $D verify python -I /code/verify_seg.py --render /data/render2 --ocr /data/ocr2 --jobs $W/jobs_kice.json --keys all --report $W/verify_kice.json
#   (flag lists -> ocr_assist.py --sids ... --attempt-tag b --temperature 0.2 --prefer ...eco ;
#    make_hints.py verify.json hints.json sids.txt -> ocr_assist.py --attempt-tag c --hints hints.json --temperature 0.1)

# 5. Build JSONL (test = 2026 수능; train = everything else). Overrides = visual-review fixes + manual fixes.
run build && $D build python -I /code/build_multi.py --jobs $W/jobs_kice.json --keys $W/answer_keys_kice.json \
  --render /data/render2 --ocr /data/ocr2 --out /data/ksat_multi \
  --overrides $W/ov_review_2026.json,$W/manual_overrides_2026.json \
  --verify-extra $W/log_review_2026.json,$W/verify_solve_test.json,$W/verify_solve_train.json

# 6. DeepSeek solve check (n=4, thinking, \boxed{}, 16k tokens), merged into verify. 0/4 suspects: truncated ones re-solved
#    with 32k tokens (--redo-length/--ids); figure/table suspects re-read by proofread_fig.py (thinking; accepted only if
#    the text-layer score is not worse) and re-solved; old-Hangul/symbol '□' re-reads (tag h, sci/soc only);
#    suspects.py lists them.
run solve && $D solvetest --net python -I /code/solve_multi.py --in /data/ksat_multi/kor_test_2026.jsonl,... --out $W/solve/test2026_r1.jsonl --n 4
run merge && python3 $M/merge_solve_multi.py $DATA/work2/verify_solve_test.json $DATA/work2/solve/test2026_r1.jsonl

# 7. Visual review of the 2026 test set (all figure/table items + sample): make_review.py -> reviewer outputs
#    (review2026/out) -> apply_review.py -> ov_review_2026.json / log_review_2026.json
# 8. units_2026.json for experiments/csat2026_baseline/code/{solve_mc,report_mc}.py, re-score check vs reports/
run units && python3 $M/build_units_2026.py $DATA/ksat_multi
