# Nanbeige4.2-3B (KSAT 수학 RL) → Snapdragon 모바일 (GenieX / llama.cpp GGUF)

RL 모델 `math_v1_step40`을 qualcomm/GenieX(커밋 `0d049c1`, 고정 llama.cpp `93cf736a`)의 llama.cpp 경로로
Snapdragon 폰(NPU=Hexagon HTP, GPU=Adreno OpenCL, CPU)에서 돌리기 위한 변환·검증·패치·빌드·실기기 런북이다.
실기기 없이(x86 CPU, Docker) 할 수 있는 것은 모두 했고, 실기기가 필요한 항목은 [§9](#9-실기기가-필요한-항목)에 따로 모았다.

- 산출물(대용량, git 밖): `/raid/cache/b300-07/ksat-rl/mobile/`
  - `gguf/` 변환·양자화 GGUF, `artifacts/android-arm64/`, `artifacts/linux-arm64/` 빌드 결과
  - `eval/` 로그확률·KL·op 커버리지 결과, `work/GenieX/` 패치가 적용된 GenieX 작업 사본
- 코드·패치(이 디렉터리): `scripts/`, `tools/`, `eval/`, `bench/`, `patches/`
- 문제 원문(2026 수능)은 실행 시 `/raid`에서 읽기만 하고 이 저장소에는 절대 쓰지 않는다.

---

## 1. 결과 요약

| 항목 | 결과 |
|---|---|
| 변환 | 고정 컨버터로 F32/F16/BF16 GGUF 생성. `nanbeige.num_loops=2`, `skip_loop_final_norm=false`, 22층(물리)→44층(논리), EOS=`<\|im_end\|>`(166101), EOG={`</s>`,`<\|im_end\|>`,`<\|endoftext\|>`}, chat template(`<think>\n`) 포함 |
| 정확성 (x86 CPU) | GGUF-F16 vs HF fp32: 평균 \|Δlogp\| **0.00021**, 최대 0.028, top-1 일치 99.97%, KL 2.3e-7 (2,889 토큰). 그리디 생성 정상, `<\|im_end\|>`에서 정지 |
| 토크나이저 | **기기 경로에서 0/46 문항이 학습 때와 다른 토큰열** → 패치 + `add_bos_token=false`로 **46/46 일치** ([§4.1](#41-토크나이저-46-문항-전부-불일치--패치)) |
| 양자화 | Q4_0는 눈에 띄게 열화(KL 0.058, top-1 94.3%). 어텐션+lm_head Q8_0이면 KL 절반(0.026), imatrix(학습 세트로 보정)까지 더하면 0.018. 권장: **Q5_K-pure-o8-imat** 2.85 GiB, KL 0.0075, top-1 97.9% (Q6_K급), 대안 **Q4_0-a8o8-imat** ([§3](#3-양자화-품질)) |
| KV/메모리 | 44 논리층이라 F16 KV = 176 KiB/토큰. 2026 풀이의 95%가 32k 안에 끝남 → **ctx 32k + KV q8_0(2.9 GiB, KL 0.00009 = 무손실)**. HTP는 F16/Q8_0 KV만 지원(q4_0 KV는 CPU로 빠짐, 분할 90). NPU 세션 주소창(3.1 GiB) 때문에 3 가상세션 권장 ([§5](#5-kv-캐시메모리-예산)) |
| Op 커버리지 | 실제 Nanbeige 그래프를 Hexagon 백엔드의 호스트측 판정 코드(무수정)로 시뮬레이션: **토큰 임베딩 get_rows(설계상 CPU) 외 전부 HTP**, 분할 2개, solve 실패 0. 루프 경계에서도 융합 유지, 가중치 중복 없음 ([§6](#6-op-커버리지와-커널-최적화)) |
| 패치 | 4개(토크나이저, 루프 모델 레이어→디바이스 매핑, HTP FA GQA 행 분할, GenieX KV 타입 env) ([§7](#7-패치)) |
| 빌드 | 공식 툴체인 컨테이너로 **Android arm64 / Linux arm64 snapdragon 빌드 성공** (CPU+OpenCL+Hexagon v73/v75/v79/v81/v85 skel, QAIRT 플러그인만 서브모듈 미체크아웃으로 OFF) ([§8](#8-빌드-산출물)) |

---

## 2. 변환과 정확성 (Task 1, 2)

```bash
mobile/scripts/convert_gguf.sh math_v1_step40 f16  nanbeige42-3b-math-v1-s40   # f32 / bf16 도 동일
mobile/scripts/convert_gguf.sh Nanbeige4.2-3B f16 nanbeige42-3b-base           # 원본(비교용)
mobile/scripts/set_add_bos_false.sh /ksat/mobile/gguf/*.gguf                    # §4.1 참고
```
- RL 체크포인트 shard는 root 소유 `0600`이라 컨버터 컨테이너만 root로 돌리고 결과는 chown 한다(체크포인트는 읽기만).
- 메타데이터: `general.architecture=nanbeige`, `block_count=22`, `num_loops=2`, `skip_loop_final_norm=False`,
  `rope.freq_base=7e7`, `head_count 48 / kv 8`, `vocab 166144`, `tokenizer.ggml.model=llama(SPM)`,
  `metaspace.prepend_scheme=first`, BOS=`<|im_start|>`(166100), EOS=`<|im_end|>`(166101).
- HF의 NeoX식 RoPE는 컨버터가 Q/K 가중치를 permute(`undo_permute`)해서 llama.cpp에서는 `ROPE_TYPE_NORM`으로 계산된다(동일 결과).

로그확률 비교 (`eval/hf_logprobs.py`, `tools/nb_logprobs.cpp`, `eval/compare_lp.py`; 같은 토큰 id를 양쪽에 투입,
한국어 LaTeX 원문 / chat template+thinking 답 / 영어 문단 / RL 롤아웃 3개, 총 2,889 채점 토큰):

| 비교 | 평균 \|Δlogp\| | 최대 | top-1 | KL 평균 |
|---|---:|---:|---:|---:|
| GGUF **F32** (KV f32) vs HF **fp32** | 0.00022 | 0.024 | 100% | 2.2e-7 |
| GGUF **F16** (KV f16) vs HF **fp32** | **0.00021** | 0.028 | 99.97% | 2.3e-7 |
| GGUF BF16 vs HF fp32 | 0.0025 | 0.55 | 99.8% | 2.1e-5 |
| GGUF BF16 vs HF bf16 | 0.0103 | 0.50 | 99.1% | 4.0e-4 |
| HF bf16 vs HF fp32 (기준선) | 0.0100 | 0.41 | 99.2% | 3.9e-4 |
| 원본 Nanbeige4.2-3B: GGUF F16 vs HF fp32 | 0.00021 | 0.009 | 100% | 2.2e-7 |

해석: 비교는 같은 정밀도끼리 해야 한다. bf16→f16 변환은 사실상 무손실(f16 지수 범위 밖의 극소값만 영향)이고 llama.cpp CPU는 활성값을 f32로 유지하므로
GGUF-F16은 HF **fp32**와 비교하는 것이 맞고, 여기서 남는 2e-4는 F32 GGUF에서도 같은 크기인 누산 순서/RoPE 각도
계산 차이다. HF bf16은 모든 잔차·norm을 bf16으로 반올림하므로 fp32 대비 0.01(이 모델은 루프로 44층을 지나며
오차가 누적돼 bf16에 민감)이고, "GGUF BF16 vs HF bf16"이 0.01로 큰 것도 두 구현이 서로 다른 지점에서 반올림하기
때문이지 그래프 오류가 아니다. 시퀀스별로 보면 HF bf16 vs fp32는 영어 문단(분포 밖) 0.043, 한국어 LaTeX 원문 0.020,
chat+thinking 0.016, RL 롤아웃(분포 안) 0.006으로 분포에서 멀수록 커진다(이전에 본 0.1~0.2는 더 긴 분포 밖 텍스트 기준);
GGUF-F16 vs fp32는 모든 시퀀스에서 0.0001~0.001이다. 그리디 생성(`--gen`)은 두 문항 모두 정답(`\boxed{145}`, `\boxed{3}`)을 내고
`<|im_end|>`에서 멈췄다.

---

## 3. 양자화 품질 (Task 3)

held-out 텍스트 = RL 모델 자신의 2026 수능 풀이(학습에 쓰이지 않음) 24개, chat 포맷, **assistant 토큰만 채점**
(34,684 토큰), 기준 = 같은 GGUF의 F16. `scripts/quantize_all.sh`, `scripts/run_quant_eval*.sh`, `eval/summarize_kl.py`.

| 변형 | GGUF | HTP 상주 | PPL (F16 1.1319) | KL 평균 | KL p99 | 평균 \|Δlogp\| | top-1 일치 | \|Δlogp\|>0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 4.13 GiB | 3.62 GiB | 1.1331 | 0.0010 | 0.012 | 0.010 | 99.24% | 0.02% |
| Q6_K-pure-o8 | 3.30 GiB | 3.07 GiB | 1.1375 | 0.0063 | 0.097 | 0.028 | 98.11% | 0.58% |
| **Q5_K-pure-o8-imat** | **2.85 GiB** | 2.71 GiB | 1.1393 | **0.0075** | 0.113 | 0.031 | **97.91%** | 0.76% |
| Q5_K-pure-o8 | 2.85 GiB | 2.71 GiB | 1.1492 | 0.0160 | 0.238 | 0.048 | 96.83% | 2.10% |
| IQ4_NL-a8o8-imat | 2.88 GiB | 2.64 GiB | 1.1453 | 0.0129 | 0.195 | 0.042 | 97.17% | 1.58% |
| IQ4_NL-a8o8 | 2.89 GiB | 2.64 GiB | 1.1522 | 0.0180 | 0.268 | 0.050 | 96.83% | 2.33% |
| **Q4_0-a8o8-imat** | 2.89 GiB | 2.61 GiB | 1.1514 | **0.0177** | 0.265 | 0.050 | 96.67% | 2.23% |
| Q4_0-a8o8 | 2.88 GiB | 2.61 GiB | 1.1639 | 0.0258 | 0.398 | 0.062 | 96.19% | 3.19% |
| Q4_0-d8o8 | 2.77 GiB | – | 1.1862 | 0.0440 | 0.650 | 0.086 | 94.96% | 5.06% |
| IQ4_NL-o8 (기본 혼합) | 2.45 GiB | 2.20 GiB | 1.1729 | 0.0340 | 0.498 | 0.074 | 95.46% | 4.10% |
| Q4_K_M (기본 혼합) | 2.40 GiB | 2.33 GiB | 1.1811 | 0.0405 | 0.601 | 0.082 | 95.25% | 4.73% |
| Q4_0-o8-imat | 2.44 GiB | 2.15 GiB | 1.1773 | 0.0374 | 0.547 | 0.078 | 95.33% | 4.41% |
| Q4_0-o8 | 2.43 GiB | 2.15 GiB | 1.2046 | 0.0576 | 0.822 | 0.104 | 94.31% | 6.22% |
| Q4_0 (기본, lm_head Q6_K) | 2.31 GiB | 2.07 GiB | 1.2059 | 0.0585 | 0.846 | 0.105 | 94.27% | 6.31% |
| Q4_0-pure | 2.19 GiB | 1.92 GiB | 1.2139 | 0.0652 | 0.918 | 0.113 | 93.82% | 6.93% |

(HTP 상주 = hexcov가 계산한 HTP 타일 버퍼 크기, token_embd(CPU 쪽 host buffer, 0.27~0.5 GiB) 제외.)

- `-o8` = lm_head(`output.weight`) Q8_0, `-a8` = 어텐션(q/k/v/o) Q8_0, `-d8` = ffn_down Q8_0, `pure` = 혼합 규칙 없이 단일 타입.
- **민감한 것은 어텐션**이다: ffn_down을 Q8_0으로 올려도(d8) 효과가 작지만 어텐션을 올리면 KL이 절반이 된다.
  루프 구조라 같은 가중치의 양자화 오차가 두 번 들어가므로 일반 3B 모델보다 4bit 열화가 크다.
- 기본 `Q4_0`는 lm_head를 Q6_K로 만든다 → **Adreno 7xx(8 Gen 2/3)에서는 Q6_K lm_head가 CPU로 빠진다**
  (`ggml-opencl.cpp`의 A7x `ne[1] >= 32768` K-quant 가드). 그래서 모든 권장안은 `-o8`(lm_head Q8_0).
- HTP 융합 관점: Q/K/V 또는 gate/up이 같은 타입이어야 `MUL_MAT_NX`로 묶인다. `IQ4_NL`, `Q4_K_M` 기본 혼합은
  attn_v를 Q5_K/Q6_K로 올려 QKV 3중 융합이 깨진다 → 균일 타입(`pure`, `a8o8`)을 권장.
- RAM 계산(§5)에는 이 표의 GGUF 크기(전체 상주량)와 HTP 상주 크기(NPU 주소창 계산용)를 쓴다.

**권장 가중치**: `Q5_K-pure-o8-imat`(2.85 GiB, KL 0.0075; imatrix 없이 0.016) — `Q4_0-a8o8(-imat)`(2.88 GiB, KL 0.018~0.026)보다
작고 품질도 좋으며 타입이 균일해 HTP 융합도 그대로다. 다만 Q5_K 타일(32×32당 768 B, HTP 상주 2.71 GiB)은 Q4_0/Q8_0(GenieX 문서가 "깨끗한 NPU
실행"으로 권하는 타입)보다 역양자화가 무거우므로, 실기기에서 Q5_K 커널이 느리면 `Q4_0-a8o8-imat`(HTP 상주 2.61 GiB)로 간다.
여유 있으면 `Q6_K-pure-o8`(3.30 GiB, KL 0.006). `Q8_0`(4.13 GiB)은 거의 무손실이지만 HTP 상주 3.6 GiB가 NPU 세션
하나의 주소창(약 3.1 GiB)을 넘는다(§5.3).

**imatrix**: `llama-imatrix`(F16, 48×1024 토큰)로 *학습 세트* 문제의 chat 포맷 풀이(`eval/make_calib.py`,
`runs/eval/passrate_train.jsonl`; 2026 평가 세트와 겹치지 않음)에서 중요도 행렬을 만들고(`gguf/imatrix-train-math.gguf`)
`scripts/run_imat.sh`로 다시 양자화했다. 모든 4/5bit 변형에서 KL이 30~55% 줄었고 크기·커널은 그대로다.
특히 **Q5_K-pure-o8-imat는 KL 0.0075로 Q6_K-pure-o8(0.0063)에 근접하면서 0.45 GiB 작다** → 최종 권장.
Q4_0/Q8_0 커널만 쓰고 싶으면 **Q4_0-a8o8-imat**(KL 0.018).

---

## 4. 기기 경로 정확성 문제와 해결

### 4.1 토크나이저: 46 문항 전부 불일치 → 패치

학습/평가(veRL, SGLang)는 `apply_chat_template` + `add_special_tokens=False`로 토큰화했다. GenieX `llama_cpp`
플러그인은 템플릿을 llama.cpp로 렌더링한 뒤 `common_tokenize(add_special=true, parse_special=true)`로 토큰화한다.
`tools/chat_tok_check.cpp`가 정확히 이 경로를 재현하고 `eval/check_chat_tokens.py`가 2026 수학 46문항 프롬프트를
HF와 비교한다.

| 설정 | 46문항 일치 | 원인 |
|---|---:|---|
| 고정 llama.cpp, 변환 그대로(add_bos=True) | **0/46** | SPM 토크나이저가 `prepend_scheme="first"`를 무시하고 **모든 특수토큰 뒤에 `▁`를 붙임**: `<\|im_start\|>▁system`, `<\|im_start\|>▁user`, `<\|im_start\|>▁assistant`, `<think>▁\n` |
| + `llama-vocab-spm-prepend-first.patch` | 0/46 | 템플릿 맨 앞 `<\|im_start\|>`를 BOS 중복 방지로 떼어낸 뒤 BOS로 다시 붙이므로 뒤따르는 `system`이 "입력의 첫 조각"이 되어 `▁` 접두가 붙음 |
| + 패치 + **GGUF `add_bos_token=false`** | **46/46** | 템플릿의 `<\|im_start\|>`가 그대로 특수토큰으로 파싱되어 HF와 동일 |

원시 텍스트 경계 사례 6개(`" leading space"`, `"a<|im_end|>b c"`, `"<think>\nok</think>..."` 등)도 패치 후 6/6 일치.
패치는 GGUF가 `prepend_scheme="first"`를 선언한 SPM 어휘에서만 동작을 바꾼다(다른 모델 영향 없음).
`add_bos_token=false`는 학습 때와 같은 "추가 BOS 없음"을 의미하며 `scripts/set_add_bos_false.sh`가 메타데이터만 제자리 수정한다.

### 4.2 정지 조건
EOG = {`</s>`, `<|im_end|>`, `<|endoftext|>`}. thinking 모드 생성은 `</think>` 뒤 답을 쓰고 `<|im_end|>`에서 멈춘다(확인).

---

## 5. KV 캐시·메모리 예산 (Task 4)

### 5.1 크기
KV/토큰 = 44 논리층 × 2(K,V) × 8 head × 128 × 바이트: **F16 176 KiB, Q8_0 93.5 KiB, Q4_0 49.5 KiB**.

| ctx | F16 | Q8_0 | Q4_0 |
|---:|---:|---:|---:|
| 8k | 1.38 GiB | 0.73 GiB | 0.39 GiB |
| 16k | 2.75 GiB | 1.46 GiB | 0.77 GiB |
| 24k | 4.13 GiB | 2.19 GiB | 1.16 GiB |
| 32k | 5.50 GiB | 2.92 GiB | 1.55 GiB |

2026 수능 수학 RL 모델 풀이 길이(736개 생성): 중앙값 3.9k, p75 11.5k, p90 21.2k, p95 32.1k 토큰.
ctx(프롬프트 ~0.8k 제외)별로 끝까지 풀 수 있는 비율 = 정확도 상한: **8k 67.8% / 16k 83.6% / 24k 92.1% / 32k 95.0%**
(SGLang 64k 기준 97.1%). → 수학 추론에는 **32k가 필요**하고, 그러려면 KV 양자화가 사실상 필수다.

### 5.2 지원 여부
- HTP: `FLASH_ATTN_EXT`의 K/V는 **F16, Q8_0만**(`htp_kparams_solve_flash_attn`), `SET_ROWS`도 같음.
  q4_0 KV는 44개 FA + 44개 SET_ROWS가 CPU로 빠져 **그래프 분할 90개**(시뮬레이션) → 사용 금지.
  양자화 KV에는 llama.cpp가 자동으로 Hadamard 회전을 넣는데(`MUL_MAT_HADAMARD` 176개) HTP가 전용 op로 지원한다.
- OpenCL: FA는 D=128, F16/Q8_0/Q4_0 KV 지원. 단 **Adreno 7xx(E031 컴파일러)는 혼합/양자화 KV FA를 거절**(드라이버 크래시
  회피) → CPU. GenieX는 GPU에서 FA 기본 OFF이고, 양자화 V는 FA가 필수이므로 Adreno 8xx에서만 q8_0 KV 권장.
- GenieX는 KV 타입을 노출하지 않았다 → `geniex-nanbeige-mobile.patch`가 `GENIEX_LLAMA_CACHE_TYPE_K/V`,
  `GENIEX_LLAMA_FLASH_ATTN` env를 추가(FFI 변경 없음, 양자화 V면 FA 자동 ON).

KV 양자화 품질 (CPU, 긴 held-out 풀이 4개 × ~6k 토큰, assistant 24,069 토큰 채점, 기준 = F16 가중치 + F16 KV):

| 가중치 / KV | PPL (기준 1.1405) | KL 평균 | KL p99 | top-1 일치 |
|---|---:|---:|---:|---:|
| F16 / **q8_0** (Hadamard 회전, 기본) | 1.1406 | **0.00009** | 0.0015 | 99.79% |
| F16 / q8_0, 회전 끔(`LLAMA_ATTN_ROT_DISABLE=1`) | 1.1409 | 0.00026 | 0.0039 | 99.68% |
| F16 / q4_0 | 1.1546 | 0.0119 | 0.173 | 97.50% |
| Q8_0 / q8_0 | 1.1411 | 0.0010 | 0.014 | 99.29% |
| Q4_0-o8 / F16 | 1.2137 | 0.0587 | 0.836 | 94.13% |
| Q4_0-o8 / q8_0 | 1.2140 | 0.0590 | 0.830 | 94.14% |

→ **q8_0 KV는 사실상 무손실**(가중치 양자화 오차에 더해지는 것이 측정 불가 수준)이고, q4_0 KV는 가중치 한 단계
양자화만큼 손실이 있으면서 HTP에서는 CPU로 빠지므로 쓰지 않는다.

### 5.3 NPU 주소창 (3.1 GiB)과 루프 모델
Hexagon 세션 하나의 DSP 가상주소 창은 `HTP_OP_MAX_VMEM_DEFAULT`=3.2 GB(실제 ~3.1 GiB). 창을 넘으면 백엔드가 op 배치마다
버퍼를 매핑/해제한다(동작은 하지만 느림). 루프 모델은 토큰마다 22층 가중치와 44층 KV 전부를 **두 번에 걸쳐**
훑으므로, 작업 집합이 창을 넘으면 루프마다 재매핑이 일어날 수 있다.

| 가중치 | HTP 상주 | + KV q8_0 32k | 단일 세션에서 창 안? |
|---|---:|---:|---|
| Q4_0-o8 | 2.15 GiB | 5.07 GiB | ~6k 토큰까지만 |
| Q4_0-a8o8 | 2.61 GiB | 5.53 GiB | 아니오 |
| Q5_K-pure-o8 | 2.71 GiB | 5.63 GiB | 아니오 |
| Q8_0 | 3.62 GiB | 6.54 GiB | 가중치만으로 초과 |

해결책은 llama.cpp 문서의 "가상 세션 레이어 분할"(`GGML_HEXAGON_DEVICES=2` 또는 `3` → 세션 `HTP0`,`HTP1`[,`HTP2`],
GenieX에는 `--device-id HTP0,HTP1`처럼 장치 목록으로 전달)이다. 그런데
**고정 llama.cpp는 루프 모델을 논리층 기준으로 나눠서 모든 가중치가 첫 세션에, 두 번째 루프의 KV만 다른 세션에 간다**.
x86에서 RPC 백엔드 2개로 같은 상황을 재현(`scripts/test_layer_split_rpc.sh`):

| | dev0 가중치 | dev1 가중치 | KV dev0/dev1 | 그래프 분할 | PPL |
|---|---:|---:|---:|---:|---:|
| 고정 llama.cpp | 1689.7 MiB (22층 전부) | 517.2 MiB (lm_head만) | 46/42 MiB | **131** | 2.2360 |
| + `llama-looped-layer-device-map.patch` | 921.7 MiB | 1285.2 MiB | 48/40 MiB | **5** | 2.2360 (동일) |

패치 후에는 물리층 i와 그 논리 쌍둥이(i, i+22)의 가중치·KV가 같은 세션에 있다. 예: Q5_K-pure-o8 + q8_0 32k를
2세션으로 나누면 lm_head(0.5 GiB)를 가진 마지막 세션이 ~1.6 GiB 가중치 + ~1.3 GiB KV + compute로 3.1 GiB 경계에
걸리므로 **3세션 권장**(세션당 2.2~2.5 GiB).

### 5.4 권장 구성

| 폰 RAM | 권장 | 총 사용량(대략) |
|---|---|---|
| 12 GB | `Q5_K-pure-o8-imat`(또는 `Q4_0-a8o8-imat`), **ctx 32k, KV q8_0**, NPU 3 가상세션 | 가중치 2.7~2.9 + KV 2.9 + 컴퓨트/호스트 ~0.6 ≈ **6.2~6.4 GB** (12GB 폰 앱 가용량 ~6–7GB의 상한 근처 → 메모리 부족 시 ctx 24k: −0.7 GB) |
| 16 GB | `Q5_K-pure-o8-imat`(또는 `Q6_K-pure-o8`), **ctx 32k, KV q8_0**, NPU 3 가상세션 | ≈ 6.4~6.9 GB (여유 있음). F16 KV 32k(+2.6 GB)도 가능하지만 대역폭이 두 배 |

- 디코드는 메모리 대역폭 한계: 토큰당 읽기 = 22층 가중치 **×2(루프)** + lm_head + KV(깊이 L).
  Q4_0-a8o8은 2×2.10+0.50 ≈ 4.7 GiB(hexcov의 "per-use bytes") + KV(q8_0, 16k에서 1.46 GiB)이다. F16 KV면 KV 항이 두 배다. q8_0 KV는 메모리뿐 아니라
  긴 문맥 디코드 속도에도 이득이다(실측 필요).
- threads: 프리필은 GenieX 기본(전 코어), 디코드는 NPU가 일을 하므로 4~6. power mode: 벤치는 `burst`,
  긴 풀이/정확도 실행은 thermal을 고려해 `sustained_high_performance`.

---

## 6. Op 커버리지와 커널 최적화 (Task 5)

### 6.1 방법: `tools/hexcov.cpp` (실기기 없이 HTP 판정 그대로 재현)
Hexagon 백엔드의 `supports_op`/`graph_compute`는 DSP에 보내기 전 **호스트에서** 모든 결정을 한다:
`alloc_size ≤ max_vmem && htp_kparams_check()` → `graph_optimize`(같은 입력을 쓰는 matmul 묶기) →
`htp_fusion_tag_graph` → `htp_kparams_solve` + `htp_fusion_try_fuse`. 이 코드(`htp-kparams.cpp`, `htp-fusion.cpp`,
`htp-opnode.cpp`)는 세션을 5개 접근자(VTCM, HVX 수, HMX 수, mdev 수, 이름)로만 본다. hexcov는 이 파일들을
**수정 없이** x86에서 컴파일하고, llama.cpp CPU 빌드가 실제로 만든 Nanbeige 그래프(스케줄러 eval 콜백으로 포착,
프리필 ubatch와 디코드 1토큰)를 넣어 op별 지원 여부, 커널(HMX/HVX), VTCM, 융합 결과를 그대로 얻는다.
근사: 백엔드 배정(HTP가 지원하면 HTP, token_embd get_rows는 llama.cpp 규칙대로 CPU), opbatch 텐서 수 제한.
세션 파라미터는 HW 질의값을 모르므로 n_hvx 4/6/8, VTCM 8/4 MB로 스윕했다.

```bash
mobile/scripts/build_hexcov.sh && mobile/scripts/run_hexcov_matrix.sh   # 결과: /raid/.../mobile/eval/hexcov/*.txt
```

### 6.2 결과 (Q4_0-o8 / Q4_0-a8o8 / Q5_K / Q6_K / Q8_0 / IQ4_NL / Q4_K_M, KV f16·q8_0, ub 128·1024)
- **계산 노드 844개(F16 KV; q8_0 KV는 Hadamard 포함 1,020개) 중 1개만 빼고 전부 HTP**, CPU는 `GET_ROWS(token_embd)` 1개(llama.cpp가 입력층을 항상 CPU에 둠, 166k×3072
  테이블에서 행 몇 개 읽기). 그래프 분할 2개(CPU 임베딩 → HTP 나머지), kparams solve 실패 0.
- matmul: 디코드 `HVX_QBLOCK`, 프리필(128/1024 토큰) `HMX_2D`. lm_head(166144×3072 Q8_0) `HVX_QBLOCK`(VTCM 7.5 MB),
  F16 lm_head는 1.02 GB로 `opt_mbuf_static`(1 GB) 초과 위험 → `-o8` 권장 이유 하나 더.
- RMSNorm·SwiGLU·RoPE(NORM, θ=7e7)·SET_ROWS·GET_ROWS(마지막 층 out_ids)·ADD 모두 HTP. 
- FA(GQA 6:1, D=128): 프리필은 모든 길이에서 HMX. 디코드는 n_hvx≤5이면 KV<1024에서 HVX, 이상은 HMX(Br=6);
  **n_hvx≥6이면 G=6이 32의 약수가 아니라(`is_tile_aligned`) 항상 HVX** → §6.4 패치 대상.
- 융합(디코드, 층당): `RMS_NORM+MUL` → `MUL_MAT_NX(Q,K,V)` → ROPE×2 → SET_ROWS×2 → FA → `MUL_MAT+ADD(attn_out+잔차)`
  → `RMS_NORM+MUL` → `MUL_MAT_NX(gate,up)` → SWIGLU → `MUL_MAT+ADD(down+잔차)`.
  **루프 경계**: `... MUL_MAT+ADD [ffn_out-21 + l_out-21]` → `RMS_NORM+MUL [loop_norm-21]` → 22층부터 같은 패턴 그대로.
  loop_norm 출력은 다음 층 attn_norm 입력과 잔차 두 곳에서 쓰여서 두 RMSNorm을 하나로 합칠 수는 없다(정상).
- 프리필에서는 `MUL_MAT+ADD`가 대부분 안 붙는다: HMX matmul이 VTCM 8 MB를 거의 다 써서 잔차 블록을 넣으면 청크가
  작아지기 때문에 백엔드가 일부러 거절(`HMX efficiency reduced`). 정상 동작.
- 양자화 KV일 때 Hadamard 노드 때문에 QKV 3중 묶음이 Q+V / K로 갈린다(메모리 범위 검사; 할당 패턴 의존).
  디코드에서 층당 matmul 1개 추가, 영향 작음 — 실기기 프로파일로 확인할 항목.
- **가중치 공유**: 디코드 그래프의 matmul 가중치는 고유 텐서 155개(22층×7 + lm_head), 각 층 가중치는 정확히 2번 사용,
  이름당 텐서 1개. HTP 타일 버퍼는 2,206 MiB(Q4_0-o8)로, 층마다 따로였다면 3,896 MiB. 재패킹도
  `ggml_hexagon_tensor_is_tiled()` 검사로 한 번만 일어나고, `layers[i+22] = layers[i]`는 포인터 복사라 HTP 버퍼 중복 없음.

### 6.3 OpenCL(Adreno) 정적 검토 (`ggml_opencl_supports_op`)
MUL_MAT Q4_0/Q8_0/Q5_K/Q6_K/IQ4_NL(F32 활성) ✓, RMS_NORM(ne0%4==0) ✓, ROPE NORM ✓, GLU SWIGLU ✓, ADD/MUL F32 ✓,
SET_ROWS F16/Q8_0/Q4_0 ✓, Hadamard(FWHT 경로) ✓, FA D=128 F16 ✓. 예외: (1) Adreno 7xx에서 K-quant lm_head(≥32768행) → CPU
(→ `-o8`로 해결), (2) Adreno 7xx에서 양자화/혼합 KV FA → CPU(→ 7xx는 F16 KV + FA off(GenieX 기본)), (3) token_embd는 CPU.

### 6.4 루프 모델 특화 최적화 아이디어 (실기기 검증 필요, 성능 이득 주장 아님)
1. **세션 분할과 재매핑**: 위 §5.3. 2~3 가상세션 + 디바이스 매핑 패치로 재매핑 없는 구성이 가능해졌다.
   단일 세션(재매핑) vs 2/3세션(그래프 분할 5/7개)의 tok/s를 실측해 선택해야 한다.
2. **가중치 두 번 읽기 상쇄 = 다중 토큰 디코드**: 루프 때문에 토큰당 가중치 트래픽이 2배다. VTCM(8 MB)·L2로는
   1.6 GB를 루프 사이에 상주시킬 수 없고 루프 2는 루프 1 출력에 의존하므로 루프 간 배칭도 불가능하다. 대신
   n-gram/prompt-lookup 추측 디코딩(GenieX `--spec-type`)은 검증 배치 k개 토큰이 가중치를 루프당 한 번만 읽게 하므로
   이 모델에서 상대 이득이 크다(수식 반복이 많은 수학 풀이). 수락률·속도 실측 필요.
3. **lm_head**: Q8_0 lm_head는 토큰당 0.53 GB(전체의 ~12%). Q4_0 lm_head는 0.28 GB지만 KL이 오른다(§3, pure).
   어휘 166k라 "top-k 후보만 Q8로 재계산" 같은 2단계 lm_head도 후보.
4. **FA 디코드 GQA 행 분할**(패치 3, §7): G=6에서 HVX 스레드 5/6/7개일 때 KV head 스트림 11/12/13 → 8.
   결과는 비트 동일, DDR 트래픽 감소분은 실측 필요. n_hvx≥6이면 HMX 대신 HVX FA를 쓰므로 해당 SoC에서 의미가 있다.
5. **KV 대역폭**: 디코드 깊이 16k에서 KV 읽기(q8_0 1.5 GB/토큰)가 가중치의 1/3 수준 → q8_0 KV의 속도 효과,
   F16 vs q8_0의 `-d` 스윕(`bench/device_bench.sh`)으로 확인.
6. **프리필 ub**: GenieX Android NPU 기본 ub=1024. ub 1024에서도 모든 matmul이 HMX이고 FA도 HMX(시뮬레이션).
   프롬프트가 0.2~0.7k라 TTFT 영향은 작다.

---

## 7. 패치

| 파일 | 대상 | 이유 | 검증(실기기 없이) |
|---|---|---|---|
| `patches/llama-vocab-spm-prepend-first.patch` | llama.cpp `src/llama-vocab.cpp` | SPM이 `prepend_scheme="first"`를 무시하고 특수토큰마다 `▁`를 붙임 → 학습과 다른 토큰열 | 46/46 프롬프트 + 6/6 경계 사례가 HF와 동일(§4.1) |
| `patches/llama-looped-layer-device-map.patch` | llama.cpp `src/llama-model.cpp` | 루프 모델에서 다중 디바이스/부분 `-ngl` 시 가중치가 한 디바이스에 몰리고 매 층 디바이스 왕복 | x86 RPC 2디바이스: 분할 131→5, 가중치 균형, PPL 동일(§5.3) |
| `patches/llama-hexagon-fa-gqa-row-split.patch` | llama.cpp `ggml-hexagon/htp/flash-attn-ops.c` | HVX FA 디코드의 균등 행 분할이 GQA 그룹을 잘라 KV head를 중복 스트리밍 | `tools/fa_rowsplit_check.c`: 모든 행 정확히 1회, 스트림 수 항상 ≤ 기존(48/6: 11/12/13→8); Hexagon 툴체인으로 v73~v85 skel 컴파일 |
| `patches/geniex-nanbeige-mobile.patch` | GenieX `sdk/` | 위 3개를 `sdk/patches/`에 넣고 `sdk/CMakeLists.txt`에 등록 + `params.cpp`에 `GENIEX_LLAMA_CACHE_TYPE_K/V`, `GENIEX_LLAMA_FLASH_ATTN` | 빌드 시 7개 패치 모두 적용(configure 로그), Android/Linux 빌드 성공 |

GenieX 규칙대로 `third-party/`는 직접 고치지 않고 `sdk/patches/*.patch`로 configure 때 적용된다.
적용: `cp -a GenieX work && cd work && git apply /path/to/mobile/patches/geniex-nanbeige-mobile.patch` (원본 `0d049c1`에 깨끗이 적용됨).

---

## 8. 빌드 산출물 (Task 6)

```bash
# 작업 사본: /raid/cache/b300-07/ksat-rl/mobile/work/GenieX (패치 적용), 컨테이너 이름 ksat-mobile-*
docker run --rm -u $(id -u):$(id -g) -v $WORK:/workspace -w /workspace/sdk \
  docker.io/qualcomm/geniex-toolchain-android:v0.2.0 bash -c \
  'cmake --preset arm64-android-snapdragon-release -B build-android -DGENIEX_PLUGIN_QAIRT=OFF . \
   && cmake --build build-android -j && cmake --install build-android --prefix pkg-geniex \
   && cmake --build build-android -j --target llama-completion llama-bench llama-perplexity llama-tokenize'
# Linux arm64: geniex-toolchain-linux:v0.2.0 + --preset arm64-linux-snapdragon-release
```
- 결과: `/raid/cache/b300-07/ksat-rl/mobile/artifacts/android-arm64/` (57 MB) — `lib/libgeniex.so`, `lib/llama_cpp/`
  {`libgeniex_plugin.so`, `libllama.so`, `libggml*.so`(cpu/opencl/hexagon), `libggml-htp-v73/75/79/81/85.so`(DSP skel)},
  `bin/geniex-bench`, `bin/llama-{completion,bench,perplexity,tokenize}`. `artifacts/linux-arm64/`도 동일 구성.
- 툴체인 이미지에 Hexagon SDK 6.7.0.0 / HEXAGON_Tools 21.0.04 / NDK r29 / Rust가 들어 있어 빠진 것은 없었다.
- **꺼야 했던 것**: `GENIEX_PLUGIN_QAIRT=OFF` — `third-party/geniex-qairt` 서브모듈이 체크아웃돼 있지 않음(이 경로엔 불필요).
- Rust model-manager는 빌드 중 crates.io에서 공개 crate를 받는다(인증 불필요, Qualcomm 서비스 아님).
- HTP skel은 서명하지 않은 빌드(Android에서는 unsigned PD로 로드 — llama.cpp Snapdragon 문서와 동일). Windows용 서명은 해당 없음.
- Android 앱(AAR): `bindings/android`에서 `assembleRelease`가 `sdk/pkg-geniex/lib`을 그대로 패키징(notes/build.md).
- 같은 패치 트리로 **x86 호스트용 GenieX SDK**(`--preset default`, CPU만)도 빌드해 `geniex-bench --accuracy --think`를
  끝까지 돌려 봤다: `GENIEX_LLAMA_CACHE_TYPE_K/V=q8_0`가 `[Optimise] context params ... type_k=q8_0, type_v=q8_0`로
  반영되고, 직접 만든 2문항(등차수열 합, 로그 객관식)에 `</think>` 뒤 한국어 풀이와 `\boxed{145}`, `\boxed{3}`을 낸 뒤
  `stop_reason=eos`로 끝났으며, 그 로그를 `bench/csat_device_eval.py`의 파서/채점기가 그대로 정답 처리했다
  (`eval/e2e/gen_cpu_seed0.log`).

---

## 9. 실기기가 필요한 항목
- tok/s(프리필/디코드, 깊이별), NPU 1세션(재매핑) vs 2/3세션, GPU vs NPU, power mode별 thermal.
- FA GQA 행 분할 패치의 실제 효과(HVX FA 경로가 쓰이는 n_hvx≥6 SoC 또는 KV<1024).
- 양자화 KV + Hadamard 경로의 HTP 수치(시뮬레이션은 지원 판정만, 결과값은 CPU로만 검증).
- Adreno 7xx/8xx별 OpenCL 경로(드라이버 컴파일러 버전에 따른 FA/K-quant 가드).
- 실제 수능 정확도(아래 스크립트), 추측 디코딩 수락률.

---

## 10. 실기기 런북

### 10.1 준비 (호스트)
```bash
ART=/raid/cache/b300-07/ksat-rl/mobile/artifacts/android-arm64
G=/raid/cache/b300-07/ksat-rl/mobile/gguf
adb devices                                   # USB 디버깅 켠 Snapdragon 폰 (SM8650/8750/8850 등)
mobile/bench/device_bench.sh push $ART $G/nanbeige42-3b-math-v1-s40-Q5_K-pure-o8-imat.gguf $G/nanbeige42-3b-math-v1-s40-Q4_0-a8o8-imat.gguf
```
배치: `/data/local/tmp/geniex/{bin,lib,lib/llama_cpp,models}`.

### 10.2 실행 (NPU / GPU / CPU 선택)
```bash
adb shell
cd /data/local/tmp/geniex
export LD_LIBRARY_PATH=./lib:./lib/llama_cpp ADSP_LIBRARY_PATH=./lib/llama_cpp GENIEX_PLUGIN_PATH=./lib
export GENIEX_LLAMA_CACHE_TYPE_K=q8_0 GENIEX_LLAMA_CACHE_TYPE_V=q8_0      # 패치된 GenieX에서만 유효
M=models/nanbeige42-3b-math-v1-s40-Q5_K-pure-o8-imat.gguf
# NPU (HTP 1세션)
./bin/geniex-bench --plugin llama_cpp --device npu -m $M -c 32768 -p 512 -n 128 --power-mode burst --log info
# NPU 3 가상세션 레이어 분할 (ctx 32k 권장 구성, 루프 디바이스 매핑 패치 필요)
GGML_HEXAGON_DEVICES=3 ./bin/geniex-bench --plugin llama_cpp --device npu --device-id HTP0,HTP1,HTP2 -m $M -c 32768 -p 512 -n 128
# GPU (Adreno OpenCL). 7xx: KV f16 + FA off(기본), 8xx: q8_0 KV 가능
GENIEX_LLAMA_CACHE_TYPE_K=f16 GENIEX_LLAMA_CACHE_TYPE_V=f16 ./bin/geniex-bench --plugin llama_cpp --device gpu -m $M -c 16384 -p 512 -n 128
# CPU
./bin/geniex-bench --plugin llama_cpp --device cpu -m $M -c 32768 -p 512 -n 128 -t 6
# llama.cpp 도구로 직접 (대화형 확인)
./bin/llama-completion -m $M -dev HTP0 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 --jinja -cnv -t 6
```
- `--log info`로 `Found device: HTP0`, `ggml-hex: HTP0 hwinfo: threads N, hvx N, hmx 1, vtcm N MB`,
  `type_k=q8_0`, 레이어 배정을 확인한다. hwinfo의 hvx/VTCM 값을 `tools/hexcov`의 `--nhvx/--vtcm`에 넣으면 해당 SoC 기준
  커버리지를 다시 계산할 수 있다.
- `-ngl`은 기본 -1(전부). 루프 모델은 패치 후 `-ngl`이 **물리층** 기준(전부 = 23 이상, 99 권장).
- 권장 플래그: ctx 32768, KV q8_0(NPU, Adreno 8xx), ub 1024(NPU 기본), 디코드 threads 4~6, power mode
  `sustained_high_performance`(긴 풀이) / `burst`(짧은 벤치).

### 10.3 속도 벤치
```bash
mobile/bench/device_bench.sh run nanbeige42-3b-math-v1-s40-Q5_K-pure-o8-imat.gguf
# COMPUTES="npu" HEX_DEVICES=3 KV="q8_0" mobile/bench/device_bench.sh run ...   (llama-bench는 -dev HTP0/HTP1/HTP2 로 바꿔 실행)
```
geniex-bench(p512/1024/4096 + n128, 3회, JSON = notes/bench.md 스키마)와 llama-bench 깊이 스윕(`-d 0,4096,8192,16384`)을
돌린다. QDC를 쓸 수 있게 되면 `sdk/benchmark/qdc/`의 `run_linux.sh`/Android pytest 진입점에 같은 행렬
(model 경로, compute, ctx)을 넣으면 된다(이 작업에서는 QDC/AI Hub 계정이 없어 실행하지 않음).

### 10.4 2026 수능 수학 정확도 (eval_csat.py와 같은 프롬프트·채점)
```bash
OUT=/raid/cache/b300-07/ksat-rl/mobile/eval/device_csat_run1        # 저장소 밖
python3 mobile/bench/csat_device_eval.py prepare --out $OUT
python3 mobile/bench/csat_device_eval.py run --out $OUT \
    --model /data/local/tmp/geniex/models/nanbeige42-3b-math-v1-s40-Q5_K-pure-o8-imat.gguf \
    --compute npu --hex-devices 3 --device-id HTP0,HTP1,HTP2 --kv q8_0 --seeds 0,1,2,3 --ctx 32768
python3 mobile/bench/csat_device_eval.py grade --out $OUT          # eval_csat.report()와 같은 표 출력
```
- 프롬프트 = `instruction(p) + "\n\n" + p["text"]`(rl/ksat_rl/make_parquet), thinking ON, temperature 0.6, top_p 0.95,
  채점 = `</think>` 뒤 마지막 `\boxed{}` (`ksat_rl.reward.grade`). 기기 쪽 차이: geniex-bench에 top_k/min_p 플래그가 없어
  GenieX 기본값(top_k 40, min_p 0.05)이 쓰이고, max tokens는 ctx 때문에 30k(SGLang 평가는 65k).
- 시드마다 46문항을 한 번씩 생성(`--no-warmup -r 1`). 프롬프트 파일은 실행 후 기기에서 지운다.
- 채점기는 SGLang 평가 결과로 만든 가짜 로그로 자체 검증했다(시드 2개, 정확도 0.9674로 원본과 일치).
- 실행 시간 감: 46문항 × 평균 ~7k 토큰 생성 → 디코드 10 tok/s 가정 시 시드당 약 9시간. 처음에는 `--seeds 0`으로.

---

## 11. 파일
- `scripts/convert_gguf.sh`, `set_add_bos_false.sh`, `quantize_all.sh`, `run_imat.sh`, `run_lp.sh`, `run_quant_eval.sh`
  (`quant` / `long` 모드), `run_quant_eval2.sh`, `build_x86_tools.sh`, `build_hexcov.sh`, `run_hexcov_matrix.sh`,
  `test_layer_split_rpc.sh`
- 결과 원본: `/raid/cache/b300-07/ksat-rl/mobile/eval/lc/{quant,kv}_summary.json`, `eval/lc/cmp_*.json`, `eval/hexcov/*.txt`,
  `logs/rpc-split-*.log`, `work/GenieX/build-logs/`. F16 기준 전체 logprob 덤프(`eval/lc/*_F16_ref`, 약 40 GB)는 양자화 재평가용이며 지워도 된다.
- `tools/nb_logprobs.cpp`(토큰 id 기반 logprob/KL/그리디), `tools/hexcov.cpp`(HTP 커버리지 시뮬레이터),
  `tools/chat_tok_check.cpp`(GenieX 템플릿·토큰화 경로 재현), `tools/fa_rowsplit_check.c`
- `eval/make_seqs.py`, `make_gen_seqs.py`, `make_calib.py`, `hf_logprobs.py`, `compare_lp.py`, `summarize_kl.py`, `check_chat_tokens.py`
- `bench/device_bench.sh`, `bench/csat_device_eval.py`
- `patches/*.patch`
