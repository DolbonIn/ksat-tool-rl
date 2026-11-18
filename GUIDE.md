# 실행 가이드: Nanbeige4.2-3B 수능 수학 RL

기준일: 2026-11-19. 이 저장소로 **(1) 학습된 모델을 받아 실행·평가하고, (2) 데이터 구축부터 RL 학습까지 재현하고, (3) 모바일(Snapdragon)에 배포**하는 방법입니다. 모든 작업은 Docker 안에서 실행합니다.

## 0. 결과 요약

| 2026학년도 수능 수학 (16회 샘플링 기대점수) | 원본 Nanbeige4.2-3B | **RL v1 (step 40)** |
|---|---:|---:|
| 확률과 통계 | 96.4 | **97.3** |
| 미적분 | 92.6 | **97.5** |
| 기하 | 95.2 | **95.5** |
| 전체 문항 정답률 | 92.9% | **97.2%** |

자세한 결과(문항별, 검증 추이, 신뢰구간)는 [`experiments/math_rl_v1/`](experiments/math_rl_v1/)에 있고, 학습 파이프라인 설계와 측정값은 [`rl/README.md`](rl/README.md)에 있습니다.

## 1. 모델 받기

| 형식 | 위치 | 용도 |
|---|---|---|
| **GGUF Q8_0 (8비트)** | HF `sionic-ai/Nanbeige4.2-3B-KSAT-Math-RL-Q8_0-GGUF` (**비공개, sionic-ai 조직 멤버만**) | llama.cpp, GenieX(모바일·PC) |
| BF16 safetensors | 서버 `/raid/cache/b300-07/ksat-rl/models/math_v1_step40` (HF 미업로드) | SGLang 서빙, 추가 학습 |
| 기타 GGUF (Q5_K-o8-imat 추천, Q4_0-a8o8-imat 등) | 서버 `/raid/cache/b300-07/ksat-rl/mobile/gguf/` | 모바일 크기와 품질 절충 |

```bash
# sionic-ai 조직에 접근 권한이 있는 HF 토큰 필요
export HF_TOKEN=...   # 저장소에 커밋하지 마세요 (.env는 gitignore 대상)
huggingface-cli download sionic-ai/Nanbeige4.2-3B-KSAT-Math-RL-Q8_0-GGUF --local-dir ./nanbeige-ksat-q8
```

## 2. 실행

### 2-1. llama.cpp (GGUF Q8_0)
1. 토크나이저 패치를 반드시 적용합니다: [`mobile/patches/llama-vocab-spm-prepend-first.patch`](mobile/patches/). 패치하지 않으면 llama.cpp가 특수 토큰 뒤에 `▁`를 넣어 학습 때와 입력이 달라집니다.
2. 실행:
   ```bash
   llama-server -m nanbeige-ksat-q8/nanbeige42-3b-ksat-math-rl-Q8_0.gguf -c 32768 -fa on \
     --cache-type-k q8_0 --cache-type-v q8_0 --jinja
   ```
3. 프롬프트는 `ksat_rl/make_parquet.py::instruction`과 같은 형식으로, 지시문 뒤에 문제를 붙입니다. 권장 샘플링은 temperature 0.6, top_p 0.95, top_k 20입니다. 풀이가 길어 32k 이상의 컨텍스트가 필요합니다.

### 2-2. 모바일 (Qualcomm GenieX, Snapdragon NPU/GPU/CPU)
- 빌드(Android/Linux arm64 크로스 컴파일), 패치 적용, adb 배포, NPU/GPU 선택, 권장 설정은 [`mobile/README.md`](mobile/README.md)에 있습니다.
- 기기 벤치마크 스크립트는 `mobile/bench/device_bench.sh`, 기기 정확도 스크립트는 `mobile/bench/csat_device_eval.py`입니다.

### 2-3. 서버 (SGLang, BF16) + 2026 수능 평가 재현
```bash
docker build -t ksat-rl/verl-sgl:v4 rl/docker          # SGLang nightly(Nanbeige 지원) + veRL main + 호환 패치
# 베이스라인과 같은 프로토콜: T=0.6, top_p 0.95, top_k 20, thinking on, 65,536 토큰
GPUS=0,1,2,3,4,5,6,7 MODEL=/models/math_v1_step40 TAG=my_eval N=16 \
  SGL_EXTRA="--tokenizer-path /models/Nanbeige4.2-3B" rl/scripts/docker_eval.sh
# 결과: /raid/cache/b300-07/ksat-rl/runs/eval/my_eval.{jsonl,report.txt,summary.json}
```
- 모델과 데이터 경로는 컨테이너 기준(`/models`, `/data`, `/runs`)이고, 실제 위치는 `/raid/cache/b300-07/ksat-rl/{models,data,runs}`입니다.
- 도구 모드(Python sandbox) 평가는 `rl/scripts/docker_eval_tool.sh`를 씁니다. sandbox 실행 방법은 [`rl/ksat_rl/tools/README.md`](rl/ksat_rl/tools/README.md)에 있습니다.

## 3. 학습 재현

| 단계 | 스크립트 | 비고 |
|---|---|---|
| ① 기출 데이터 구축 (OCR, 정답표, 검증) | `data_pipeline/run_pipeline.sh` (2022–26 수학), `data_pipeline/old/` (2017–21), `data_pipeline/multi/run_multi.sh` (국어·영어·탐구) | DeepSeek V4.1 Flash eco(opengateway)로 OCR과 검증, 정답은 공식 정답표에서만 |
| ② 합성 문항 | `python -m ksat_rl.synth_variants --mode {harder,easier,notation,terms}` | eco 생성 → eco 4회 독립 풀이, 3회 이상 일치하면 채택 |
| ③ 난이도 측정 | `rl/scripts/docker_eval.sh` + `EVAL_EXTRA="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report"` | 결과를 바로 저장하므로 `--resume`과 `--reverse`로 나누거나 이어받기 가능 |
| ④ RL 세트 | `python -m ksat_rl.build_rl_set` (v1) / `build_rl_set_v2` (2차) | pass@8이 1/8–7/8인 문항만, 2026 수능과 9월 모평 계열 자동 제외 |
| ⑤ GRPO 학습 | `rl/scripts/run_math_v1.sh` (v1), `run_math_v2.sh` (2차) | 8×B300, step당 약 12–13분 |
| ⑥ 체크포인트 → 모델 폴더 | `rl/scripts/ops/prepare_ckpt_model.sh <ckpt>/actor/huggingface <name>` | veRL HF 체크포인트에 빠진 모델 코드와 토크나이저를 보완 |
| ⑦ 평가 | 2-3절 참고 | |

**사전 준비**
- `.env`(gitignore 대상)에 `OPENGATEWAY_API_KEY`, `OPENGATEWAY_BASE_URL=https://apis.opengateway.ai/v1`를 둡니다. HF 업로드가 필요하면 `HF_TOKEN`도 둡니다.
- 모델 ID는 반드시 `deepseek/deepseek-v4.1-flash-eco` 형식(조직 접두사 포함)으로 씁니다.
- 학습용 모델 폴더 `Nanbeige4.2-3B-train-fused`는 원본 가중치에 학습용 모델 코드(`rl/patches/modeling_nanbeige_train.py`)를 붙인 것입니다. 이유는 [`rl/README.md`](rl/README.md)에 있습니다.

**알아 둘 점**
- 원본 Nanbeige HF 코드는 transformers 5.x에서 RoPE 버퍼가 초기화되지 않아 NaN이 납니다. 그래서 학습은 별도 구현을 씁니다.
- veRL ↔ SGLang 호환 패치(`rl/docker/patches/`)가 없으면 SGLang이 가중치 업데이트를 **아무 경고 없이** 무시합니다.
- 실행 중인 bash 스크립트(`train_grpo.sh`, `docker_eval.sh`)는 수정하지 마세요. bash는 스크립트를 조금씩 읽기 때문에 실행 중인 작업이 깨집니다. 수정하려면 복사본(예: `train_grpo_v1b.sh`)을 쓰세요.

### 추가 전략 (2단계)
- **디딤돌·표기·용어 문항:** `synth_variants --mode easier|notation|terms`
- **힌트 기반 STaR:** `ksat_rl/star.py`, `rl/scripts/run_star_sft.sh`. 이 모델은 힌트를 그대로 인용하며 풀어서, 힌트 없는 SFT 데이터로 쓸 수 있는 풀이가 적었습니다(5,664개 중 62개). 대신 RL 세트에 1단계 힌트 문항을 섞습니다.
- **도구 호출 RL:** `rl/scripts/run_math_tool_v1.sh` (few-shot 예시 2 → 1 → 0개 커리큘럼). GPU smoke test는 통과했습니다. 학습 없이 도구만 주면 점수가 79/77/78로 오히려 떨어집니다.

## 4. 데이터 경계
- 기출 원문, 정답표, 학습·평가 JSONL은 저장소에 **넣지 않습니다**(`/raid/cache/b300-07/ksat-rl/data/`에만 둠). EBSi 자료는 재배포 금지라 내부 전용입니다.
- 평가 결과 `*.jsonl.gz`에는 모델 출력이 들어 있어, 출력이 지문 일부를 인용할 수 있습니다(기존 베이스라인 실험과 같은 기준).
- 키와 토큰은 `.env`에만 둡니다.
