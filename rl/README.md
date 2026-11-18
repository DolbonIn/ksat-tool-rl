# Nanbeige4.2-3B 수학 RL 파이프라인 (veRL + SGLang, Docker)

기준일 2026-11-18. `ksat_tool_rl_v2/README.md`의 Phase-1 설계(검수 문제·공식 정답·규칙 보상·GRPO)를 실제로 실행하기 위한 코드입니다. 이번 단계는 **도구 없는(no-tool) GRPO**이며, 도구 호출 RL은 이 파이프라인 검증 뒤 단계입니다.

## 구성

| 경로 | 내용 |
|---|---|
| `docker/Dockerfile` | `lmsysorg/sglang:nightly-dev-cu13-20261030-302b78cc`(Nanbeige4.2 지원 sglang#32151 포함, B300/sm_103, FA4) + veRL main `b0871fd89d` |
| `docker/patches/patch_verl_sglang_session.py` | veRL↔SGLang nightly 호환 패치 (아래) |
| `patches/modeling_nanbeige_train.py` | 학습용 Nanbeige 구현 (FA4 varlen, 공식 체크포인트와 동일 파라미터 이름) |
| `ksat_rl/reward.py` | 규칙 보상: `</think>` 뒤 마지막 `\boxed{}`가 공식 정답이면 1, 아니면 0. 베이스라인 `solve.py::grade`와 동일 판정 |
| `ksat_rl/make_parquet.py` | 문제 JSONL → veRL parquet. 프롬프트는 베이스라인 평가 프롬프트와 같은 형식(시험명만 다름) |
| `ksat_rl/select_by_passrate.py` | 베이스 모델 pass@8로 학습 문항 선택 (0 < pass < 1) |
| `scripts/train_grpo.sh`, `scripts/docker_train.sh` | GRPO 실행 (모든 설정은 환경변수) |
| `scripts/eval_csat.py`, `scripts/docker_eval.sh` | 2026 수능 평가 — 베이스라인과 같은 프로토콜(T=0.6, top_p 0.95, top_k 20, 65,536 토큰, n=8) |
| `scripts/check_sglang_vs_train.py` | **필수 정합성 검사**: SGLang rollout log-prob vs 학습 모델 log-prob |
| `scripts/check_train_model.py` | 학습 구현 vs 공식 HF 코드 / fp32 비교 |
| `scripts/bench_rollout.py` | SGLang decode 처리량 측정 |

## 모델 쪽에서 확인한 사실

1. **공식 HF 원격 코드는 transformers 5.x에서 그대로 쓸 수 없음.** 비영속 버퍼인 RoPE `inv_freq`가 초기화되지 않아(값이 1e-38 수준 쓰레기) 출력이 NaN이 되고, cache 경로는 `DynamicCache.from_legacy_cache` 제거로 실패합니다. 또 FA2 경로는 `attention_mask=None`일 때 `flash_attn_func`를 호출해, veRL remove-padding처럼 여러 시퀀스를 이어 붙인 입력에서 시퀀스 간 attention이 섞입니다.
2. 그래서 **학습 전용 구현**(`modeling_nanbeige_train.py`)을 작성했습니다. config에서 실제로 켜진 기능만 구현합니다(22개 물리 레이어 × 2 loop, loop마다 마지막 `norm` 적용, GQA 48/8, RoPE θ=7e7). 켜지지 않은 기능(ngram, hyper-connection 등)이 config에 있으면 예외를 냅니다. `inv_freq`는 매번 계산하고, attention은 `position_ids`로 시퀀스 경계를 구하는 FA4 varlen 커널입니다.
3. **이 모델은 bf16 오차가 매우 큽니다.** 비분포 텍스트에서 bf16과 fp32의 토큰 log-prob 차이는 평균 0.1~0.27(최대 9.6)입니다. 공식 구현의 bf16도 마찬가지입니다. 다만 bf16 구현끼리는 서로 0.03~0.08 안으로 일치합니다(같은 지점에서 residual을 반올림하기 때문). FA4 연산 자체는 SDPA와 같은 정확도입니다(fp32 대비 backward cosine 0.99999).
4. **결정적 검사: 실제 rollout 토큰에서 SGLang ↔ 학습 모델** (`check_sglang_vs_train.py`)
   - |Δlogp| 평균 0.0085, p99 0.19, 최대 0.34
   - IS 비율 p1~p99 = 0.855~1.15, 시퀀스 평균 편향 0.0000
   - veRL 학습 중 `rollout_corr/log_ppl_abs_diff` ≈ 0.001
5. FSDP2 래핑: 디코더 레이어가 forward 한 번에 2번 실행되므로 레이어마다 FSDP 단위로 감싸지 않습니다. `NanbeigeLayerStack`(파라미터 이름은 `model.layers.{i}.*` 그대로, embedding 제외)을 한 단위로 묶었습니다. 4B 파라미터는 B300 한 장에 샤딩 없이도 들어갑니다.

## veRL 패치 (`docker/patches/patch_verl_sglang_session.py`)

- 새 SGLang은 `update_weights_from_tensor`를 `begin_weight_update`/`end_weight_update` 세션 안에서만 허용합니다. 세션 밖에서 호출하면 **예외 없이 `success=False`만 반환**하므로, 그대로 두면 rollout 엔진이 초기 가중치로 학습 내내 조용히 남습니다. 패치는 세션을 열고, 실패하면 예외를 냅니다.
- untied 모델에서 veRL은 `lm_head`를 별도 FSDP2 단위로 감쌉니다. 그런데 fused log-prob 경로는 `lm_head.forward`를 부르지 않고 `lm_head.weight`만 읽어 DTensor 오류가 납니다. 그래서 `lm_head`를 root 단위에 남겼습니다.
- 이미지의 sglang 버전 문자열이 `0.0.0.dev`이어서 veRL이 Hopper 전용 FA3 backend를 고릅니다. `engine_kwargs.sglang.attention_backend`로 명시합니다.

## 속도 최적화 측정 (Nanbeige4.2-3B, B300)

학습 단계 (2 GPU, step당 ~120k 토큰, 4,096 응답):

| 단계 | 기본 | fused log-prob + lm_head 패치 | + elementwise fusion |
|---|---:|---:|---:|
| old_log_prob | 6.0 s | 1.9 s | – |
| update_actor | 20.6 s | 4.3 s | 3.9 s |
| update_weights | 24 s | 2.2 s | 2.2 s |
| 최대 메모리 | 144 GB | 61 GB | 58 GB |
| step | 77 s | 35 s | 32 s |

- `torch.compile` entropy를 끄면 update_actor가 30~40 s로 느려집니다(기본값 유지).
- bf16 fused-linear(torch backend)는 166k vocab의 fp32 logits(마이크로배치당 43 GB)를 만들지 않습니다.

SGLang decode 처리량 (1 GPU, 2,048 토큰 고정, `bench_rollout.py`):

| backend | 동시 1 | 동시 32 | 동시 128 |
|---|---:|---:|---:|
| flashinfer | 258 tok/s | 6,013 | 8,748 |
| **trtllm_mha** | **275** | **6,537** | **9,890** |
| fa4 | 242 | 5,909 | 8,544 |
| flashinfer + NGRAM 투기 디코딩 | 353 | 2,672 | – |

NGRAM 투기 디코딩은 단일 스트림만 빠르고, 배치 처리량이 절반 이하로 떨어져 RL에 쓰지 않습니다.

32k 응답 실측 (4 GPU, 16 문제 × 8 rollout, 2회 업데이트/step, steady state): gen 163 s, old_log_prob 6.3 s, update_actor 17.8 s, update_weights 3.6 s. step 시간의 약 85%가 생성이며, 가장 긴 응답이 시간을 결정합니다.

## 실행

```bash
# 이미지
docker build -t ksat-rl/verl-sgl:v4 rl/docker
# 학습 (예: 8 GPU)
IMAGE_TAG=v4 GPUS=0,1,2,3,4,5,6,7 NAME=ksat-train TRAIN_FILE=/data/ksat_math/rl_train.parquet EXP=grpo_v1 \
  BS=64 MINI_BS=16 N=8 MAX_RESP=32768 GPU_UTIL=0.8 STEPS=100 MODEL=/models/Nanbeige4.2-3B-train-fused \
  rl/scripts/docker_train.sh actor_rollout_ref.model.use_fused_kernels=True actor_rollout_ref.model.fused_kernel_options.impl_backend=torch
# 2026 수능 평가
GPUS=0,1,2,3,4,5,6,7 MODEL=/models/Nanbeige4.2-3B TAG=base rl/scripts/docker_eval.sh
```

대용량 파일은 `/raid/cache/b300-07/ksat-rl/{models,data,runs}`에 있습니다(저장소에 포함하지 않음).
