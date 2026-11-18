#!/usr/bin/env bash
# Strategy 3: small SFT on the policy's OWN correct solutions obtained with hints (STaR-style rationalisation),
# trained on the hint-free prompt. Runs inside ksat-rl/verl-sgl (FSDP SFT trainer, Nanbeige training modeling).
#
# Pipeline (GPU phase):
#   1. python -m ksat_rl.star make-prompts --hints /data/hints/hints.jsonl --pool /data/synth/seed_pool_all.jsonl --out /data/star/hint_prompts.jsonl
#   2. GPUS=0-7 MODEL=<policy ckpt> TAG=star_rollouts PROBLEMS=/data/star/hint_prompts.jsonl N=8 MAX_TOKENS=32768 \
#        EVAL_EXTRA="--temperature 0.8 --top-p 1.0 --top-k -1 --no-report" rl/scripts/docker_eval.sh
#   3. python -m ksat_rl.star build-sft --samples /out/star_rollouts.jsonl --pool /data/synth/seed_pool_all.jsonl \
#        --test /data/ksat_math/test_2026.jsonl --out /data/star/star_sft.parquet --per-item 2
#   4. this script (then continue GRPO from the SFT checkpoint)
# Verified on CPU (scripts/check_sft_dataset.py): MultiTurnSFTDataset tokens == whole chat template, and the loss mask
# covers only <think>…</think> + answer + <|im_end|> (same boundary as RL rollouts: prompt ends with "<think>\n").
set -euo pipefail
NGPUS=${NGPUS:-8}
MODEL=${MODEL:?policy checkpoint (HF dir, Nanbeige training modeling) to fine-tune}
DATA=${DATA:-/data/star/star_sft.parquet}
EXP=${EXP:-star_sft_v1}
LR=${LR:-2e-6}
EPOCHS=${EPOCHS:-2}
torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS -m verl.trainer.sft_trainer \
  data.train_files=$DATA data.val_files=null \
  data.messages_key=messages data.enable_thinking_key=enable_thinking \
  data.max_length=40960 data.truncation=left \
  data.use_dynamic_bsz=True data.max_token_len_per_gpu=65536 data.train_batch_size=32 data.micro_batch_size_per_gpu=1 \
  engine=fsdp engine.strategy=fsdp2 \
  model.path=$MODEL model.trust_remote_code=True model.use_remove_padding=True model.enable_gradient_checkpointing=True \
  +model.override_config.attn_implementation=sdpa \
  optim.lr=$LR optim.weight_decay=0.0 optim.lr_warmup_steps_ratio=0.05 \
  'checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
  trainer.default_local_dir=/runs/$EXP trainer.project_name=ksat_math trainer.experiment_name=$EXP \
  trainer.total_epochs=$EPOCHS trainer.logger='["console"]' "$@" 2>&1 | tee /runs/$EXP.log
