# 조사 출처

확인일 2026-11-18. 전체 데이터는 내려받거나 재계수하지 않았습니다. 제작 시작 시 commit SHA와 원본 파일 SHA-256을 고정하세요.

## S01: Nemotron-Math-Proofs-v3-SFT
https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v3-SFT

414,890 samples, 15,818 unique problems; 4 task types; counts reported by card, not independently recounted.

## S02: Nemotron-Math-Proofs-v3-RL
https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v3-RL

9,597 prompts; responses and rewards generated online.

## S03: An Open Recipe for IMO Gold
https://arxiv.org/html/2609.10712v1

2026 proof-system recipe, not a 4B KSAT accuracy guarantee.

## S04: Nemotron-SFT-Math-v3
https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3

3,638,783 examples; pure CoT and Python TIR, per-record licenses.

## S05: OpenMathReasoning
https://huggingface.co/datasets/nvidia/OpenMathReasoning

306K unique problems with solutions; 3.2M CoT, 1.7M TIR, 566K GenSelect; another 193K prompt-only problems. Rounded counts.

## S06: OpenThoughts3 recipe
https://www.open-thoughts.ai/blog/ot3

1.2M pairs; QwQ-32B; teacher rank differs from benchmark rank; filtering did not improve their controlled experiments.

## S07: OpenThoughts3-1.2M
https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M

ShareGPT conversations; 850K math, 250K code, 100K science pairs.

## S08: OpenR1-Math-220k
https://huggingface.co/datasets/open-r1/OpenR1-Math-220k

Default subset roughly 94K problem rows, distinct from the family name; multiple candidates and verifier fields.

## S09: s1K
https://huggingface.co/datasets/simplescaling/s1K

1,000 examples; question/solution/thinking_trajectories.

## S10: s1 paper
https://arxiv.org/abs/2501.19393

Small selected SFT set and test-time budget scaling; not proof 1,000 samples suffice for any small model.

## S11: NeMo-Skills construction
https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/releases/openmathreasoning/dataset.md

Public generation pipeline; pin commit; do not inherit MCQ exclusion or majority answer replacement for KSAT.

## S12: TRL SFTTrainer
https://huggingface.co/docs/trl/sft_trainer

Chat formats and masks; confirm actual tokenizer template.

## S13: TRL dataset formats
https://huggingface.co/docs/trl/dataset_formats

SFT, preference and prompt-only formats.

## S14: TRL GRPOTrainer
https://huggingface.co/docs/trl/grpo_trainer

Online rollouts and reward interface.

## S15: Nemotron multilingual
https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v1

3,065,255 examples; six subsets exclude Korean; English reasoning preserved.

## S16: HRM8K
https://huggingface.co/datasets/HAERAE-HUB/HRM8K

Korean-English math benchmark; preserve as evaluation rather than training/evaluation overlap.