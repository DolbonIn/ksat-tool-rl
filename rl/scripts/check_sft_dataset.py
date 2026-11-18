"""CPU check: veRL MultiTurnSFTDataset on Nanbeige with a thinking assistant turn.
Verifies (1) dataset input_ids == whole-conversation chat-template tokenization, (2) loss mask covers exactly
the assistant turn (<think>…</think> + answer + <|im_end|>) and nothing of the user/system prompt."""
import sys, tempfile
import pandas as pd, torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
tok = AutoTokenizer.from_pretrained("/models/Nanbeige4.2-3B", trust_remote_code=True)
row = {"messages": [{"role": "user", "content": "다음은 대학수학능력시험 수학 영역 단답형 문제입니다. 1+1의 값을 구하시오."},
                    {"role": "assistant", "content": "따라서 답은 \\boxed{2}입니다.", "reasoning_content": "1과 1을 더하면 2이다. 검산: 2-1=1."}],
       "enable_thinking": True}
f = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False).name
pd.DataFrame([row]).to_parquet(f)
cfg = OmegaConf.create({"messages_key": "messages", "tools_key": "tools", "enable_thinking_key": "enable_thinking",
                        "max_length": 4096, "truncation": "error", "pad_mode": "no_padding", "apply_chat_template_kwargs": {},
                        "ignore_input_ids_mismatch": False})
ds = MultiTurnSFTDataset(parquet_files=f, tokenizer=tok, config=cfg)
item = ds[0]
ids = item["input_ids"].tolist(); mask = item["loss_mask"].tolist() if "loss_mask" in item else None
full = tok.apply_chat_template(row["messages"], tokenize=False)
full_ids = tok(full, add_special_tokens=False)["input_ids"]
print("keys", list(item.keys()))
print("dataset == whole-template tokens:", ids == full_ids, len(ids), len(full_ids))
if mask is not None:
    trained = tok.decode([t for t, m in zip(ids, mask) if m])
    print("TRAINED SPAN:", repr(trained))
    untrained = tok.decode([t for t, m in zip(ids, mask) if not m])
    print("UNTRAINED SPAN tail:", repr(untrained[-120:]))
print("FULL TEXT:", repr(full))
