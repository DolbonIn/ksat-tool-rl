# [ksat-patch-nanbeige] installed into verl/utils/tokenizer/nanbeige.py by rl/docker/patches/patch_verl_nanbeige_tools.py
"""Continuous Token builder for Nanbeige4.x (config.json model_type="nanbeige").

Nanbeige uses ChatML turns like Qwen3: every turn ends with "<|im_end|>\\n" in the chat template, while
generation stops AT "<|im_end|>" (the eos token), so the "\\n" must be restored before a tool/user turn is
appended.  veRL's QwenContinuousTokenBuilder does exactly that but cannot be reused:

  * Nanbeige's tokenizer pre-tokenizer is Metaspace(prepend_scheme="first"): a string tokenized ON ITS OWN
    gets a dummy-prefix "▁", so tokenizer.encode("\\n") == ["▁", "<0x0A>"] (two ids) and the Qwen builder
    raises.  Inside the conversation (after a special token) "\\n" is the single byte token <0x0A>; we take
    the in-context id from encoding "<|im_end|>\\n" and dropping the <|im_end|>.
  * The `nanbeige` tool parser stops generation at "</tool_call>" (so the model cannot write its own
    <tool_response>).  The template closes that assistant turn with "<|im_end|>\\n", which the model then
    never generated: both tokens are inserted as boundary tokens (loss mask 0, logprob 0).

Tool/user deltas themselves are rendered by the base class as a token-level suffix of a whole-string
template render (synthetic prefix + appended messages), i.e. they are tokenized IN CONTEXT; no standalone
tokenization (and hence no stray "▁") is involved.  With this builder the merged rollout stream equals a
single-pass chat-template render of the same conversation (checked by verify_continuous_token_builder and
by rl/ksat_rl/tools/tests/test_verl_tool.py).
"""

from __future__ import annotations

import logging
from typing import Any

from .continuous_token import ContinuousTokenBuilder, MergeResult, require_token_id

logger = logging.getLogger(__name__)


class NanbeigeContinuousTokenBuilder(ContinuousTokenBuilder):
    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._im_end_id = require_token_id(tokenizer, "<|im_end|>")
        self._tool_call_end_id = require_token_id(tokenizer, "</tool_call>")
        ids = list(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        if len(ids) != 2 or ids[0] != self._im_end_id:
            raise ValueError(f"Expected '<|im_end|>\\n' to encode as [<|im_end|>, newline], got {ids!r}")
        self._newline_id = int(ids[1])

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted: list[int] = []
        if prefix and prefix[-1] == self._im_end_id:
            inserted = [self._newline_id]                      # model stopped at eos
        elif prefix and prefix[-1] == self._tool_call_end_id:
            inserted = [self._im_end_id, self._newline_id]     # stopped by the tool parser at </tool_call>
        elif prefix:
            # Any other ending (should not happen with the nanbeige parser): close the turn as the template does.
            logger.warning("Nanbeige CT merge: runtime prefix does not end with <|im_end|> or </tool_call>; closing the turn")
            inserted = [self._im_end_id, self._newline_id]
        return MergeResult(
            token_ids=prefix + inserted + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            inserted_token_ids=inserted,
        )
